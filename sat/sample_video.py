import argparse
import gc
import json
from packaging import version as pver
import math
import os
import pickle
from pathlib import Path
from typing import List, Union
import shutil
import cv2

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as TT
from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from einops import rearrange, repeat
from omegaconf import ListConfig
from torchvision.io import write_video
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image
from tqdm import tqdm
from utils.flow_utils import process_traj
from utils.misc import vis_tensor

from sat import mpu
from sat.arguments import set_random_seed
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def draw_points(video, points):
    """
    Draw points onto video frames.

    Parameters:
        video (torch.tensor): Video tensor with shape [T, H, W, C], where T is the number of frames,
                            H is the height, W is the width, and C is the number of channels.
        points (list): Positions of points to be drawn as a tensor with shape [N, T, 2],
                            each point contains x and y coordinates.

    Returns:
        torch.tensor: The video tensor after drawing points, maintaining the same shape [T, H, W, C].
    """

    T = video.shape[0]
    N = len(points)
    device = video.device
    dtype = video.dtype
    video = video.cpu().numpy().copy()
    traj = np.zeros(video.shape[-3:], dtype=np.uint8)  # [H, W, C]
    for n in range(N):
        for t in range(1, T):
            cv2.line(traj, tuple(points[n][t - 1]), tuple(points[n][t]), (255, 1, 1), 2)
    for t in range(T):
        mask = traj[..., -1] > 0
        mask = repeat(mask, "h w -> h w c", c=3)
        alpha = 0.7
        video[t][mask] = video[t][mask] * (1 - alpha) + traj[mask] * alpha
        for n in range(N):
            cv2.circle(video[t], tuple(points[n][t]), 3, (160, 230, 100), -1)
    video = torch.from_numpy(video).to(device, dtype)
    return video

def images_to_video(folder_path, output_video_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))])

   
    if not image_files:
        return

   
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, _ = first_image.shape

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = len(image_files) / 6 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        out.write(img)


    out.release()

    shutil.rmtree(folder_path)
    print(f" {output_video_path}")


def concatenate_videos_horizontally(camera_path=None, indices=None, generated_video=None, p=None, name=None):
    if not camera_path or not generated_video or indices is None or p is None:
        raise ValueError("Can not None")

    
    a = indices[-1].item() 

    cap1 = cv2.VideoCapture(camera_path)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

   
    cap2 = cv2.VideoCapture(generated_video)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
  
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    height = height2
    width1_new = int(width1 * (height2 / height1))

    output_width = width1_new + width2


    total_frames_cap2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    
    output_dir = p / f"{name}_pic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
   
    frame_indices = np.linspace(0, 48, total_frames_cap2, dtype=int)
  
    output_path = p / f"{name}.mp4"
    

    for i in range(total_frames_cap2):
      

        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i])
        ret1, frame1 = cap1.read()

        
        if not ret1:
            print(f"警告: cap1 在第 {frame_indices[i]} 帧读取失败")
            break
        ret2, frame2 = cap2.read()
        if not ret2:
            print("警告: cap2 播放完毕")
            break

       
        frame1 = cv2.resize(frame1, (width1_new, height2))
        concatenated_frame = np.hstack((frame1, frame2))
        frame_filename = f"{output_dir}/{i+1:04d}.png"

        cv2.imwrite(str(frame_filename), concatenated_frame)


   
    cap1.release()
    cap2.release()
    print(f"output_path is {output_path}")
    images_to_video(
        folder_path = output_dir, 
        output_video_path = output_path
    )


    return output_path


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor,
    save_path: str,
    name: str,
    fps: int = 5,
    args=None,
    key=None,
    traj_points=None,
    prompt="",
    camera_path=None,
    indices=None
):
    os.makedirs(save_path, exist_ok=True)
    p = Path(save_path)

    for i, vid in enumerate(video_batch):
        x = rearrange(vid, "t c h w -> t h w c")
        x = x.mul(255).add(0.5).clamp(0, 255).to("cpu", torch.uint8)  # [T H W C]
        os.makedirs(p / "video", exist_ok=True)
        os.makedirs(p / "prompt", exist_ok=True)
        
        if traj_points is not None:
            os.makedirs(p / "traj", exist_ok=True)
            os.makedirs(p / "traj_video", exist_ok=True)
            write_video(
                p / "video" / f"{name}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
            with open(p / "traj" / f"{name}.pkl", "wb") as f:
                pickle.dump(traj_points, f)
            x = draw_points(x, traj_points)
            write_video(
                p / "traj_video" / f"{name}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
        else:
            write_video(
                p / "video" / f"{name}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
        
        if camera_path is not None:
            os.makedirs(p/ "camera_video",exist_ok=True)
            camera_path = os.path.join("./realestate10K/ref_video_49",f"{name}.mp4")
            generated_video= p / "video" / f"{name}.mp4"
            concatenate_videos_horizontally(camera_path,indices,generated_video,p / "camera_video",name)
        
        with open(p / "prompt" / f"{name}_{i:06d}.txt", "w") as f:
            f.write(prompt)

    
def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)
        
def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B,V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def sampling_main(args, model_cls):

    
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    image_size = [480, 720]

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8  #T=13
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device
    
    with torch.no_grad():
      
        json_file_path = "./sat/assets/text/t2v/zhongji/zhongji-dynamic10.json"
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        cnt = -1

        for item in data:
            pose_file = item.get("pose_file").split('/')[-1]
         
            text = item.get("caption")
            clip_name = item.get("clip_name")
            clip_path = item.get("clip_path").split('/')[-1]
    
            pose_file_tem = os.path.join("gt_pose",pose_file)
            if not os.path.isfile(pose_file_tem):
                print(f"Not exist")
                pose_file = os.path.join("RealEstate10K_Txt",pose_file)
            else:
                pose_file = pose_file_tem
            cnt = cnt + 1
            print(f"Generating {text}, Camera is {pose_file}")
            set_random_seed(args.seed)
            if args.flow_from_prompt:
                text, flow_files = text.split("\t")
            total_num_frames = (T - 1) * 4 + 1  # 49
            if args.no_flow_injection:
                video_flow = None
            elif args.flow_from_prompt:
                assert args.flow_path is not None, "Flow path must be provided if flow_from_prompt is True"
                p = os.path.join(args.flow_path, flow_files)
                print(f"Flow path: {p}")
                video_flow = (
                    torch.load(p, map_location="cpu", weights_only=True)[:total_num_frames].unsqueeze_(0).cuda()
                )
            elif args.flow_path:
                print(f"Flow path: {args.flow_path}")
                video_flow = torch.load(args.flow_path, map_location=device, weights_only=True)[
                    :total_num_frames
                ].unsqueeze_(0)
            elif args.point_path:
                if type(args.point_path) == str:
                    args.point_path = json.loads(args.point_path)
                print(f"Point path: {args.point_path}")
                video_flow, points = process_traj(args.point_path, total_num_frames, image_size, device=device)#[49,480,720,2]
               
                video_flow = video_flow.unsqueeze_(0)  #[1,49,480,720,2]
            else:
                print("No flow injection")
                video_flow = None
            
            if pose_file is not None:
                
                with open(pose_file, 'r') as f:
                    poses = f.readlines()
                
                poses = [pose.strip().split(' ') for pose in poses[1:]]
                
                if len(poses) < 2*49: 
                    continue
              
               
                print(f"poses shape is{len(poses)}")
                cam_params = [[float(x) for x in pose] for pose in poses]
                cam_params = [Camera(cam_param) for cam_param in cam_params]
            

              
                image_width = 720
                image_height = 480
                video_path = os.path.join("./realestate10K/ref_video_49",clip_path)
                if not os.path.isfile(video_path):
                    video_path = os.path.join("./realestate10k/train/videos",clip_path)  
                cap = cv2.VideoCapture(video_path)
                original_pose_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
                original_pose_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
                
                print(f"original_pose_height is {original_pose_height},original_pose_width is {original_pose_width}")
                
                sample_wh_ratio = image_width / image_height
        
                pose_wh_ratio = original_pose_width / original_pose_height
            
                if pose_wh_ratio > sample_wh_ratio:
                    resized_ori_w = image_height * pose_wh_ratio
                    for cam_param in cam_params:
                        cam_param.fx = resized_ori_w * cam_param.fx / image_width
                else:
                    resized_ori_h = image_width / pose_wh_ratio
                    for cam_param in cam_params:
                        cam_param.fy = resized_ori_h * cam_param.fy / image_height
                intrinsic = np.asarray([[cam_param.fx * image_width,
                                cam_param.fy * image_height,
                                cam_param.cx * image_width,
                                cam_param.cy * image_height]
                                for cam_param in cam_params], dtype=np.float32)
                
                K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
                c2ws = get_relative_pose(cam_params)
                c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
                plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device='cpu')[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
                
                plucker_embedding = plucker_embedding[None].to(device)  # B V 6 H W
                plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
                RT = plucker_embedding
            
                RT = RT.repeat(2, 1, 1, 1, 1).to(device)
                print(f"RT shape is {RT.shape}")
                
                
                camera_traj = RT
            else:
                camera_traj = None


            if video_flow is not None:
                model.to("cpu")  # move model to cpu, run vae on gpu only.
                tmp = rearrange(video_flow[0], "T H W C -> T C H W") 
                video_flow = flow_to_image(tmp).unsqueeze_(0).to("cuda")  
               
                if args.vis_traj_features:
                    os.makedirs("samples/flow", exist_ok=True)
                    vis_tensor(tmp, *tmp.shape[-2:], "samples/flow/flow1_vis.gif")
                    imageio.mimwrite(
                        "samples/flow/flow2_vis.gif",
                        rearrange(video_flow[0], "T C H W -> T H W C").cpu(),
                        fps=8,
                        loop=0,
                    )
                del tmp
                video_flow = (
                    rearrange(video_flow / 255.0 * 2 - 1, "B T C H W -> B C T H W").contiguous().to(torch.bfloat16)
                )   #[1,3,49,480,720]
                torch.cuda.empty_cache()
                video_flow = video_flow.repeat(2, 1, 1, 1, 1).contiguous()  # for uncondition [2,3,49,480,720]
                model.first_stage_model.to(device)
                video_flow = model.encode_first_stage(video_flow, None)  
                video_flow = video_flow.permute(0, 2, 1, 3, 4).contiguous() 
                model.to(device)

          
            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(  # c['crossattn'].shape = [B,226,4096] 
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            #print(f"c shape is {c.shape}, uc shape is {uc.shape}")
            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
            for index in range(args.batch_size):
                # reload model on GPU
                model.to(device)
                samples_z, sample_z_reference = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    video_flow=video_flow,
                    camera_traj = camera_traj,
                    text = text
                )
                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous() 

                # Unload the model from GPU to save GPU memory
                model.to("cpu")
                torch.cuda.empty_cache()
                first_stage_model = model.first_stage_model
                first_stage_model = first_stage_model.to(device)

                latent = 1.0 / model.scale_factor * samples_z

                # Decode latent serial to save GPU memory
                recons = []
                loop_num = (T - 1) // 2  #6
                for i in range(loop_num): 
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3 
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    with torch.no_grad():
                        recon = first_stage_model.decode( 
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )
                    recons.append(recon)  

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                save_path = args.output_dir
               
                if args.flow_from_prompt:
                    name = Path(flow_files).stem
                indices = torch.arange(49)
                if mpu.get_model_parallel_rank() == 0:
                    save_video_as_grid_and_mp4(
                        samples,
                        save_path,
                        clip_name,
                        fps=args.sampling_fps,
                        traj_points=locals().get("points", None),
                        prompt=text,
                        camera_path = pose_file,
                        indices=indices
                    )

            del samples_z, samples_x, samples, video_flow, latent, recon, recons, c, uc, batch, batch_uc
            gc.collect()


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    args.model_config.en_and_decode_n_samples_a_time = 1

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
