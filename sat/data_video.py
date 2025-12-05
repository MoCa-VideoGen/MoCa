import io
import math
import os
import random
import sys
from fractions import Fraction
from functools import partial
from einops import rearrange
from typing import Any, Dict, Optional, Tuple, Union
from packaging import version as pver
import decord
import numpy as np
import torch
import torchvision.transforms as TT
from decord import VideoReader
from sgm.webds import MetaDistributedWebDataset
from torch.utils.data import Dataset
from torchvision.io import _video_opt
from torchvision.io.video import _align_audio_frames, _check_av_available, _read_from_stream, av
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    info = {}
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

    with av.open(filename, metadata_errors="ignore") as container:
        if container.streams.audio:
            audio_timebase = container.streams.audio[0].time_base
        if container.streams.video:
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
            )
            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)

        if container.streams.audio:
            audio_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],
                {"audio": 0},
            )
            info["audio_fps"] = container.streams.audio[0].rate

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


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


def pad_last_frame(tensor, num_frames, selected_frame_indices):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        last_index = selected_frame_indices[-1]
        selected_frame_indices = np.append(selected_frame_indices, [last_index] * pad_length)

        return padded_tensor,selected_frame_indices
    else:
        return tensor[:num_frames], selected_frame_indices[:num_frames]


def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    start = random.randint(skip_frms_num, max_seek + 1)
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(start, end))
    assert temp_frms is not None
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    return pad_last_frame(tensor_frms, num_frames)


import threading


def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    thread = threading.Thread(target=target_function)
    thread.start()
    timeout = 20
    thread.join(timeout)

    if thread.is_alive():
        print("Loading video timed out")
        raise TimeoutError
    return video_container.get("video", None).contiguous()


def process_video(
    video_path,
    image_size=None,
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    """
    video_path: str or io.BytesIO
    image_size: .
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
    num_frames: wanted num_frames.
    wanted_fps: .
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
    """

    video = load_video_with_timeout(
        video_path,
        duration=duration,
        num_frames=num_frames,
        wanted_fps=wanted_fps,
        actual_fps=actual_fps,
        skip_frms_num=skip_frms_num,
        nb_read_frames=nb_read_frames,
    )

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    # resize
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")

    return video


def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:
        r = next(src)
        if "mp4" in r:
            video_data = r["mp4"]
        elif "avi" in r:
            video_data = r["avi"]
        else:
            print("No video data found")
            continue

        if txt_key not in r:
            txt = ""
        else:
            txt = r[txt_key]

        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        else:
            txt = str(txt)

        duration = r.get("duration", None)
        if duration is not None:
            duration = float(duration)
        else:
            continue

        actual_fps = r.get("fps", None)
        if actual_fps is not None:
            actual_fps = float(actual_fps)
        else:
            continue

        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps

        if duration is not None and duration < required_duration:
            continue

        try:
            frames = process_video(
                io.BytesIO(video_data),
                num_frames=num_frames,
                wanted_fps=fps,
                image_size=image_size,
                duration=duration,
                actual_fps=actual_fps,
                skip_frms_num=skip_frms_num,
            )
            frames = (frames - 127.5) / 127.5
        except Exception as e:
            print(e)
            continue

        item = {
            "mp4": frames,
            "txt": txt,
            "num_frames": num_frames,
            "fps": fps,
        }

        yield item

def completely_random_sampling(start, end, num_frames):
  
    available_frames = end - start
    if num_frames >= available_frames:
       
        return np.arange(start, end)
    else:
     
        indices = np.random.choice(range(start, end), size=num_frames, replace=False)
        return np.sort(indices) 

class VideoDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        num_frames,
        fps,
        skip_frms_num=0.0,
        nshards=sys.maxsize,
        seed=1,
        meta_names=None,
        shuffle_buffer=1000,
        include_dirs=None,
        txt_key="caption",
        **kwargs,
    ):
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []

        if path.startswith(";"):
            path, include_dirs = path.split(";", 1)
        super().__init__(
            path,
            partial(
                process_fn_video, num_frames=num_frames, image_size=image_size, fps=fps, skip_frms_num=skip_frms_num
            ),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs,
        )

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(path, **kwargs)

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[0:4]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[6:]).reshape(3, 4)
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
    source_cam_c2w = abs_c2ws[0]
    cam_to_origin = 0
    #cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
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
    # c2w: B, V, 4, 4 ->[B,49,4,4]
    # K: B, V, 4 ->[B,49,4]

    B,V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    #i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    #j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

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
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

class SFTDataset(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset, self).__init__()

        self.video_paths_list = []

        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        decord.bridge.set_bridge("torch")
        print("Walking video dir to get video paths...")
        for root, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".mp4"):
                    self.video_paths_list.append((root, filename))
        print(f"Total number of videos: {len(self.video_paths_list)}")
        self.flag = np.zeros(len(self.video_paths_list), dtype=bool)

    def get_data(self, root, filename):
        try:
            video_size, fps, max_num_frames, skip_frms_num = (
                self.video_size,  #480,720
                self.fps,     # 8 
                self.max_num_frames, #49
                self.skip_frms_num, #3
            )
            video_path = os.path.join(root, filename)  
            
            camera_traj_path = os.path.join(root, filename.replace(".mp4", ".json")).replace("videos", "camera_traj")
            import json
            with open(camera_traj_path, 'r') as f:
                RT = json.load(f)
            
            # print(f"get_data")
            vr = VideoReader(uri=video_path, height=-1, width=-1)
            first_frame = vr[0]  
            original_pose_height, original_pose_width = first_frame.shape[:2]  # (H, W, C) 或 (H, W)
            image_height =video_size[0]
            image_width =video_size[1]

            actual_fps = vr.get_avg_fps() 
            ori_vlen = len(vr)   #49

            #camera_traj_path = os.path.join(root, filename.replace(".mp4", ".json")).replace("videos", "camera_traj")
            #import json
            #with open(camera_traj_path, 'r') as f:
            #    RT = json.load(f)
            RT = np.array(RT)
            RT = torch.tensor(RT).float()
            #RT = RT[:,6:18]
            check = RT.shape[0]
            assert check == ori_vlen
            
            selected_frame_indices = None
            if (ori_vlen - skip_frms_num * 2) / actual_fps * fps > max_num_frames and actual_fps >= fps:
              
                num_frames = max_num_frames
                start = random.randint(skip_frms_num, ori_vlen - skip_frms_num - int(num_frames / fps * actual_fps)) #0
                end = int(start + num_frames / fps * actual_fps) #end:0+49/8*30=180
                end_safty = min(int(start + num_frames / fps * actual_fps), int(ori_vlen))
                #indices = np.arange(start, end, (end - start) // num_frames).astype(int)  #[0,180,4]

                indices = completely_random_sampling(start, end, num_frames).astype(int)
               
                selected_frame_indices = indices
                temp_frms = vr.get_batch(np.arange(start, end_safty)) #[183,720,1280,3]
                assert temp_frms is not None
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:
                if ori_vlen - skip_frms_num * 2 > max_num_frames:
                    num_frames = max_num_frames
                    #start = int(skip_frms_num)
                    start = random.randint(skip_frms_num, ori_vlen - skip_frms_num - max_num_frames)
                    end = int(ori_vlen - skip_frms_num)
                    indices = np.arange(start, end, (end - start) // num_frames).astype(int)
                  
                    temp_frms = vr.get_batch(np.arange(start, end))
                    assert temp_frms is not None
                    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                    selected_frame_indices = indices
                    tensor_frms = tensor_frms[torch.tensor((indices-start).tolist())]  #[61,720,,]
               
                        start = int(skip_frms_num)
                        end = int(ori_vlen - skip_frms_num)
                        num_frames = nearest_smaller_4k_plus_1(
                            end - start
                        )  # 3D VAE requires the number of frames to be 4k+1
                        end = int(start + num_frames)
                        temp_frms = vr.get_batch(np.arange(start, end))
                        assert temp_frms is not None
                        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms

            tensor_frms, selected_frame_indices = pad_last_frame(
                tensor_frms, max_num_frames, selected_frame_indices    #P146
            )  # the len of indices may be less than num_frames, due to round error
            tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            tensor_frms = resize_for_rectangle_crop(tensor_frms, video_size, reshape_mode="center")
            tensor_frms = (tensor_frms - 127.5) / 127.5

            # caption
            caption_path = os.path.join(root, filename.replace(".mp4", ".txt")).replace("videos", "labels")
            if os.path.exists(caption_path):
                caption = open(caption_path, "r").read().splitlines()[0]
            else:
                caption = ""
            
            
            if selected_frame_indices is not None:
                RT = RT[selected_frame_indices,]  #[49,18]
                
            poses = RT.tolist()
            cam_params = [[float(x) for x in pose] for pose in poses]
            cam_params = [Camera(cam_param) for cam_param in cam_params]
     
            
            sample_wh_ratio = image_width / image_height
         
            pose_wh_ratio = original_pose_width / original_pose_height
            #print(f"pose width is {original_pose_width}, height is {original_pose_height}")
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
            device = RT.device
            plucker_embedding = plucker_embedding[None].to(device)  # B V 6 H W
            plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
            RT = plucker_embedding.squeeze(0) 
         
            
            item = {
                "mp4": tensor_frms,
                "txt": caption,
                "num_frames": num_frames,
                "fps": fps,
                "camera_traj":RT,
            }
            return item
        except Exception as e:
            print(f"ERROR when reading video {video_path}, trying to read a valid one. ERROR msg: {e}")
            print("Detailed error trace:")
            import traceback
            print(traceback.format_exc()) 
            return None

    def __getitem__(self, index):
        while True:
            item = self.get_data(*self.video_paths_list[index])
            if item is None:
                index = self._rand_another()
                continue
            else:
                self.flag[index] = True
            return item

    def _rand_another(self):
        pool = np.where(self.flag)[0]
        if len(pool) == 0:
            return np.random.randint(len(self.video_paths_list))
        else:
            return np.random.choice(pool)

    def __len__(self):
        return len(self.video_paths_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
