import torch
from dit_video_concat import WaveletTransform
# 假设输入具有 requires_grad=True
input_tensor = torch.randn(1, 17550, 3072, requires_grad=True, device='cuda')  
wavelet = WaveletTransform(levels=1).cuda()

output = wavelet(input_tensor)

print("Input requires grad:", input_tensor.requires_grad)  # True
print("Output requires grad:", output.requires_grad)  # 如果为 False，则梯度被截断

input_tensor = torch.randn(1, 17550, 3072, requires_grad=True, device='cuda')
wavelet = WaveletTransform(levels=1).cuda()

output = wavelet(input_tensor)
output.retain_grad()  # 让 output 能够存储梯度

loss = output.mean()  # 简单的损失函数
loss.backward()  # 反向传播

print("fangan2: Output gradient:", output.grad)  # 如果为 None，则梯度被截断

input_tensor = torch.randn(1, 17550, 3072, requires_grad=True, device='cuda')
wavelet = WaveletTransform(levels=1).cuda()

output = wavelet(input_tensor)
gradients = torch.autograd.grad(outputs=output.mean(), inputs=input_tensor, retain_graph=True, allow_unused=True)

print("Gradient w.r.t input:", gradients[0])  # 如果为 None，则梯度被截断


from torch.autograd import gradcheck

wavelet = WaveletTransform(levels=1).double().cuda()
input_tensor = torch.randn(1, 17550, 3072, dtype=torch.double, requires_grad=True, device='cuda')

print("Running gradcheck...")
is_gradcheck_successful = gradcheck(wavelet, (input_tensor,), eps=1e-6, atol=1e-4)
print("Gradcheck success:", is_gradcheck_successful)