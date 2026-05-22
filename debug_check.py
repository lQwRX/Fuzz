import torch, os
ckpt = 'model/checkpoints_improved/same_length/class_2/generator_step10.pth'
result = f'exists: {os.path.exists(ckpt)}\n'
if os.path.exists(ckpt):
    ck = torch.load(ckpt, map_location='cpu')
    result += f'vocab_size: {ck["embedding.weight"].shape[0]}\n'
open('debug_speed2.txt', 'w').write(result)