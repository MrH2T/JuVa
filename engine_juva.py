import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2
from PIL import Image

import util.misc as misc
import util.lr_sched as lr_sched
# import torch_fidelity # Disabled for Video I2V demo as it requires FVD setup
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (video, ref_img) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # x shape: [B, C, T, H, W] (from Video Dataset or Pseudo-Video Wrapper)
        x = video.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        # Extract Reference Image (First Frame): [B, C, H, W]
        ref_img = x[:, :, 0, :, :].clone()
        
        # Ref Img: (B, C, H, W)
        ref = ref_img.to(device, non_blocking=True).to(torch.float32).div_(255)
        ref = ref * 2.0 - 1.0
        
        

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # No labels passed
            loss = model(x, ref)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

def save_video_grid(video_tensor, save_path, grid_size=None):
    """
    video_tensor: (T, H, W, C) numpy array, uint8
    Saves as a 4x4 grid image for 16 frames.
    Forces 3-channel BGR output (No Transparency).
    """
    T, H, W, C = video_tensor.shape
    
    print(f"Saving video grid with shape: {video_tensor.shape}")
    # 强制排成 4x4 网格
    grid_w = int(np.ceil(np.sqrt(T))) 
    grid_h = int(np.ceil(T / grid_w))
    
    # 创建黑色画布，强制为 3 通道 (RGB)
    # 避免 Alpha 通道带来的透明问题
    canvas = np.zeros((H * grid_h, W * grid_w, 3), dtype=np.uint8)
    
    temp_canvas=np.zeros((H, W,3), dtype=np.uint8)
    
    for t in range(T):
        row = t // grid_w
        col = t % grid_w
        
        # 取出当前帧
        frame = video_tensor[t]
        os.makedirs(save_path[:-4]+'/', exist_ok=True)
        temp_path=os.path.join(save_path[:-4]+'/', f"temp_frame_{t}.png")
        
        temp_canvas[:,:,:]=frame[:, :, :3]  # 强制剔除 Alpha 通道（如果存在）
        cv2.imwrite(temp_path, frame[:, :, ::-1])  # BGR to RGB for saving
        
        # 如果输入是 4 通道 (RGBA)，丢弃 Alpha 通道
        if C == 4:
            frame = frame[:, :, :3]
        # 如果输入是 1 通道 (Grayscale)，复制为 3 通道
        elif C == 1:
            frame = np.repeat(frame, 3, axis=-1)
        
        print(np.min(frame), np.max(frame), frame.shape)
            
        canvas[row * H : (row + 1) * H, col * W : (col + 1) * W, :] = frame
    
    # RGB to BGR for cv2
    canvas = canvas[:, :, ::-1]
        
    cv2.imwrite(save_path, canvas)


def evaluate(model_without_ddp, data_loader, args, epoch, batch_size=None, log_writer=None):
    if batch_size is None:
        batch_size = args.batch_size

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    
    num_demo_batches = 2 

    save_folder = os.path.join(
        args.output_dir,
        "epoch{}-{}-steps{}-cfg{}-res{}".format(
            epoch, model_without_ddp.method, model_without_ddp.steps, 
            model_without_ddp.cfg_scale, args.img_size
        )
    )
    
    if misc.get_rank() == 0:
        print("Generating demo videos to:", save_folder)
        os.makedirs(save_folder, exist_ok=True)

    # switch to ema params
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        if name in ema_state_dict:
            ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema for evaluation")
    model_without_ddp.load_state_dict(ema_state_dict)

    if misc.get_rank()==0:
        cnt = 0
        # 修改：直接解包 video, ref_img
        for i, (video, ref_img) in enumerate(data_loader):
            if i >= num_demo_batches:
                break
                
            print(f"Generating batch {i}...")
            
            # 处理 Reference Image
            ref = ref_img.to(args.device).to(torch.float32).div_(255)
            ref = ref * 2.0 - 1.0

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sampled_videos = generate_autoregressive(
                    model_without_ddp,
                    ref,
                    total_frames=4*args.num_frames,  # 4x for longer videos
                    window_size=args.num_frames,
                    stabilization_noise=0.05
                )

            # torch.distributed.barrier()

            sampled_videos = (sampled_videos + 1) / 2
            sampled_videos = sampled_videos.clamp(0, 1)
            sampled_videos = sampled_videos.detach().cpu() # B, 3, T, H, W
            
            for b in range(sampled_videos.size(0)):
                # 1. 保存生成的视频 (Grid)
                vid_np = sampled_videos[b].permute(1, 2, 3, 0).numpy() # (T, H, W, 3)
                vid_np = (vid_np * 255).astype(np.uint8)
                
                # 2. 保存参考图 (Reference)
                # 强制转换为 (H, W, 3) 并处理 BGR 转换
                ref_np = ref_img[b].permute(1, 2, 0).numpy().astype(np.uint8)
                
                # 强制剔除 Alpha 通道（如果存在）
                if ref_np.shape[2] == 4:
                    ref_np = ref_np[:, :, :3]
                elif ref_np.shape[2] == 1:
                    ref_np = np.repeat(ref_np, 3, axis=-1)
                    
                # 保存
                vid_filename = f"batch{i}_idx{b}_generated.png" 
                ref_filename = f"batch{i}_idx{b}_ref.png"
                
                save_video_grid(vid_np, os.path.join(save_folder, vid_filename))
                cv2.imwrite(os.path.join(save_folder, ref_filename), ref_np[:, :, ::-1])

            cnt += 1

    torch.distributed.barrier()

    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    
# In engine_jit.py

def generate_autoregressive(model, ref_img, total_frames=64, window_size=16, stabilization_noise=0.05):
    """
    Performs long-horizon generation using a sliding window.
    
    Args:
        model: The Denoiser/JiT model.
        ref_img: The initial conditioning frame (B, C, H, W).
        total_frames: Total number of frames to generate (e.g., 64).
        window_size: The context size the model was trained on (e.g., 16).
        stabilization_noise: Small noise level added to past frames to prevent divergence 
                             (The "Stabilization" trick from Section 3.3).
    """
    device = ref_img.device
    B, C, H, W = ref_img.shape
    
    # 1. Initialize the sequence with the reference frame
    # We will append generated frames to this list
    generated_frames = [ref_img.unsqueeze(2)] # List of (B, C, 1, H, W)
    
    print(f"Starting autoregressive rollout for {total_frames} frames...")
    
    # 2. Autoregressive Loop
    for i in range(1, total_frames):
        # --- A. Construct Sliding Window Context ---
        # We need a context of length `window_size`. 
        # If we have fewer frames than window_size, we pad with zeros/noise.
        # If we have more, we take the most recent ones (sliding window).
        
        current_len = len(generated_frames)
        start_idx = max(0, current_len - (window_size - 1))
        
        # Stack currently available frames: (B, C, T_avail, H, W)
        context_list = generated_frames[start_idx:]
        context_stack = torch.cat(context_list, dim=2) 
        
        # --- B. Apply Stabilization (Noise Injection) ---
        # The paper suggests adding small noise to past frames so the model 
        # effectively sees "slightly noisy" context, improving stability.
        if stabilization_noise > 0:
            noise = torch.randn_like(context_stack) * stabilization_noise
            context_stack = context_stack + noise
            
        # --- C. Prepare Input for Model ---
        # Pad with pure noise for the "Future" frame we want to predict.
        # We currently have `T_avail` frames. We need `window_size` frames.
        # The last frame in the window is the one we want to generate.
        
        needed_padding = window_size - context_stack.shape[2]
        if needed_padding > 0:
            # This happens at the very beginning (if training allowed variable length)
            # For fixed-length models, we usually usually just repeat or pad zeros.
            # Here we assume we just generate the NEXT token.
            pad = torch.randn(B, C, needed_padding, H, W, device=device)
            model_input = torch.cat([context_stack, pad], dim=2)
        else:
            # Standard case: We have full context, we append 1 noise frame for the target
            # Note: The model input size must match window_size (16).
            # So we take last 15 from context + 1 New Noise Frame.
            
            context_part = context_stack[:, :, -(window_size-1):, :, :]
            target_noise = torch.randn(B, C, 1, H, W, device=device)
            model_input = torch.cat([context_part, target_noise], dim=2)

        # --- D. Run Diffusion Forcing Sampling ---
        # We call the model to denoise *only the last frame* of the window.
        # We assume model.generate_next_token() handles the specific sampling schedule
        # (keeping past fixed, denoising future).
        
        with torch.no_grad():
            # This function needs to be added to Denoiser (see Step 3 below)
            # It returns the fully denoised sequence (B, C, Window, H, W)
            denoised_window = model.generate_next_token(model_input)
            
        # --- E. Extract and Append the New Frame ---
        # The last frame of the output is our new prediction
        new_frame = denoised_window[:, :, -1:, :, :] # (B, C, 1, H, W)
        generated_frames.append(new_frame)
        
        if i % 10 == 0:
            print(f"Generated frame {i}/{total_frames}")

    # Concatenate all frames into a video tensor
    full_video = torch.cat(generated_frames, dim=2) # (B, C, Total, H, W)
    return full_video