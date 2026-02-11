import torch
import torch.nn as nn
from model_juva import JiT_models


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=6,
            # num_classes removed
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            num_frames=args.num_frames
        )
        self.img_size = args.img_size
        self.num_frames = args.num_frames
        
        # label_drop_prob REUSED as image_drop_prob for training
        self.image_drop_prob = args.label_drop_prob 
        
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def get_model_input(self, z, ref_img):
        # z: (B, 3, T, H, W)
        # ref_img: (B, 3, H, W)
        if ref_img.ndim == 4:
            ref_img = ref_img.unsqueeze(2)
        ref_img_expanded = ref_img.expand(-1, -1, z.shape[2], -1, -1)
        return torch.cat([z, ref_img_expanded], dim=1)

    def forward(self, x, ref_img): # REMOVED labels
        # Visual Condition Dropout for Training
        if self.training and self.image_drop_prob > 0:
            # Mask: 1 means keep, 0 means drop (replace with zeros)
            mask_prob = 1 - self.image_drop_prob
            # Create a mask of shape (B, 1, 1, 1)
            mask = torch.bernoulli(torch.full((ref_img.shape[0], 1, 1, 1), mask_prob, device=ref_img.device))
            ref_img_input = ref_img * mask
        else:
            ref_img_input = ref_img

            
        # Diffusion Forcing: Separate noise each frame
        t=torch.rand((x.shape[0], x.shape[2]), device=x.device)  # (B, T) 
        t_expand=t.view(x.shape[0], 1, x.shape[2], 1, 1)  # (B, 1, T, 1, 1)       
        
        e = torch.randn_like(x) * self.noise_scale

        z = t_expand * x + (1 - t_expand) * e
        v = (x - z) / (1 - t_expand ).clamp_min(self.t_eps)

        model_input = self.get_model_input(z, ref_img_input)

        # Removed labels from net forward
        x_pred = self.net(model_input, t) 
        v_pred = (x_pred - z) / (1 - t_expand).clamp_min(self.t_eps)

        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3, 4)).mean()

        return loss

    @torch.no_grad()
    def generate(self, ref_img): # REMOVED labels
        device = ref_img.device
        bsz = ref_img.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.num_frames, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, ref_img)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], ref_img)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, ref_img):
        # Conditional pass (with reference image)
        model_input_cond = self.get_model_input(z, ref_img)
        x_cond = self.net(model_input_cond, t.flatten())
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # Unconditional pass (with BLACK image)
        ref_img_null = torch.zeros_like(ref_img)
        model_input_uncond = self.get_model_input(z, ref_img_null)
        x_uncond = self.net(model_input_uncond, t.flatten())
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, ref_img):
        v_pred = self._forward_sample(z, t, ref_img)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, ref_img):
        v_pred_t = self._forward_sample(z, t, ref_img)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, ref_img)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next
    
    # update_ema remains same
    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)