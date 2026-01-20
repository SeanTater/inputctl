import torch
import torch.nn as nn
from torchvision import models
from ..data.keys import NUM_KEYS


class ReflexNet(nn.Module):
    def __init__(
        self,
        context_frames: int = 3,
        num_keys: int = NUM_KEYS,
        inv_dynamics: bool = False,
    ):
        super().__init__()

        # 1. Vision Backbone (ResNet-18)
        # Standard ResNet-18 expects 3 input channels.
        # We have context_frames * 3 channels.
        input_channels = context_frames * 3

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first layer to accept stacking
        # We average the weights of the original first layer across the new channels
        # to preserve pretrained initialization quality.
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize new conv weights
        with torch.no_grad():
            # Old weights: [64, 3, 7, 7]
            # New weights: [64, 9, 7, 7]
            # Copy old weights 3 times? Or divide by 3?
            # Stacking [t, t-1, t-2].
            # A simple strategy is to copy the weights to each time-slice, divided by K
            # so the initial activation magnitude is similar.
            for i in range(context_frames):
                new_conv.weight[:, i * 3 : (i + 1) * 3, :, :] = (
                    old_conv.weight / context_frames
                )

        self.backbone.conv1 = new_conv

        # Remove FC layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        fusion_dim = num_ftrs

        # 4. Heads

        # Policy: Keys (Multi-label)
        self.head_keys = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_keys),  # Logits -> BCEWithLogitsLoss
        )

        # Policy: Mouse (Regression 0-1)
        self.head_mouse = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # x, y
            nn.Sigmoid(),  # Force 0-1 range
        )

        # Value function for RL (predicts expected return from fused state)
        self.head_value = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self.inv_dynamics_enabled = inv_dynamics

        # Auxiliary: Inverse Dynamics
        # Predict Action given State_t, State_{t+1}
        # Input: [Feat_t, Feat_{t+1}] -> size 2*num_ftrs
        self.head_inv_dynamics = None
        if self.inv_dynamics_enabled:
            self.head_inv_dynamics = nn.Sequential(
                nn.Linear(num_ftrs * 2, 256),
                nn.ReLU(),
                nn.Linear(256, num_keys),  # Predict keys held
            )

    def forward(
        self,
        pixel_stack: torch.Tensor,
        next_pixel_stack: torch.Tensor | None = None,
    ):
        """
        Args:
            pixel_stack: (B, C*k, H, W) - current context
            next_pixel_stack: (B, 3, H, W) or (B, C*k, H, W) - optional.
                If 3-channel, we shift the current stack and append the next frame
                to build the t+1 stack for inverse dynamics.
        """

        # Extract Features
        features = self.backbone(pixel_stack)  # (B, 512)

        # Main Heads
        keys_logits = self.head_keys(features)
        mouse_pos = self.head_mouse(features)
        value = self.head_value(features).squeeze(-1)  # (B,) scalar value

        # Auxiliary
        inv_dyn_logits = None

        if self.inv_dynamics_enabled and self.head_inv_dynamics is not None:
            if next_pixel_stack is not None:
                if next_pixel_stack.ndim == 4:
                    if next_pixel_stack.shape[1] == pixel_stack.shape[1]:
                        next_stack = next_pixel_stack
                    else:
                        next_stack = self._stack_next_frame(
                            pixel_stack, next_pixel_stack
                        )
                else:
                    next_stack = self._stack_next_frame(pixel_stack, next_pixel_stack)
                next_features = self.backbone(next_stack)
                inv_dyn_logits = self.head_inv_dynamics(
                    torch.cat([features, next_features], dim=1)
                )

        return keys_logits, mouse_pos, inv_dyn_logits, value

    @staticmethod
    def _stack_next_frame(
        pixel_stack: torch.Tensor, next_frame: torch.Tensor
    ) -> torch.Tensor:
        if next_frame.ndim == 3:
            next_frame = next_frame.unsqueeze(0)
        context_frames = pixel_stack.shape[1] // 3
        if context_frames <= 1:
            return next_frame
        existing = pixel_stack[:, 3:, :, :]
        return torch.cat([existing, next_frame], dim=1)
