import torch
import torch.nn as nn
from torchvision import models
from ..data.keys import NUM_KEYS
from ..data.intent import INTENTS

class ReflexNet(nn.Module):
    def __init__(self, 
                 context_frames: int = 3,
                 goal_dim: int = 2,
                 num_keys: int = NUM_KEYS):
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
        new_conv = nn.Conv2d(input_channels, old_conv.out_channels, 
                             kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, 
                             padding=old_conv.padding, 
                             bias=old_conv.bias is not None)
        
        # Initialize new conv weights
        with torch.no_grad():
            # Old weights: [64, 3, 7, 7]
            # New weights: [64, 9, 7, 7]
            # Copy old weights 3 times? Or divide by 3?
            # Stacking [t, t-1, t-2].
            # A simple strategy is to copy the weights to each time-slice, divided by K
            # so the initial activation magnitude is similar.
            for i in range(context_frames):
                new_conv.weight[:, i*3:(i+1)*3, :, :] = old_conv.weight / context_frames
                
        self.backbone.conv1 = new_conv
        
        # Remove FC layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 2. Goal Encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. Fusion
        fusion_dim = num_ftrs + 64
        
        # 4. Heads
        
        # Policy: Keys (Multi-label)
        self.head_keys = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_keys) # Logits -> BCEWithLogitsLoss
        )
        
        # Policy: Mouse (Regression 0-1)
        self.head_mouse = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2), # x, y
            nn.Sigmoid() # Force 0-1 range
        )

        # Intent prediction from visual features only (for online use)
        self.head_intent = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, len(INTENTS))
        )
        
        # Auxiliary: Inverse Dynamics
        # Predict Action given State_t, State_{t+1}
        # Input: [Feat_t, Feat_{t+1}] -> size 2*num_ftrs
        # Note: We detach the specific goal? Or include it?
        # Inverse dynamics is pure physics: what action caused change?
        # Goal is irrelevant to physics.
        self.head_inv_dynamics = nn.Sequential(
            nn.Linear(num_ftrs * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_keys) # Predict keys held
        )
        
    def forward(self, 
                pixel_stack: torch.Tensor, 
                goal_vector: torch.Tensor,
                next_pixel_stack: torch.Tensor = None):
        """
        Args:
            pixel_stack: (B, C*k, H, W) - current context
            goal_vector: (B, 2)
            next_pixel_stack: (B, C, H, W) - Optional, for training auxiliary heads.
                              Wait, next stack should be future context?
                              For InvDynamic(t, t+1), we usually compare Feature(t) and Feature(t+1).
                              Feature(t) comes from backbone(stack_t).
                              Feature(t+1) comes from backbone(stack_{t+1}).
                              But stack_{t+1} = [t+1, t, t-1].
                              So we need to run backbone twice if we want full features.
        """
        
        # Extract Features
        features = self.backbone(pixel_stack) # (B, 512)
        
        # Goal Features
        g_feat = self.goal_encoder(goal_vector)
        
        # Fusion
        fused = torch.cat([features, g_feat], dim=1)
        
        # Main Heads
        keys_logits = self.head_keys(fused)
        mouse_pos = self.head_mouse(fused)
        intent_logits = self.head_intent(features)
        
        # Auxiliary
        inv_dyn_logits = None
        
        if next_pixel_stack is not None:
            # We assume next_pixel_stack is the full stack at t+1?
            # Or just the single frame t+1?
            # ResNet needs full stack to generate compatible features?
            # Or at least compatible input shapes.
            # If we pass just frame t+1 (3 chans), we need a separate encoder?
            # Or we padding to 9 chans?
            # Ideally: The dataset should provide stack_{t+1}.
            # Let's assume for now we run backbone on next stack.
            # Warning: expensive (2x backbone passes).
            # Optimization: Freeze backbone for aux pass?
            
            with torch.no_grad(): # Don't update backbone from future features? 
                                  # DREAMER updates both.
                                  # IDM usually updates both.
                                  # Let's allow grad?
                pass
            
            # For simplicity in V1:
            # We don't implement dense InvDyn requiring 2nd pass yet.
            # We focus on the policy first.
            pass
            
        return keys_logits, mouse_pos, inv_dyn_logits, intent_logits
