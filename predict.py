import os
import sys
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision import transforms
from typing import List

# Ensure the cloned repo is in the path
# This allows us to import the module even if pip install -e had issues (double safety)
REPO_PATH = '/src/dinov3_repo'
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "/src/checkpoints/model.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Did the build step fail?")

        print(f"Loading DINOv3 model from {model_path} to {self.device}...")
        
        # Load model using torch.hub from local source
        # This respects the repository's intended interface
        try:
            self.model = torch.hub.load(
                REPO_PATH, 
                'dinov3_vitl16', 
                source='local', 
                pretrained=False
            )
        except ImportError as e:
            # Fallback for slight naming variations in the repo
            print(f"Warning: Standard load failed ({e}), trying generic 'vit_large'...")
            try:
                self.model = torch.hub.load(
                    REPO_PATH, 
                    'vit_large', 
                    source='local', 
                    pretrained=False,
                    patch_size=16
                )
            except Exception as inner_e:
                raise RuntimeError(f"Could not load model architecture from {REPO_PATH}. Error: {inner_e}")

        # Load weights
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle "teacher" or "model" keys common in DINO checkpoints
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "teacher" in state_dict:
            state_dict = state_dict["teacher"]
            
        # Strip "module." prefix (DDP artifact)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load state dict
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        self.model.to(self.device)
        
        # Optimize for T4 (and modern GPUs): Use float16
        # This reduces VRAM usage and speeds up inference significantly on T4s
        self.model.half()
        
        self.model.eval()

        # Standard ImageNet preprocessing
        # DINOv2/v3 typically use 518x518 for best performance, or 224x224 for speed
        # We stick to 518 as it's the standard for high-quality features in recent ViTs
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Path = Input(description="Input image")) -> List[float]:
        """Run a single prediction on the model"""
        try:
            img = Image.open(image).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")

        # Convert to tensor, move to device, and cast to half precision (fp16)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device).half()
            
        with torch.no_grad():
            # Run inference
            output = self.model(img_tensor)
            
            # Extract embedding based on return type
            if isinstance(output, dict):
                # Look for CLS token or similar keys
                if "x_norm_clstoken" in output:
                    embedding = output["x_norm_clstoken"]
                else:
                    # Fallback: take the first value
                    embedding = next(iter(output.values()))
            elif isinstance(output, tuple):
                embedding = output[0]
            else:
                embedding = output

        # Flatten and convert to list
        if hasattr(embedding, 'squeeze'):
            embedding = embedding.squeeze()
            
        return embedding.cpu().tolist()
