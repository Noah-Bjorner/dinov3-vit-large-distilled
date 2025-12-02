import os
import time
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision import transforms
from typing import List
import timm


def log(message: str):
    """Print a timestamped log message"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        setup_start = time.time()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Device: {self.device}")
        if self.device == "cuda":
            log(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        
        model_path = "/src/checkpoints/model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        file_size = os.path.getsize(model_path) / 1e9
        log(f"Model file: {file_size:.2f} GB")

        # Create model
        self.model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            num_classes=0,
            img_size=518,
        )

        # Load weights
        state_dict = torch.load(model_path, map_location='cpu')
        
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        if any(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        total_params = len(self.model.state_dict().keys())
        loaded = total_params - len(missing)
        log(f"Weights loaded: {loaded}/{total_params} ({100*loaded/total_params:.0f}%)")

        self.model.to(self.device)
        self.model.half()
        self.model.eval()
        
        if self.device == "cuda":
            log(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        log(f"Setup complete in {time.time() - setup_start:.1f}s")

    def predict(self, image: Path = Input(description="Input image")) -> List[float]:
        """Run a single prediction on the model"""
        start = time.time()
        
        img = Image.open(image).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device).half()
            
        with torch.no_grad():
            embedding = self.model(img_tensor)

        result = embedding.squeeze().cpu().tolist()
        
        log(f"Inference: {time.time() - start:.3f}s | dim: {len(result)}")
        
        return result
