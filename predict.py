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
        log("=" * 50)
        log("SETUP STARTED")
        log("=" * 50)
        
        # Device info
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Device: {self.device}")
        if self.device == "cuda":
            log(f"CUDA device: {torch.cuda.get_device_name(0)}")
            log(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Check model file
        model_path = "/src/checkpoints/model.pth"
        log(f"Checking model file at: {model_path}")
        
        if not os.path.exists(model_path):
            log(f"ERROR: Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}. Did the build step fail?")
        
        file_size = os.path.getsize(model_path) / 1e9
        log(f"Model file found. Size: {file_size:.2f} GB")

        # Create model architecture
        log("Creating ViT-L/16 model architecture using timm...")
        arch_start = time.time()
        
        self.model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            num_classes=0,  # Remove classification head, we want embeddings
            img_size=518,   # DINOv2/v3 typically uses 518x518
        )
        
        log(f"Architecture created in {time.time() - arch_start:.2f}s")
        log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

        # Load weights
        log("Loading checkpoint from disk...")
        load_start = time.time()
        state_dict = torch.load(model_path, map_location='cpu')
        log(f"Checkpoint loaded in {time.time() - load_start:.2f}s")
        log(f"Checkpoint keys: {list(state_dict.keys())[:10]}...")
        
        # Handle nested state dict
        original_keys = len(state_dict)
        if "model" in state_dict:
            log("Found 'model' key in checkpoint, extracting...")
            state_dict = state_dict["model"]
        elif "teacher" in state_dict:
            log("Found 'teacher' key in checkpoint, extracting...")
            state_dict = state_dict["teacher"]
        elif "state_dict" in state_dict:
            log("Found 'state_dict' key in checkpoint, extracting...")
            state_dict = state_dict["state_dict"]
        
        log(f"State dict has {len(state_dict)} keys")
        log(f"First 5 keys: {list(state_dict.keys())[:5]}")
            
        # Strip prefixes
        if any(k.startswith("module.") for k in state_dict.keys()):
            log("Stripping 'module.' prefix (DDP artifact)...")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        if any(k.startswith("backbone.") for k in state_dict.keys()):
            log("Stripping 'backbone.' prefix...")
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        # Log weight shapes for debugging
        log("Sample weight shapes from checkpoint:")
        for i, (k, v) in enumerate(state_dict.items()):
            if i < 5:
                log(f"  {k}: {v.shape}")
        
        # Log expected keys from model
        model_keys = list(self.model.state_dict().keys())
        log(f"Model expects {len(model_keys)} keys")
        log(f"First 5 expected keys: {model_keys[:5]}")
        
        # Load state dict
        log("Loading state dict into model...")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        log(f"Missing keys: {len(missing)}")
        log(f"Unexpected keys: {len(unexpected)}")
        if missing:
            log(f"Missing keys (first 10): {missing[:10]}")
        if unexpected:
            log(f"Unexpected keys (first 10): {unexpected[:10]}")
        
        # Validate that enough weights loaded
        total_params = len(model_keys)
        loaded_params = total_params - len(missing)
        load_percentage = (loaded_params / total_params) * 100
        log(f"Loaded {loaded_params}/{total_params} parameter tensors ({load_percentage:.1f}%)")
        
        if load_percentage < 50:
            log("WARNING: Less than 50% of weights loaded! Model may output garbage.")
            log("This usually means the checkpoint format doesn't match timm's ViT architecture.")
            log("Check if key names in checkpoint match what the model expects.")

        # Move to device
        log(f"Moving model to {self.device}...")
        self.model.to(self.device)
        
        # Convert to FP16
        log("Converting to FP16 for faster inference...")
        self.model.half()
        
        # Set eval mode
        self.model.eval()
        log("Model set to eval mode")
        
        if self.device == "cuda":
            log(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        log("Image transform pipeline created (518x518, ImageNet normalization)")
        
        total_time = time.time() - setup_start
        log("=" * 50)
        log(f"SETUP COMPLETE in {total_time:.2f}s")
        log("=" * 50)

    def predict(self, image: Path = Input(description="Input image")) -> List[float]:
        """Run a single prediction on the model"""
        predict_start = time.time()
        log("-" * 50)
        log(f"PREDICTION STARTED")
        log(f"Input image path: {image}")
        
        # Load image
        log("Loading image...")
        try:
            img = Image.open(image).convert('RGB')
            log(f"Image loaded. Size: {img.size}, Mode: {img.mode}")
        except Exception as e:
            log(f"ERROR: Failed to open image: {e}")
            raise ValueError(f"Failed to open image: {e}")

        # Transform
        log("Applying transforms...")
        transform_start = time.time()
        img_tensor = self.transform(img).unsqueeze(0).to(self.device).half()
        log(f"Transform complete in {time.time() - transform_start:.3f}s")
        log(f"Tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
        
        # Inference
        log("Running inference...")
        inference_start = time.time()
        with torch.no_grad():
            embedding = self.model(img_tensor)
        inference_time = time.time() - inference_start
        log(f"Inference complete in {inference_time:.3f}s")
        log(f"Output shape: {embedding.shape}")

        # Post-process
        embedding = embedding.squeeze()
        result = embedding.cpu().tolist()
        log(f"Embedding dimension: {len(result)}")
        log(f"Embedding range: [{min(result):.4f}, {max(result):.4f}]")
        
        total_time = time.time() - predict_start
        log(f"PREDICTION COMPLETE in {total_time:.3f}s")
        log("-" * 50)
        
        return result
