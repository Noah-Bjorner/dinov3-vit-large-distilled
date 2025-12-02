# DINOv3 ViT-L/16 Cog Model

This is a [Cog](https://github.com/replicate/cog) model for Meta's [DINOv3](https://github.com/facebookresearch/dinov3), specifically the ViT-L/16 distilled version.

## Setup

The model weights are downloaded automatically during the build process from `https://static.peelapi.com/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.

## Usage

### Building the Model

```bash
cog build
```

### Running Predictions

You can run predictions locally using `cog predict`:

```bash
cog predict -i image=@path/to/your/image.jpg
```

This will return the embedding vector (List[float]) for the input image.

### Deploying to Replicate

```bash
cog push r8.im/your-username/your-model-name
```

## Model Details

- **Architecture**: ViT-L/16 (Vision Transformer Large, Patch Size 16)
- **Input**: RGB Image
- **Output**: 1024-dimensional embedding vector (CLS token)
- **Preprocessing**: Resize to 518x518, ImageNet normalization

# dinov3-vit-large-distilled
