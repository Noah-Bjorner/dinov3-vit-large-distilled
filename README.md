# DINOv3 ViT-L/16 Distilled

A production-ready [Cog](https://github.com/replicate/cog) implementation of Meta's **DINOv3 ViT-L/16 Distilled** model for generating high-quality image embeddings.

**200K+ runs on Replicate.**

[Run on Replicate →](https://replicate.com/noah-bjorner/dinov3-vit-large-distilled)

## Overview

This repository packages Meta's [DINOv3](https://github.com/facebookresearch/dinov3) **ViT-L/16 distilled** model for easy deployment with [Cog](https://github.com/replicate/cog) and Replicate.

Given an image, the model returns a **1024-dimensional embedding vector** representing its visual features.

These embeddings can be used for tasks such as:

- Image similarity
- Semantic image search
- Image retrieval
- Clustering
- Deduplication
- Visual recommendation systems
- Downstream computer vision pipelines

## Run with Replicate

The model is publicly hosted on Replicate:

**https://replicate.com/noah-bjorner/dinov3-vit-large-distilled**

Example using the Replicate Python client:

```python
import replicate

output = replicate.run(
    "noah-bjorner/dinov3-vit-large-distilled",
    input={
        "image": "https://example.com/image.jpg"
    }
)

print(output)
```

The output is a 1024-dimensional image embedding:

```python
[
    -1.08203125,
    0.2381591796875,
    0.66064453125,
    ...
]
```

## Run Locally

Install [Cog](https://github.com/replicate/cog), then clone this repository.

Build the model:

```bash
cog build
```

Run a prediction:

```bash
cog predict -i image=@path/to/image.jpg
```

The model weights are downloaded automatically during the build process.

## Model Details

| | |
|---|---|
| **Model** | DINOv3 ViT-L/16 Distilled |
| **Architecture** | Vision Transformer Large (ViT-L/16) |
| **Developer** | Meta |
| **Input** | RGB image |
| **Output** | 1024-dimensional embedding |
| **Feature** | CLS token |
| **Input resolution** | 518 × 518 |
| **Preprocessing** | ImageNet normalization |
| **Training dataset** | LVD-1689M |

## About DINOv3

[DINOv3](https://github.com/facebookresearch/dinov3) is Meta's self-supervised vision model family designed to learn strong general-purpose visual representations without requiring task-specific labels.

The resulting image embeddings can be used as general visual features across a wide range of computer vision applications.

## Deployment

To deploy your own version to Replicate:

```bash
cog push r8.im/your-username/your-model-name
```

## Credits

DINOv3 was developed by Meta.

This repository provides the Cog implementation and Replicate deployment for the ViT-L/16 distilled model.

- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [Replicate deployment](https://replicate.com/noah-bjorner/dinov3-vit-large-distilled)
