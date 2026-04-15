"""
glimpse/embedder.py
CLIP-based image embedding pipeline.
"""

import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union


class Embedder:
    """Wraps OpenAI CLIP to embed images into a shared vector space."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.dim = 512  # ViT-B/32 output dimension

    def embed_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Embed a single image. Accepts a file path or PIL Image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

        return features.cpu().numpy().squeeze()

    def embed_batch(self, images: list, batch_size: int = 64) -> np.ndarray:
        """Embed a list of image paths or PIL Images."""
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = []

            for img in batch:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert("RGB")
                tensors.append(self.preprocess(img))

            tensor_batch = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(tensor_batch)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query (enables text-to-image search)."""
        tokens = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().squeeze()
