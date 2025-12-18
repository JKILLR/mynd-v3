"""
MYND Brain - Vision Understanding
==================================
Uses CLIP for local image understanding.
Generates embeddings and descriptions from images.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import io
import base64


class VisionEngine:
    """
    Local image understanding using CLIP.
    Can generate embeddings and match images to text concepts.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: torch.device = None
    ):
        """
        Initialize CLIP model.

        Args:
            model_name: CLIP model architecture
                - ViT-B-32: Fast, good balance (~400MB)
                - ViT-B-16: Better accuracy (~600MB)
                - ViT-L-14: High accuracy (~1.7GB)
            pretrained: Pretrained weights to use
            device: torch device (mps for M2)
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or torch.device("cpu")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.initialized = False

        print(f"📦 Loading CLIP {model_name} model...")
        self._load_model()

    def _load_model(self):
        """Load the CLIP model."""
        try:
            import open_clip

            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            self.model.eval()  # Set to evaluation mode
            self.initialized = True
            print(f"✅ CLIP {self.model_name} loaded on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load CLIP: {e}")
            self.initialized = False

    def embed_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Generate embedding for an image.

        Args:
            image_data: Raw image bytes (PNG, JPG, etc.)

        Returns:
            Image embedding as numpy array
        """
        if not self.initialized:
            return None

        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error embedding image: {e}")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text (for comparison with images)."""
        if not self.initialized:
            return None

        try:
            text_tokens = self.tokenizer([text]).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_text(text_tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            return embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error embedding text: {e}")
            return None

    def describe_image(
        self,
        image_data: bytes,
        candidate_labels: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate description for an image by matching to concepts.

        Args:
            image_data: Raw image bytes
            candidate_labels: List of possible descriptions to match against
                If None, uses default concept categories
            top_k: Number of top matches to return

        Returns:
            Dict with descriptions and confidence scores
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "CLIP model not initialized"
            }

        # Default concept categories for mind mapping
        if candidate_labels is None:
            candidate_labels = [
                # Objects & Things
                "a diagram or chart",
                "a screenshot of software",
                "a photograph of a person",
                "a photograph of nature",
                "a photograph of a building",
                "a photograph of food",
                "a photograph of an object",
                "handwritten notes",
                "a whiteboard with writing",
                "a book or document",
                # Abstract concepts
                "a visualization of data",
                "an illustration or artwork",
                "a logo or brand",
                "a map or floor plan",
                "a technical drawing",
                "a mind map or concept diagram",
                "a flowchart or process diagram",
                "a photograph of technology",
                "a photograph of travel or location",
                "a photograph of an event or activity"
            ]

        try:
            # Load image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Tokenize labels
            text_tokens = self.tokenizer(candidate_labels).to(self.device)

            # Get embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                scores = similarity[0].cpu().numpy()

            # Get top matches
            top_indices = np.argsort(scores)[::-1][:top_k]
            matches = [
                {
                    "label": candidate_labels[i],
                    "confidence": float(scores[i])
                }
                for i in top_indices
            ]

            # Generate a natural description from top match
            top_match = matches[0]["label"]
            description = self._generate_description(top_match, matches)

            return {
                "success": True,
                "description": description,
                "top_match": top_match,
                "confidence": float(matches[0]["confidence"]),
                "matches": matches
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_description(self, top_match: str, matches: List[Dict]) -> str:
        """Generate a natural language description from matches."""
        # Remove "a " or "an " prefix for cleaner output
        clean_match = top_match
        if clean_match.startswith("a "):
            clean_match = clean_match[2:]
        elif clean_match.startswith("an "):
            clean_match = clean_match[3:]

        # If confidence is high, be direct
        if matches[0]["confidence"] > 0.5:
            return clean_match.capitalize()

        # If uncertain, mention alternatives
        if len(matches) > 1 and matches[1]["confidence"] > 0.2:
            second = matches[1]["label"]
            if second.startswith("a "):
                second = second[2:]
            elif second.startswith("an "):
                second = second[3:]
            return f"{clean_match.capitalize()} (possibly {second})"

        return clean_match.capitalize()

    def find_similar_concepts(
        self,
        image_data: bytes,
        concepts: List[str],
        threshold: float = 0.2
    ) -> List[Tuple[str, float]]:
        """
        Find which concepts from a list match the image.
        Useful for auto-tagging or categorization.

        Args:
            image_data: Raw image bytes
            concepts: List of concept strings to check
            threshold: Minimum similarity to include

        Returns:
            List of (concept, score) tuples above threshold
        """
        if not self.initialized or not concepts:
            return []

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            text_tokens = self.tokenizer(concepts).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T)[0].cpu().numpy()

            matches = [
                (concepts[i], float(similarity[i]))
                for i in range(len(concepts))
                if similarity[i] >= threshold
            ]

            return sorted(matches, key=lambda x: x[1], reverse=True)

        except Exception as e:
            print(f"Error finding similar concepts: {e}")
            return []

    def get_info(self) -> Dict[str, Any]:
        """Get model info."""
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "initialized": self.initialized,
            "device": str(self.device)
        }
