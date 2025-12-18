"""
MYND Brain - Voice Transcription
=================================
Uses OpenAI Whisper for local speech-to-text.
Runs entirely on your M2 Mac - audio never leaves your machine.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
import tempfile
import os


class VoiceTranscriber:
    """
    Local voice transcription using Whisper.
    Optimized for Apple Silicon via MPS.
    """

    def __init__(
        self,
        model_size: str = "base",  # tiny, base, small, medium, large
        device: torch.device = None
    ):
        """
        Initialize Whisper model.

        Args:
            model_size: Whisper model size
                - tiny: ~39M params, fastest, ~32MB
                - base: ~74M params, good balance, ~142MB (default)
                - small: ~244M params, better accuracy, ~466MB
                - medium: ~769M params, high accuracy, ~1.5GB
                - large: ~1.5B params, best accuracy, ~2.9GB
            device: torch device (mps for M2)
        """
        self.model_size = model_size
        self.device = device or torch.device("cpu")
        self.model = None
        self.initialized = False

        print(f"📦 Loading Whisper {model_size} model...")
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper

            # Load model - will download on first use
            self.model = whisper.load_model(
                self.model_size,
                device=str(self.device) if self.device.type != "mps" else "cpu"
            )

            # Note: Whisper doesn't fully support MPS yet, so we use CPU
            # but MPS is used for other operations that do support it
            self.initialized = True
            print(f"✅ Whisper {self.model_size} loaded")

        except Exception as e:
            print(f"❌ Failed to load Whisper: {e}")
            self.initialized = False

    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        task: str = "transcribe"  # or "translate" (to English)
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes (WAV, MP3, etc.)
            language: Language code (e.g., "en", "es") or None for auto-detect
            task: "transcribe" or "translate"

        Returns:
            Dict with text, language, segments, etc.
        """
        if not self.initialized or self.model is None:
            return {
                "success": False,
                "error": "Whisper model not initialized"
            }

        try:
            # Write audio to temp file (Whisper needs file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                # Transcribe
                result = self.model.transcribe(
                    temp_path,
                    language=language,
                    task=task,
                    fp16=False  # MPS doesn't support fp16 well
                )

                return {
                    "success": True,
                    "text": result["text"].strip(),
                    "language": result.get("language", "unknown"),
                    "segments": [
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"].strip()
                        }
                        for seg in result.get("segments", [])
                    ]
                }

            finally:
                # Clean up temp file
                os.unlink(temp_path)

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def transcribe_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio from a file path."""
        if not self.initialized or self.model is None:
            return {
                "success": False,
                "error": "Whisper model not initialized"
            }

        try:
            result = self.model.transcribe(
                file_path,
                fp16=False,
                **kwargs
            )

            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip()
                    }
                    for seg in result.get("segments", [])
                ]
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_info(self) -> Dict[str, Any]:
        """Get model info."""
        return {
            "model_size": self.model_size,
            "initialized": self.initialized,
            "device": str(self.device)
        }
