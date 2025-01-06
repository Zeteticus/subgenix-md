import asyncio
from typing import Optional, List, Tuple, cast, Dict
from loguru import logger
import whisper
import torch
import numpy as np
from pathlib import Path
from .progress_manager import ProgressManager

class AudioTranscriber:
    def __init__(self, progress_manager: ProgressManager, model_name: str = "large-v2"):
        self.progress_manager = progress_manager
        self.model_name = model_name
        self.model = None
        self.british_vocab = {
            "colour": "color",
            "favour": "favor",
            "centre": "center"
        }
        logger.info("AudioTranscriber initialized")

    def load_model(self, use_gpu: bool):
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(self.model_name, device=device)
        self.model.decoder.temperature = 0.7
        return self.model

    async def transcribe_audio(
        self, 
        audio_file: str, 
        language: Optional[str] = "en", 
        use_gpu: bool = True,
        british_english: bool = True
    ) -> List[Tuple[float, float, str]]:
        self.progress_manager.start_task("Loading Whisper model")
        logger.info(f"Loading Whisper model: {self.model_name}")
        
        try:
            model = self.load_model(use_gpu)
            self.progress_manager.complete_task("Model loaded")
            
            processed_audio = self._preprocess_audio(audio_file)
            
            self.progress_manager.start_task("Transcribing audio")
            logger.info(f"Transcribing audio file: {audio_file}")
            
            transcription_options = {
                "language": language,
                "word_timestamps": True,
                "initial_prompt": "This is a British English recording." if british_english else None,
                "task": "transcribe",
                "beam_size": 5,
                "best_of": 5
            }
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: model.transcribe(processed_audio, **transcription_options)
            )
            
            word_timestamps = self._post_process_transcription(
                self._extract_word_timestamps(result),
                british_english=british_english
            )
            
            self.progress_manager.complete_task("Transcription completed")
            logger.info(f"Transcription completed: {len(word_timestamps)} words")
            
            return word_timestamps
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            self.progress_manager.fail_task("Transcription failed")
            raise

    def _preprocess_audio(self, audio_file: str) -> str:
        return audio_file

    def _post_process_transcription(
        self,
        word_timestamps: List[Tuple[float, float, str]],
        british_english: bool = True
    ) -> List[Tuple[float, float, str]]:
        if not british_english:
            return word_timestamps
            
        processed_timestamps = []
        for start, end, text in word_timestamps:
            processed_text = self._apply_british_corrections(text)
            processed_timestamps.append((start, end, processed_text))
            
        return processed_timestamps

    def _apply_british_corrections(self, text: str) -> str:
        text_lower = text.lower()
        
        if text_lower in self.british_vocab:
            return self.british_vocab[text_lower]
            
        return text

    def _extract_word_timestamps(self, result: dict) -> List[Tuple[float, float, str]]:
        word_timestamps = []
        for segment in result["segments"]:
            for word in segment["words"]:
                start = word["start"]
                end = word["end"]
                text = word["word"].strip()
                if text:
                    word_timestamps.append((start, end, text))
        return word_timestamps

    async def detect_language(self, audio_file: str) -> str:
        model = self.load_model(use_gpu=False)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.detect_language(audio_file, sample_len=30)
        )
        return cast(str, result)
