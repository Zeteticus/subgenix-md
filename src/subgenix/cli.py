import asyncio
import click
from loguru import logger
import unicodedata
import re
from typing import Optional

from .audio_extractor import AudioExtractor
from .audio_transcriber import AudioTranscriber
from .subtitle_generator import SubtitleGenerator
from .cache_manager import CacheManager
from .progress_manager import ProgressManager
from .british_english_corrections import british_corrections

class SubtitleGenerator:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for better transcription accuracy."""
        # Remove accent marks
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        # Replace common British English variations
        for british, standard in british_corrections.items():
            text = text.replace(british, standard)
        return text

async def async_main(
    video_file: str, 
    output: Optional[str] = None, 
    language: Optional[str] = 'en-GB',  # Default to British English
    model: str = 'medium',  # Larger model by default
    show_progress: bool = True, 
    structured_logging: bool = False, 
    use_gpu: bool = True
) -> int:
    """Enhanced async main function for subtitle generation."""
    # Configure logging
    log_config = {"serialize": True} if structured_logging else {"format": "{time} {level} {message}"}
    logger.add("subgenix.log", level="INFO", **log_config)

    logger.info(f"Processing video: {video_file}")
    logger.info(f"Using model: {model}")

    try:
        # Initialize components
        cache_manager = CacheManager(video_file)
        progress_manager = ProgressManager(show_progress)
        audio_extractor = AudioExtractor(cache_manager, progress_manager)
        audio_transcriber = AudioTranscriber(progress_manager, model)
        subtitle_generator = SubtitleGenerator(progress_manager)

        # Extract audio with enhanced processing
        logger.info("Extracting audio")
        audio_file, duration = await audio_extractor.process_video(video_file)
        logger.info(f"Audio extracted: {duration:.2f} seconds")

        # Transcribe with British English focus
        logger.info("Transcribing audio")
        word_timestamps = await audio_transcriber.transcribe_audio(
            audio_file, 
            language='en-GB', 
            use_gpu=use_gpu
        )

        # Normalize transcription
        normalized_timestamps = [
            {**wt, 'text': SubtitleGenerator.normalize_text(wt['text'])} 
            for wt in word_timestamps
        ]

        # Generate subtitles
        output_file = output or f"{video_file}.srt"
        srt_file = await subtitle_generator.generate_subtitles(normalized_timestamps, output_file)
        logger.info(f"Subtitles generated: {srt_file}")

    except Exception as e:
        logger.error(f"Subtitle generation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        return 1
    finally:
        await cache_manager.cleanup()

    click.echo("Subtitle generation completed successfully.")
    return 0

@click.command()
@click.argument("video_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output SRT file path")
@click.option("--language", "-l", default='en-GB', help="Language for transcription")
@click.option("--model", "-m", type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']), default='medium')
@click.option("--show-progress/--hide-progress", default=True)
@click.option("--structured-logging", is_flag=True)
@click.option("--use-gpu/--no-gpu", default=True)
def main(video_file, output, language, model, show_progress, structured_logging, use_gpu):
    """Generate subtitles with enhanced accuracy."""
    return asyncio.run(async_main(
        video_file, output, language, model, 
        show_progress, structured_logging, use_gpu
    ))

if __name__ == "__main__":
    main()
