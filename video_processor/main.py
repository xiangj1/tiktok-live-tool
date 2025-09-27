"""
Command-line interface for video speech processing
"""
import click
import os
import sys
from pathlib import Path
import logging

# Try to load dotenv, but don't fail if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .pipeline import VideoProcessor, ProcessingConfig

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Video Speech Processing Tool
    
    Process videos by extracting speech, rephrasing with LLM, and replacing audio.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)

@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.option('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--whisper-model', default='base', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model to use for speech recognition')
@click.option('--openai-model', default='gpt-3.5-turbo',
              help='OpenAI model for text rephrasing')
@click.option('--tts-voice', default='alloy',
              type=click.Choice(['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']),
              help='TTS voice to use')
@click.option('--preserve-original', is_flag=True,
              help='Preserve original audio as background')
@click.option('--original-volume', default=0.1, type=float,
              help='Volume level for original audio (0.0-1.0)')
@click.option('--new-volume', default=1.0, type=float,
              help='Volume level for new audio (0.0-1.0)')
@click.option('--max-timing-error', default=1.0, type=float,
              help='Maximum allowed timing error in seconds')
@click.pass_context
def process(ctx, input_video, output_video, api_key, whisper_model, openai_model,
           tts_voice, preserve_original, original_volume, new_volume, max_timing_error):
    """Process a video file with speech rephrasing"""
    
    # Validate API key
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        click.echo("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key", err=True)
        sys.exit(1)
    
    # Validate input file
    input_path = Path(input_video)
    if not input_path.exists():
        click.echo(f"Error: Input video file not found: {input_video}", err=True)
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = ProcessingConfig(
        whisper_model=whisper_model,
        openai_api_key=api_key,
        openai_model=openai_model,
        tts_voice=tts_voice,
        preserve_original_audio=preserve_original,
        original_audio_volume=original_volume,
        new_audio_volume=new_volume,
        max_timing_error=max_timing_error
    )
    
    # Process video
    processor = VideoProcessor(config)
    
    try:
        click.echo(f"Processing video: {input_video}")
        click.echo(f"Output will be saved to: {output_video}")
        
        with click.progressbar(length=7, label='Processing video') as bar:
            # This is a simplified progress bar - in reality, we'd need more sophisticated progress tracking
            result_path = processor.process_video(str(input_path), str(output_path))
            bar.update(7)
        
        click.echo(f"✅ Video processing completed successfully!")
        click.echo(f"📁 Output saved to: {result_path}")
        
    except Exception as e:
        click.echo(f"❌ Error processing video: {e}", err=True)
        sys.exit(1)
    finally:
        processor.cleanup()

@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.option('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--whisper-model', default='base',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model to use for speech recognition')
@click.pass_context
def analyze(ctx, input_video, api_key, whisper_model):
    """Analyze speech in video without processing"""
    
    # Validate API key (needed for potential rephrasing preview)
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    
    config = ProcessingConfig(
        whisper_model=whisper_model,
        openai_api_key=api_key
    )
    
    processor = VideoProcessor(config)
    
    try:
        click.echo(f"Analyzing speech in: {input_video}")
        
        segments = processor.extract_and_analyze_speech(input_video)
        
        if not segments:
            click.echo("No speech detected in the video")
            return
        
        click.echo(f"\n📊 Found {len(segments)} speech segments:")
        click.echo("=" * 80)
        
        total_speech_duration = sum(seg.duration for seg in segments)
        
        for i, segment in enumerate(segments, 1):
            click.echo(f"\n{i:2d}. [{segment.start:6.2f}s - {segment.end:6.2f}s] ({segment.duration:5.2f}s)")
            click.echo(f"    {segment.text}")
        
        click.echo("=" * 80)
        click.echo(f"📈 Total speech time: {total_speech_duration:.2f} seconds")
        
        # Get video info for context
        video_info = processor.video_editor.get_video_info(input_video)
        speech_ratio = total_speech_duration / video_info['duration'] * 100
        click.echo(f"📺 Video duration: {video_info['duration']:.2f} seconds")
        click.echo(f"🎯 Speech coverage: {speech_ratio:.1f}%")
        
    except Exception as e:
        click.echo(f"❌ Error analyzing video: {e}", err=True)
        sys.exit(1)
    finally:
        processor.cleanup()

@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.option('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--whisper-model', default='base',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model to use')
@click.option('--openai-model', default='gpt-3.5-turbo',
              help='OpenAI model for rephrasing')
@click.pass_context
def preview(ctx, input_video, api_key, whisper_model, openai_model):
    """Preview rephrasing without processing video"""
    
    # Validate API key
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        click.echo("Error: OpenAI API key is required for rephrasing preview", err=True)
        sys.exit(1)
    
    config = ProcessingConfig(
        whisper_model=whisper_model,
        openai_api_key=api_key,
        openai_model=openai_model
    )
    
    processor = VideoProcessor(config)
    
    try:
        click.echo(f"Previewing rephrasing for: {input_video}")
        
        # Extract speech
        original_segments = processor.extract_and_analyze_speech(input_video)
        
        if not original_segments:
            click.echo("No speech detected in the video")
            return
        
        # Preview rephrasing for first few segments
        preview_segments = original_segments[:3]  # Limit to avoid API costs
        click.echo(f"\n🔄 Rephrasing preview (first {len(preview_segments)} segments)...")
        
        rephrased_segments = processor.preview_rephrasing(preview_segments)
        
        click.echo("\n📝 Rephrasing Results:")
        click.echo("=" * 100)
        
        for i, (original, rephrased) in enumerate(zip(preview_segments, rephrased_segments), 1):
            click.echo(f"\n{i}. [{original.start:6.2f}s - {original.end:6.2f}s] ({original.duration:5.2f}s)")
            click.echo(f"   Original:  {original.text}")
            click.echo(f"   Rephrased: {rephrased.text}")
        
        if len(original_segments) > len(preview_segments):
            remaining = len(original_segments) - len(preview_segments)
            click.echo(f"\n... and {remaining} more segments would be processed")
        
    except Exception as e:
        click.echo(f"❌ Error previewing rephrasing: {e}", err=True)
        sys.exit(1)
    finally:
        processor.cleanup()

@cli.command()
def version():
    """Show version information"""
    from . import __version__
    click.echo(f"Video Speech Processing Tool v{__version__}")

def main():
    """Entry point for the CLI"""
    cli()

if __name__ == '__main__':
    main()