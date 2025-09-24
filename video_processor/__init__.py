"""
Video Processor - A tool for extracting, rephrasing, and replacing speech in videos
"""

__version__ = "0.1.0"
__author__ = "xiangj1"

# Import components individually to handle missing dependencies gracefully
try:
    from .pipeline import VideoProcessor, ProcessingConfig
    from .speech_to_text import SpeechToTextProcessor, SpeechSegment
    from .llm_rephraser import LLMRephraser
    from .text_to_speech import TextToSpeechProcessor, AudioSegment
    from .video_editor import VideoEditor
    
    __all__ = [
        "VideoProcessor",
        "ProcessingConfig",
        "SpeechToTextProcessor", 
        "SpeechSegment",
        "LLMRephraser",
        "TextToSpeechProcessor",
        "AudioSegment",
        "VideoEditor"
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Some dependencies not available: {e}. Please install requirements.txt")
    __all__ = []