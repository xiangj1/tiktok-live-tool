from setuptools import setup, find_packages

setup(
    name="tiktok-live-tool",
    version="0.1.0",
    description="Video speech processing tool with LLM rephrasing",
    author="xiangj1",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "openai-whisper>=20231117",
        "openai>=1.35.0",
        "moviepy>=1.0.3",
        "pydub>=0.25.1",
        "torch>=2.0.1",
        "transformers>=4.33.2",
        "scipy>=1.11.2",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy>=1.24.3",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.7",
        "tqdm>=4.66.1",
    ],
    entry_points={
        "console_scripts": [
            "video-rephrase=video_processor.main:main",
        ],
    },
)