"""
Hypnosis Audio Builder

A Python library for creating self-hypnosis audio tracks with
binaural beats, ambient music, and subliminal messaging.

Workflow for Self-Hypnosis Creation:
1. Write your script (present tense affirmations)
2. Record your voice
3. Use this tool to add theta waves, ambient, and subliminals
4. Listen daily (morning and night)
"""

from .audio_builder import (
    HypnosisAudioBuilder,
    MixLevels,
    BinauralConfig,
    AudioTrack,
    BinauralBeatGenerator,
    SubliminalProcessor,
    AmbientMusicManager,
    ScriptValidator,
    create_test_voice,
)

__version__ = "1.1.0"
__author__ = "Claude (Anthropic)"

__all__ = [
    "HypnosisAudioBuilder",
    "MixLevels",
    "BinauralConfig",
    "AudioTrack",
    "BinauralBeatGenerator",
    "SubliminalProcessor",
    "AmbientMusicManager",
    "ScriptValidator",
    "create_test_voice",
]
