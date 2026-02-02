# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hypnosis Audio Builder (v1.1.0) is a Python audio production tool for creating self-hypnosis tracks. It layers voice recordings with binaural beats (theta, alpha, and beta waves), ambient music, and subliminal messaging.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg  # macOS (required for MP3 support)

# Basic usage
python hypnosis_audio_builder.py --voice script.wav --subliminal-from-voice --output output.mp3

# Session presets (Theta - subconscious access)
python hypnosis_audio_builder.py --voice script.wav --session morning --subliminal-from-voice -o morning.mp3
python hypnosis_audio_builder.py --voice script.wav --session night --subliminal-from-voice -o night.mp3
python hypnosis_audio_builder.py --voice script.wav --session sleep --subliminal-from-voice -o sleep.mp3

# Session presets (Alpha/Beta - focus and active thinking)
python hypnosis_audio_builder.py --voice script.wav --session focus --subliminal-from-voice -o focus.mp3
python hypnosis_audio_builder.py --voice script.wav --session active --subliminal-from-voice -o active.mp3

# Custom brainwave frequency (4-30 Hz range)
python hypnosis_audio_builder.py --voice script.wav --theta-freq 10 --subliminal-from-voice -o alpha.mp3

# Open interactive frequency explorer
python hypnosis_audio_builder.py --open-frequency-guide

# Validate affirmation script
python hypnosis_audio_builder.py --validate-script affirmations.txt

# Test setup with generated audio
python hypnosis_audio_builder.py --test

# List available presets
python hypnosis_audio_builder.py --list-presets

# Batch processing
python hypnosis_audio_builder.py --batch scripts_folder/ --output-dir processed/ --subliminal-from-voice

# Run tests
python -m pytest tests/test_audio_builder.py -v
python tests/test_audio_builder.py  # alternative
```

## Architecture

### Entry Points
- `hypnosis_audio_builder.py` - CLI interface with argparse, YAML/JSON config support, progress display, batch processing
- `src/audio_builder.py` - Core library with all audio processing logic

### Core Classes (src/audio_builder.py)

**HypnosisAudioBuilder** - Main orchestrator
- `build()` - Complete pipeline: load voice, generate binaural, prepare ambient/subliminal, mix, export
- `SESSION_PRESETS` dict - Predefined configurations:
  - Theta (4-8 Hz) - Subconscious access:
    - `morning` - Energizing, theta 6.0 Hz, prominent beats
    - `night` - Relaxing, theta 5.5 Hz, gentler ambient
    - `sleep` - Deep programming, theta 4.0 Hz, very gentle
    - `subliminal` - No audible voice, deep theta
    - `standard` - Balanced defaults
  - Alpha (8-13 Hz) - Relaxed focus:
    - `focus` - Alpha 10 Hz, calm awareness, learning
  - Beta (13-30 Hz) - Active thinking:
    - `active` - Beta 15 Hz, concentration, mental clarity
- `list_presets()` - Returns formatted preset descriptions
- Manages `MixLevels`, `BinauralConfig`, and all track generation

**BinauralBeatGenerator** - Generates stereo binaural beats using sine wave synthesis
- Left ear: base frequency (e.g., 200 Hz)
- Right ear: base + theta frequency (e.g., 206 Hz for 6 Hz theta)
- Applies cosine fade envelope for smooth in/out
- Uses numpy for waveform generation, converts to pydub AudioSegment

**SubliminalProcessor** - Creates subliminal tracks at -35 dB (near-inaudible)
- `create_from_voice()` - Copies voice track, applies low/high-pass EQ, loops to duration (recommended)
- `create_from_audio()` - Loads pre-recorded subliminal file, processes and loops
- `create_from_text()` - TTS generation via pyttsx3 or macOS `say` command

**AmbientMusicManager** - Handles ambient layer (supports WAV, MP3, FLAC, OGG, M4A, AAC)
- `load_track()` - Loads music file, extends via crossfade looping to target duration
- `generate_placeholder()` - Creates pink noise with slow amplitude modulation when no music provided

**ScriptValidator** - Validates affirmation text files for hypnotic best practices
- Checks for negations ("not", "don't"), future tense ("will be"), uncertainty words ("maybe", "hope")
- Scores based on power words, hypnotic language ("imagine", "feel"), sensory descriptions
- Returns validation dict with score, issues, and suggestions

### Data Classes
- `MixLevels` - Volume levels in dB with factory methods: `for_morning()`, `for_night()`, `for_sleep()`, `subliminal_only()`
- `BinauralConfig` - Carrier/brainwave frequency with presets: `for_deep_programming()`, `for_visualization()`, `for_learning()`, `for_relaxation()`, `for_alpha()`, `for_beta()`
- `AudioTrack` - AudioSegment wrapper with name and volume

### Audio Flow
1. Load voice (normalize, convert to stereo)
2. Generate binaural beats (duration = intro + voice + outro)
3. Load/generate ambient music (loop to match duration)
4. Create subliminal from voice (EQ, loop) if enabled
5. Mix: binaural base -> overlay ambient -> overlay subliminal -> overlay voice at intro offset
6. Apply fade in/out, export to MP3/WAV/FLAC

## Key Dependencies
- `pydub` - Audio manipulation and mixing
- `numpy` - Sine wave synthesis for binaural beats
- `pyyaml` - YAML/JSON configuration file support
- `ffmpeg` - Required system dependency for MP3 encoding

## Optional Dependencies
- `pyttsx3` - Offline text-to-speech for subliminal generation from text
