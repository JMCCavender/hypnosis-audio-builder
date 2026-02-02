#!/usr/bin/env python3
"""
Hypnosis Audio Builder - Core Audio Processing Module

Creates self-hypnosis audio tracks with binaural beats, ambient music,
voice narration, and optional subliminal messaging.

Based on the self-hypnosis creation workflow:
1. Script Writing - Present tense affirmations with sensory/emotional language
2. Hypnotic Techniques - Language patterns, embedded commands, trigger words
3. Recording & Layering - Voice + subliminals + theta waves + ambient music
4. Daily Listening - Morning and night repetition for subconscious programming

Author: Claude (Anthropic)
License: MIT
"""

import io
import logging
import math
import re
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter

logger = logging.getLogger(__name__)


@dataclass
class MixLevels:
    """Volume levels for each audio layer in dB."""
    voice: float = 0.0       # Reference level
    binaural: float = -10.0  # Subtle background
    ambient: float = -20.0   # Quiet atmosphere
    subliminal: float = -35.0  # Near-inaudible (beneath conscious hearing)
    
    @classmethod
    def for_morning(cls) -> "MixLevels":
        """Preset for energizing morning sessions - slightly more prominent beats."""
        return cls(voice=0.0, binaural=-8.0, ambient=-18.0, subliminal=-35.0)
    
    @classmethod
    def for_night(cls) -> "MixLevels":
        """Preset for relaxing night sessions - gentler, more ambient."""
        return cls(voice=-2.0, binaural=-12.0, ambient=-15.0, subliminal=-35.0)
    
    @classmethod
    def for_sleep(cls) -> "MixLevels":
        """Preset for sleep/overnight listening - very gentle."""
        return cls(voice=-5.0, binaural=-15.0, ambient=-12.0, subliminal=-35.0)
    
    @classmethod 
    def subliminal_only(cls) -> "MixLevels":
        """Preset for subliminal-only tracks (no audible voice)."""
        return cls(voice=-50.0, binaural=-10.0, ambient=-15.0, subliminal=-30.0)


class ScriptValidator:
    """
    Validates and analyzes affirmation scripts based on self-hypnosis best practices.
    
    Rules for effective scripts:
    - Present tense ("I am" not "I will be")
    - Sensory and descriptive language
    - Emotional words (confident, powerful, calm, magnetic)
    - No negations (avoid "not", "don't", "never", "won't")
    - Absolute statements (no "maybe", "hope", "try", "wish")
    """
    
    # Words/patterns to avoid
    NEGATION_PATTERNS = [
        r"\bnot\b", r"\bdon'?t\b", r"\bwon'?t\b", r"\bcan'?t\b", r"\bnever\b",
        r"\bno\b", r"\bnowhere\b", r"\bnothing\b", r"\bneither\b", r"\bnobody\b",
        r"\bisn'?t\b", r"\baren'?t\b", r"\bwasn'?t\b", r"\bweren'?t\b",
        r"\bshouldn'?t\b", r"\bwouldn'?t\b", r"\bcouldn'?t\b", r"\bmustn'?t\b"
    ]
    
    UNCERTAINTY_WORDS = [
        "maybe", "perhaps", "hope", "hopefully", "wish", "try", "trying",
        "might", "could", "would", "should", "want to", "need to", "have to",
        "someday", "eventually", "soon", "later", "if only"
    ]
    
    FUTURE_PATTERNS = [
        r"\bwill be\b", r"\bwill have\b", r"\bgoing to\b", r"\bwill become\b",
        r"\bI'll be\b", r"\bI'll have\b", r"\bI'll become\b"
    ]
    
    # Positive indicators
    POWER_WORDS = [
        "am", "is", "are", "feel", "know", "believe", "trust", "accept",
        "embrace", "embody", "radiate", "attract", "create", "manifest",
        "confident", "powerful", "calm", "magnetic", "certain", "grounded",
        "abundant", "worthy", "capable", "strong", "peaceful", "present",
        "focused", "clear", "vibrant", "alive", "energized", "relaxed"
    ]
    
    HYPNOTIC_WORDS = [
        "imagine", "notice", "feel", "allow", "become", "experience",
        "sense", "realize", "discover", "remember", "understand", "know",
        "deeply", "completely", "naturally", "effortlessly", "automatically",
        "now", "already", "always", "every", "each"
    ]
    
    SENSORY_WORDS = [
        "see", "hear", "feel", "sense", "touch", "warm", "cool", "light",
        "bright", "soft", "smooth", "flowing", "expanding", "glowing",
        "tingling", "pulsing", "breathing", "floating", "grounded"
    ]
    
    @classmethod
    def validate(cls, text: str) -> dict:
        """
        Validate an affirmation script and return analysis.
        
        Returns dict with:
            - valid: bool - passes basic validation
            - score: int - quality score 0-100
            - issues: list - problems found
            - suggestions: list - improvement suggestions
            - stats: dict - word/pattern counts
        """
        # Filter out comment lines (starting with #) and empty lines
        lines = [l.strip() for l in text.split('\n') 
                 if l.strip() and not l.strip().startswith('#')]
        filtered_text = '\n'.join(lines)
        text_lower = filtered_text.lower()
        
        issues = []
        suggestions = []
        
        # Check for negations
        negations_found = []
        for pattern in cls.NEGATION_PATTERNS:
            matches = re.findall(pattern, text_lower)
            negations_found.extend(matches)
        
        if negations_found:
            issues.append(f"Negations found: {', '.join(set(negations_found))}")
            suggestions.append("Reframe negatively: 'I am not anxious' → 'I am calm and centered'")
        
        # Check for uncertainty words
        uncertainty_found = [w for w in cls.UNCERTAINTY_WORDS if w in text_lower]
        if uncertainty_found:
            issues.append(f"Uncertainty words found: {', '.join(uncertainty_found)}")
            suggestions.append("Use absolute statements: 'I hope to be' → 'I am'")
        
        # Check for future tense
        future_found = []
        for pattern in cls.FUTURE_PATTERNS:
            if re.search(pattern, text_lower):
                future_found.append(pattern.replace(r'\b', '').replace(r'\'?', "'"))
        
        if future_found:
            issues.append(f"Future tense found: {', '.join(future_found)}")
            suggestions.append("Use present tense: 'I will be confident' → 'I am confident'")
        
        # Count positive indicators
        power_count = sum(1 for w in cls.POWER_WORDS if w in text_lower)
        hypnotic_count = sum(1 for w in cls.HYPNOTIC_WORDS if w in text_lower)
        sensory_count = sum(1 for w in cls.SENSORY_WORDS if w in text_lower)
        
        # Suggestions for enhancement
        if hypnotic_count < 3:
            suggestions.append("Add hypnotic language: 'imagine', 'notice', 'feel', 'allow', 'become'")
        
        if sensory_count < 2:
            suggestions.append("Add sensory descriptions: feelings, images, physical sensations")
        
        # Calculate score
        base_score = 70
        
        # Deductions
        base_score -= len(negations_found) * 10
        base_score -= len(uncertainty_found) * 8
        base_score -= len(future_found) * 8
        
        # Bonuses
        base_score += min(power_count * 2, 15)
        base_score += min(hypnotic_count * 3, 10)
        base_score += min(sensory_count * 3, 10)
        
        # Clamp score
        score = max(0, min(100, base_score))
        
        # Determine validity
        valid = len(negations_found) == 0 and len(future_found) == 0
        
        return {
            "valid": valid,
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
            "stats": {
                "lines": len(lines),
                "words": len(filtered_text.split()),
                "power_words": power_count,
                "hypnotic_words": hypnotic_count,
                "sensory_words": sensory_count,
                "negations": len(negations_found),
                "uncertainty": len(uncertainty_found),
                "future_tense": len(future_found)
            }
        }
    
    @classmethod
    def format_report(cls, analysis: dict) -> str:
        """Format validation results as readable report."""
        lines = []
        
        status = "✓ VALID" if analysis["valid"] else "✗ NEEDS REVISION"
        lines.append(f"Script Analysis: {status} (Score: {analysis['score']}/100)")
        lines.append("")
        
        stats = analysis["stats"]
        lines.append(f"Statistics:")
        lines.append(f"  • {stats['lines']} affirmations, {stats['words']} words")
        lines.append(f"  • Power words: {stats['power_words']}")
        lines.append(f"  • Hypnotic language: {stats['hypnotic_words']}")
        lines.append(f"  • Sensory words: {stats['sensory_words']}")
        
        if analysis["issues"]:
            lines.append("")
            lines.append("Issues Found:")
            for issue in analysis["issues"]:
                lines.append(f"  ⚠ {issue}")
        
        if analysis["suggestions"]:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in analysis["suggestions"]:
                lines.append(f"  → {suggestion}")
        
        return "\n".join(lines)


@dataclass
class BinauralConfig:
    """
    Configuration for binaural beat generation.
    
    Binaural beats work by playing slightly different frequencies in each ear.
    The brain perceives the difference as a pulsing beat at the target frequency.
    
    Brainwave Frequency Ranges:
    - Delta (0.5-4 Hz): Deep sleep, healing, unconscious mind
    - Theta (4-8 Hz): Deep meditation, creativity, subconscious access *
    - Alpha (8-13 Hz): Relaxation, light meditation, calm focus
    - Beta (13-30 Hz): Alert, active thinking, concentration
    - Gamma (30-100 Hz): Peak awareness, insight, information processing
    
    * Theta is ideal for self-hypnosis as it opens access to the subconscious.
    """
    base_frequency: float = 200.0  # Hz - carrier frequency (what you "hear")
    theta_frequency: float = 6.0   # Hz - desired brainwave frequency
    sample_rate: int = 44100
    
    # Theta sub-ranges for different purposes (4-8 Hz)
    THETA_DEEP = 4.0      # Deep meditation, near-sleep, subconscious access
    THETA_CREATIVE = 5.5  # Creativity, visualization, insight
    THETA_LEARNING = 6.0  # Learning, memory, programming (default)
    THETA_LIGHT = 7.5     # Light meditation, relaxation, transition to alpha

    # Alpha range (8-13 Hz) - relaxed awareness, calm focus
    ALPHA_CALM = 10.0     # Relaxed awareness, light meditation

    # Beta range (13-30 Hz) - active thinking, concentration
    BETA_FOCUS = 15.0     # Concentration, active thinking
    
    @property
    def left_frequency(self) -> float:
        """Left ear frequency."""
        return self.base_frequency
    
    @property
    def right_frequency(self) -> float:
        """Right ear frequency (slightly higher to create beat)."""
        return self.base_frequency + self.theta_frequency
    
    @classmethod
    def for_deep_programming(cls) -> "BinauralConfig":
        """Deep theta for maximum subconscious access (sleep/night sessions)."""
        return cls(base_frequency=180.0, theta_frequency=cls.THETA_DEEP)
    
    @classmethod
    def for_visualization(cls) -> "BinauralConfig":
        """Creative theta for vivid visualization and imagery."""
        return cls(base_frequency=200.0, theta_frequency=cls.THETA_CREATIVE)
    
    @classmethod
    def for_learning(cls) -> "BinauralConfig":
        """Standard theta for learning and memory (default)."""
        return cls(base_frequency=200.0, theta_frequency=cls.THETA_LEARNING)
    
    @classmethod
    def for_relaxation(cls) -> "BinauralConfig":
        """Light theta transitioning toward alpha for gentle relaxation."""
        return cls(base_frequency=220.0, theta_frequency=cls.THETA_LIGHT)

    @classmethod
    def for_alpha(cls) -> "BinauralConfig":
        """Alpha waves for relaxed awareness and calm focus."""
        return cls(base_frequency=200.0, theta_frequency=cls.ALPHA_CALM)

    @classmethod
    def for_beta(cls) -> "BinauralConfig":
        """Beta waves for concentration and active focus."""
        return cls(base_frequency=200.0, theta_frequency=cls.BETA_FOCUS)


@dataclass
class AudioTrack:
    """Represents a single audio track with metadata."""
    audio: AudioSegment
    name: str
    volume_db: float = 0.0
    
    @property
    def duration_ms(self) -> int:
        return len(self.audio)
    
    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


class BinauralBeatGenerator:
    """Generates binaural beats using sine wave synthesis."""
    
    def __init__(self, config: BinauralConfig):
        self.config = config
    
    def generate(
        self, 
        duration_seconds: float,
        fade_in_seconds: float = 3.0,
        fade_out_seconds: float = 3.0,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> AudioSegment:
        """
        Generate stereo binaural beat audio.
        
        Args:
            duration_seconds: Length of the audio in seconds
            fade_in_seconds: Duration of fade-in effect
            fade_out_seconds: Duration of fade-out effect
            progress_callback: Optional callback for progress updates (0.0-1.0)
        
        Returns:
            AudioSegment with binaural beats
        """
        logger.info(
            f"Generating binaural beats: {duration_seconds:.1f}s, "
            f"base={self.config.base_frequency}Hz, "
            f"theta={self.config.theta_frequency}Hz"
        )
        
        sample_rate = self.config.sample_rate
        num_samples = int(duration_seconds * sample_rate)
        
        # Time array
        t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
        
        # Generate sine waves for left and right channels
        left_freq = self.config.left_frequency
        right_freq = self.config.right_frequency
        
        if progress_callback:
            progress_callback(0.2)
        
        # Generate left channel (base frequency)
        left_channel = np.sin(2 * np.pi * left_freq * t)
        
        if progress_callback:
            progress_callback(0.4)
        
        # Generate right channel (base + theta frequency)
        right_channel = np.sin(2 * np.pi * right_freq * t)
        
        if progress_callback:
            progress_callback(0.6)
        
        # Apply fade in/out envelope
        envelope = self._create_envelope(
            num_samples, 
            fade_in_seconds, 
            fade_out_seconds, 
            sample_rate
        )
        
        left_channel *= envelope
        right_channel *= envelope
        
        if progress_callback:
            progress_callback(0.8)
        
        # Convert to 16-bit integer samples
        left_int = (left_channel * 32767 * 0.8).astype(np.int16)
        right_int = (right_channel * 32767 * 0.8).astype(np.int16)
        
        # Interleave stereo channels
        stereo = np.empty(num_samples * 2, dtype=np.int16)
        stereo[0::2] = left_int
        stereo[1::2] = right_int
        
        # Create WAV in memory
        audio = self._numpy_to_audiosegment(stereo, sample_rate, channels=2)
        
        if progress_callback:
            progress_callback(1.0)
        
        logger.info("Binaural beat generation complete")
        return audio
    
    def _create_envelope(
        self, 
        num_samples: int, 
        fade_in_seconds: float, 
        fade_out_seconds: float,
        sample_rate: int
    ) -> np.ndarray:
        """Create amplitude envelope with fade in/out."""
        envelope = np.ones(num_samples, dtype=np.float32)
        
        # Calculate fade samples, capping at half the total length each
        max_fade_samples = num_samples // 2
        
        # Fade in
        fade_in_samples = min(int(fade_in_seconds * sample_rate), max_fade_samples)
        if fade_in_samples > 0:
            fade_in = np.linspace(0, 1, fade_in_samples, dtype=np.float32)
            # Apply smooth (cosine) curve
            fade_in = (1 - np.cos(fade_in * np.pi)) / 2
            envelope[:fade_in_samples] = fade_in
        
        # Fade out
        fade_out_samples = min(int(fade_out_seconds * sample_rate), max_fade_samples)
        if fade_out_samples > 0:
            fade_out = np.linspace(1, 0, fade_out_samples, dtype=np.float32)
            # Apply smooth (cosine) curve
            fade_out = (1 + np.cos((1 - fade_out) * np.pi)) / 2
            envelope[-fade_out_samples:] = fade_out
        
        return envelope
    
    def _numpy_to_audiosegment(
        self, 
        samples: np.ndarray, 
        sample_rate: int, 
        channels: int = 2
    ) -> AudioSegment:
        """Convert numpy array to AudioSegment."""
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
        
        buffer.seek(0)
        return AudioSegment.from_wav(buffer)


class SubliminalProcessor:
    """
    Processes affirmations into subliminal audio tracks.
    
    Subliminal messages are audio content played at volumes beneath 
    conscious hearing (-30 to -40 dB). The subconscious mind can still 
    process this information, especially when combined with theta waves 
    that open access to subconscious programming.
    
    Best Practice: Use the SAME affirmations as your voice script.
    The voice track delivers to conscious mind, subliminal to subconscious.
    """
    
    # Subliminal audio processing parameters
    SUBLIMINAL_LOW_PASS = 4000   # Hz - gentle roll-off for softer sound
    SUBLIMINAL_HIGH_PASS = 100   # Hz - remove rumble
    
    def __init__(self, voice_path: Optional[Path] = None):
        """
        Initialize subliminal processor.
        
        Args:
            voice_path: Optional path to voice file to use as subliminal base
        """
        self.voice_path = voice_path
    
    def create_from_voice(
        self,
        voice_audio: AudioSegment,
        target_duration_seconds: float,
        speed_factor: float = 1.0,
        repetition_gap_ms: int = 500
    ) -> AudioSegment:
        """
        Create subliminal track from the voice recording itself.
        
        This is the recommended approach - same affirmations delivered
        to both conscious (voice) and subconscious (subliminal) mind.
        
        Args:
            voice_audio: The main voice recording
            target_duration_seconds: Target duration for subliminal track
            speed_factor: Speed adjustment (1.0 = normal, 1.2 = 20% faster)
            repetition_gap_ms: Gap between repetitions in ms
        
        Returns:
            AudioSegment with processed subliminal audio
        """
        logger.info("Creating subliminal track from voice recording")
        
        # Process the audio for subliminal use
        subliminal = voice_audio
        
        # Optional speed adjustment (some prefer slightly faster subliminals)
        if speed_factor != 1.0:
            # Change speed without changing pitch
            new_frame_rate = int(subliminal.frame_rate * speed_factor)
            subliminal = subliminal._spawn(
                subliminal.raw_data,
                overrides={"frame_rate": new_frame_rate}
            ).set_frame_rate(voice_audio.frame_rate)
        
        # Apply EQ for subliminal characteristics
        subliminal = low_pass_filter(subliminal, self.SUBLIMINAL_LOW_PASS)
        subliminal = high_pass_filter(subliminal, self.SUBLIMINAL_HIGH_PASS)
        
        # Add gap between repetitions
        gap = AudioSegment.silent(duration=repetition_gap_ms)
        subliminal_with_gap = subliminal + gap
        
        # Loop to fill duration
        target_ms = int(target_duration_seconds * 1000)
        if len(subliminal_with_gap) < target_ms:
            loops_needed = math.ceil(target_ms / len(subliminal_with_gap))
            subliminal = subliminal_with_gap * loops_needed
        
        # Trim to exact duration
        subliminal = subliminal[:target_ms]
        
        return subliminal
    
    def create_from_text(
        self,
        text: str,
        duration_seconds: float,
        sample_rate: int = 44100,
        use_tts: bool = False,
        tts_engine: str = "pyttsx3"
    ) -> Optional[AudioSegment]:
        """
        Create subliminal audio from text affirmations.
        
        Args:
            text: Affirmation text (one affirmation per line)
            duration_seconds: Target duration
            sample_rate: Sample rate for generated audio
            use_tts: Whether to attempt TTS generation
            tts_engine: TTS engine to use ("pyttsx3", "gtts", "say")
        
        Returns:
            AudioSegment with subliminal affirmations, or None if unavailable
        """
        if not use_tts:
            logger.warning(
                "TTS not enabled for subliminal generation.\n"
                "Options:\n"
                "1. Record affirmations as audio file and use --subliminal\n"
                "2. Use --subliminal-from-voice to copy main voice track\n"
                "3. Install pyttsx3: pip install pyttsx3"
            )
            return None
        
        # Try TTS generation
        try:
            return self._generate_tts(text, duration_seconds, tts_engine, sample_rate)
        except ImportError as e:
            logger.warning(f"TTS engine not available: {e}")
            return None
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    def _generate_tts(
        self,
        text: str,
        duration_seconds: float,
        engine: str,
        sample_rate: int
    ) -> Optional[AudioSegment]:
        """Generate TTS audio using specified engine."""
        import tempfile
        
        if engine == "pyttsx3":
            try:
                import pyttsx3
                tts = pyttsx3.init()
                
                # Slower rate for subliminals
                tts.setProperty('rate', 120)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                
                tts.save_to_file(text, temp_path)
                tts.runAndWait()
                
                audio = AudioSegment.from_wav(temp_path)
                Path(temp_path).unlink()
                
                return self._process_for_subliminal(audio, duration_seconds)
                
            except ImportError:
                raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")
        
        elif engine == "say":
            # macOS built-in TTS
            import subprocess
            with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
                temp_path = f.name
            
            subprocess.run(
                ['say', '-o', temp_path, text],
                check=True,
                capture_output=True
            )
            
            audio = AudioSegment.from_file(temp_path)
            Path(temp_path).unlink()
            
            return self._process_for_subliminal(audio, duration_seconds)
        
        else:
            raise ValueError(f"Unknown TTS engine: {engine}")
    
    def _process_for_subliminal(
        self,
        audio: AudioSegment,
        target_duration_seconds: float
    ) -> AudioSegment:
        """Apply subliminal processing to audio."""
        # Apply EQ
        audio = low_pass_filter(audio, self.SUBLIMINAL_LOW_PASS)
        audio = high_pass_filter(audio, self.SUBLIMINAL_HIGH_PASS)
        
        # Loop to fill duration
        target_ms = int(target_duration_seconds * 1000)
        gap = AudioSegment.silent(duration=500)
        audio_with_gap = audio + gap
        
        if len(audio_with_gap) < target_ms:
            loops_needed = math.ceil(target_ms / len(audio_with_gap))
            audio = audio_with_gap * loops_needed
        
        return audio[:target_ms]
    
    def create_from_audio(
        self,
        audio_path: Path,
        target_duration_seconds: float
    ) -> AudioSegment:
        """
        Create subliminal track from pre-recorded audio.
        
        Args:
            audio_path: Path to recorded affirmations audio
            target_duration_seconds: Target duration (will loop if needed)
        
        Returns:
            AudioSegment looped to fill duration
        """
        logger.info(f"Loading subliminal audio from: {audio_path}")
        audio = AudioSegment.from_file(str(audio_path))
        
        return self._process_for_subliminal(audio, target_duration_seconds)


class AmbientMusicManager:
    """Manages ambient music tracks."""
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    def __init__(self, music_directory: Optional[Path] = None):
        """
        Initialize ambient music manager.
        
        Args:
            music_directory: Optional directory containing ambient music files
        """
        self.music_directory = music_directory
        self._available_tracks: list[Path] = []
        
        if music_directory and music_directory.exists():
            self._scan_directory()
    
    def _scan_directory(self):
        """Scan music directory for available tracks."""
        if not self.music_directory:
            return
        
        self._available_tracks = [
            f for f in self.music_directory.iterdir()
            if f.suffix.lower() in self.SUPPORTED_FORMATS
        ]
        
        logger.info(f"Found {len(self._available_tracks)} ambient music tracks")
    
    def load_track(
        self,
        track_path: Path,
        target_duration_seconds: Optional[float] = None,
        crossfade_seconds: float = 5.0
    ) -> AudioSegment:
        """
        Load and prepare ambient music track.
        
        Args:
            track_path: Path to music file
            target_duration_seconds: Optional target duration (loops if needed)
            crossfade_seconds: Crossfade duration when looping
        
        Returns:
            Prepared AudioSegment
        """
        logger.info(f"Loading ambient track: {track_path}")
        
        if not track_path.exists():
            raise FileNotFoundError(f"Ambient music file not found: {track_path}")
        
        audio = AudioSegment.from_file(str(track_path))
        
        if target_duration_seconds:
            audio = self._extend_to_duration(
                audio, 
                target_duration_seconds, 
                crossfade_seconds
            )
        
        return audio
    
    def _extend_to_duration(
        self,
        audio: AudioSegment,
        target_seconds: float,
        crossfade_seconds: float
    ) -> AudioSegment:
        """Extend audio to target duration using crossfade looping."""
        target_ms = int(target_seconds * 1000)
        crossfade_ms = int(crossfade_seconds * 1000)
        
        if len(audio) >= target_ms:
            return audio[:target_ms]
        
        # Build up by crossfade looping
        result = audio
        while len(result) < target_ms:
            # Crossfade append
            result = result.append(audio, crossfade=crossfade_ms)
        
        # Trim to exact duration with fade out
        result = result[:target_ms]
        result = result.fade_out(duration=crossfade_ms)
        
        return result
    
    def generate_placeholder(
        self,
        duration_seconds: float,
        sample_rate: int = 44100
    ) -> AudioSegment:
        """
        Generate placeholder ambient track (pink noise with gentle modulation).
        
        This provides a basic ambient backdrop when no music file is provided.
        For production use, real ambient music is recommended.
        
        Args:
            duration_seconds: Duration of ambient track
            sample_rate: Sample rate
        
        Returns:
            AudioSegment with generated ambient sound
        """
        logger.info(f"Generating placeholder ambient track: {duration_seconds:.1f}s")
        
        num_samples = int(duration_seconds * sample_rate)
        
        # Generate pink noise using Voss-McCartney algorithm (simplified)
        white_noise = np.random.randn(num_samples).astype(np.float32)
        
        # Simple pink noise approximation using cascaded first-order filters
        pink_noise = self._white_to_pink(white_noise)
        
        # Apply slow amplitude modulation for movement
        t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
        modulation = 0.8 + 0.2 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz modulation
        pink_noise *= modulation
        
        # Normalize and convert to stereo
        pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.3
        
        # Create gentle stereo effect
        left = pink_noise * (0.9 + 0.1 * np.sin(2 * np.pi * 0.02 * t))
        right = pink_noise * (0.9 + 0.1 * np.sin(2 * np.pi * 0.02 * t + np.pi/4))
        
        # Convert to 16-bit
        left_int = (left * 32767).astype(np.int16)
        right_int = (right * 32767).astype(np.int16)
        
        # Interleave
        stereo = np.empty(num_samples * 2, dtype=np.int16)
        stereo[0::2] = left_int
        stereo[1::2] = right_int
        
        # Convert to AudioSegment
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(stereo.tobytes())
        
        buffer.seek(0)
        audio = AudioSegment.from_wav(buffer)
        
        # Apply low-pass filter for smoother sound
        audio = low_pass_filter(audio, 2000)
        
        return audio
    
    def _white_to_pink(self, white: np.ndarray) -> np.ndarray:
        """Convert white noise to pink noise using simple filtering."""
        # Simple IIR filter approximation
        pink = np.zeros_like(white)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        
        # Apply filter (simplified - scipy would be better)
        for i in range(len(white)):
            pink[i] = (
                b[0] * white[i] +
                (b[1] * white[i-1] if i > 0 else 0) +
                (b[2] * white[i-2] if i > 1 else 0) +
                (b[3] * white[i-3] if i > 2 else 0) -
                (a[1] * pink[i-1] if i > 0 else 0) -
                (a[2] * pink[i-2] if i > 1 else 0) -
                (a[3] * pink[i-3] if i > 2 else 0)
            )
        
        return pink


class HypnosisAudioBuilder:
    """
    Main class for building complete hypnosis audio tracks.
    
    Orchestrates the mixing of voice, binaural beats, ambient music,
    and optional subliminal messages into a cohesive audio production.
    
    Workflow for Self-Hypnosis Audio Creation:
    1. Record your affirmation script (present tense, sensory, emotional)
    2. Run through this builder to add theta waves and ambient
    3. Optionally create subliminal layer from the same script
    4. Export and listen daily (morning and night)
    
    The theta waves open access to the subconscious mind.
    The voice delivers affirmations to conscious awareness.
    The subliminal reinforces to the subconscious beneath awareness.
    Repetition over time rewires default thoughts and responses.
    """
    
    # Session type presets
    SESSION_PRESETS = {
        "morning": {
            "mix": MixLevels.for_morning(),
            "binaural": BinauralConfig.for_learning(),
            "intro_silence": 3.0,
            "outro_silence": 5.0,
            "description": "Energizing morning session for starting the day"
        },
        "night": {
            "mix": MixLevels.for_night(),
            "binaural": BinauralConfig.for_visualization(),
            "intro_silence": 5.0,
            "outro_silence": 8.0,
            "description": "Relaxing evening session for winding down"
        },
        "sleep": {
            "mix": MixLevels.for_sleep(),
            "binaural": BinauralConfig.for_deep_programming(),
            "intro_silence": 10.0,
            "outro_silence": 15.0,
            "description": "Gentle overnight session for deep programming"
        },
        "subliminal": {
            "mix": MixLevels.subliminal_only(),
            "binaural": BinauralConfig.for_deep_programming(),
            "intro_silence": 2.0,
            "outro_silence": 2.0,
            "description": "Subliminal-only track (no audible voice)"
        },
        "standard": {
            "mix": MixLevels(),
            "binaural": BinauralConfig.for_learning(),
            "intro_silence": 3.0,
            "outro_silence": 5.0,
            "description": "Standard balanced session"
        },
        "focus": {
            "mix": MixLevels.for_morning(),
            "binaural": BinauralConfig.for_alpha(),
            "intro_silence": 2.0,
            "outro_silence": 3.0,
            "description": "Alpha waves (10 Hz) for relaxed focus and calm awareness"
        },
        "active": {
            "mix": MixLevels.for_morning(),
            "binaural": BinauralConfig.for_beta(),
            "intro_silence": 2.0,
            "outro_silence": 3.0,
            "description": "Beta waves (15 Hz) for concentration and active thinking"
        }
    }
    
    def __init__(
        self,
        mix_levels: Optional[MixLevels] = None,
        sample_rate: int = 44100,
        session_type: str = "standard"
    ):
        """
        Initialize the audio builder.
        
        Args:
            mix_levels: Volume levels for each layer (overrides session preset)
            sample_rate: Target sample rate for output
            session_type: Preset type ("morning", "night", "sleep", "subliminal", "standard")
        """
        # Get preset defaults
        preset = self.SESSION_PRESETS.get(session_type, self.SESSION_PRESETS["standard"])
        
        self.mix_levels = mix_levels or preset["mix"]
        self.sample_rate = sample_rate
        self.session_type = session_type
        self._preset = preset
        
        self.binaural_generator = BinauralBeatGenerator(preset["binaural"])
        self.ambient_manager = AmbientMusicManager()
        self.subliminal_processor = SubliminalProcessor()
        self.script_validator = ScriptValidator()
        
        self._tracks: list[AudioTrack] = []
    
    @classmethod
    def list_presets(cls) -> str:
        """Return formatted list of available session presets."""
        lines = ["Available Session Presets:", ""]

        # Group presets by brainwave type for clarity
        theta_presets = ["morning", "night", "sleep", "subliminal", "standard"]
        alpha_presets = ["focus"]
        beta_presets = ["active"]

        lines.append("  THETA (4-8 Hz) - Subconscious access, meditation, programming")
        lines.append("  " + "-" * 50)
        for name in theta_presets:
            preset = cls.SESSION_PRESETS[name]
            lines.append(f"  {name}:")
            lines.append(f"    {preset['description']}")
            lines.append(f"    Frequency: {preset['binaural'].theta_frequency} Hz")
            lines.append(f"    Voice: {preset['mix'].voice} dB, Binaural: {preset['mix'].binaural} dB")
            lines.append("")

        lines.append("  ALPHA (8-13 Hz) - Relaxed awareness, calm focus, learning")
        lines.append("  " + "-" * 50)
        for name in alpha_presets:
            preset = cls.SESSION_PRESETS[name]
            lines.append(f"  {name}:")
            lines.append(f"    {preset['description']}")
            lines.append(f"    Frequency: {preset['binaural'].theta_frequency} Hz")
            lines.append(f"    Voice: {preset['mix'].voice} dB, Binaural: {preset['mix'].binaural} dB")
            lines.append("")

        lines.append("  BETA (13-30 Hz) - Active thinking, concentration, alertness")
        lines.append("  " + "-" * 50)
        for name in beta_presets:
            preset = cls.SESSION_PRESETS[name]
            lines.append(f"  {name}:")
            lines.append(f"    {preset['description']}")
            lines.append(f"    Frequency: {preset['binaural'].theta_frequency} Hz")
            lines.append(f"    Voice: {preset['mix'].voice} dB, Binaural: {preset['mix'].binaural} dB")
            lines.append("")

        return "\n".join(lines)
    
    def set_binaural_config(self, config: BinauralConfig):
        """Configure binaural beat parameters."""
        self.binaural_generator.config = config
    
    def validate_script(self, script_path: Path) -> dict:
        """
        Validate an affirmation script file.
        
        Args:
            script_path: Path to text file with affirmations
        
        Returns:
            Validation analysis dict
        """
        text = script_path.read_text()
        return self.script_validator.validate(text)
    
    def load_voice(
        self,
        voice_path: Path,
        normalize_audio: bool = True
    ) -> AudioTrack:
        """
        Load the main voice narration track.
        
        Args:
            voice_path: Path to voice recording
            normalize_audio: Whether to normalize volume
        
        Returns:
            AudioTrack object
        """
        logger.info(f"Loading voice track: {voice_path}")
        
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
        audio = AudioSegment.from_file(str(voice_path))
        
        # Convert to stereo if mono
        if audio.channels == 1:
            audio = audio.set_channels(2)
        
        # Normalize
        if normalize_audio:
            audio = normalize(audio)
        
        track = AudioTrack(
            audio=audio,
            name="voice",
            volume_db=self.mix_levels.voice
        )
        
        logger.info(f"Voice track loaded: {track.duration_seconds:.1f}s")
        return track
    
    def generate_binaural_track(
        self,
        duration_seconds: float,
        theta_frequency: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> AudioTrack:
        """
        Generate binaural beats track.
        
        Args:
            duration_seconds: Duration of the track
            theta_frequency: Desired brainwave frequency (overrides config)
            progress_callback: Optional progress callback
        
        Returns:
            AudioTrack object
        """
        # Update config if frequency specified
        if theta_frequency is not None:
            self.binaural_generator.config.theta_frequency = theta_frequency
        self.binaural_generator.config.sample_rate = self.sample_rate
        
        audio = self.binaural_generator.generate(
            duration_seconds,
            progress_callback=progress_callback
        )
        
        return AudioTrack(
            audio=audio,
            name="binaural",
            volume_db=self.mix_levels.binaural
        )
    
    def load_ambient_track(
        self,
        ambient_path: Optional[Path],
        duration_seconds: float,
        use_placeholder: bool = True
    ) -> AudioTrack:
        """
        Load or generate ambient music track.
        
        Args:
            ambient_path: Path to ambient music file (or None for placeholder)
            duration_seconds: Target duration
            use_placeholder: Generate placeholder if no file provided
        
        Returns:
            AudioTrack object
        """
        if ambient_path and ambient_path.exists():
            audio = self.ambient_manager.load_track(
                ambient_path,
                target_duration_seconds=duration_seconds
            )
        elif use_placeholder:
            logger.info("No ambient music provided, generating placeholder")
            audio = self.ambient_manager.generate_placeholder(
                duration_seconds,
                sample_rate=self.sample_rate
            )
        else:
            raise ValueError("No ambient music file provided and placeholder disabled")
        
        return AudioTrack(
            audio=audio,
            name="ambient",
            volume_db=self.mix_levels.ambient
        )
    
    def create_subliminal_from_voice(
        self,
        voice_track: AudioTrack,
        target_duration_seconds: float,
        speed_factor: float = 1.0
    ) -> AudioTrack:
        """
        Create subliminal track from the voice recording.
        
        This is the recommended approach - same affirmations delivered
        to both conscious (voice) and subconscious (subliminal) mind.
        
        Args:
            voice_track: The loaded voice track
            target_duration_seconds: Target duration
            speed_factor: Speed adjustment (1.0 = normal)
        
        Returns:
            AudioTrack with subliminal audio
        """
        logger.info("Creating subliminal track from voice (same affirmations)")
        
        audio = self.subliminal_processor.create_from_voice(
            voice_track.audio,
            target_duration_seconds,
            speed_factor=speed_factor
        )
        
        return AudioTrack(
            audio=audio,
            name="subliminal",
            volume_db=self.mix_levels.subliminal
        )
    
    def load_subliminal_track(
        self,
        subliminal_path: Optional[Path],
        duration_seconds: float,
        text_affirmations: Optional[str] = None,
        voice_track: Optional[AudioTrack] = None,
        use_voice_as_subliminal: bool = False
    ) -> Optional[AudioTrack]:
        """
        Load or create subliminal affirmations track.
        
        Priority order:
        1. use_voice_as_subliminal - copy from voice track
        2. subliminal_path - load from file
        3. text_affirmations - generate via TTS
        
        Args:
            subliminal_path: Path to pre-recorded subliminal audio
            duration_seconds: Target duration
            text_affirmations: Optional text to convert to TTS
            voice_track: Voice track to copy for subliminals
            use_voice_as_subliminal: If True, create subliminal from voice
        
        Returns:
            AudioTrack object or None if not available
        """
        audio = None
        
        # Priority 1: Copy from voice track
        if use_voice_as_subliminal and voice_track:
            return self.create_subliminal_from_voice(voice_track, duration_seconds)
        
        # Priority 2: Load from file
        if subliminal_path and subliminal_path.exists():
            audio = self.subliminal_processor.create_from_audio(
                subliminal_path,
                target_duration_seconds=duration_seconds
            )
        # Priority 3: TTS from text
        elif text_affirmations:
            audio = self.subliminal_processor.create_from_text(
                text_affirmations,
                duration_seconds=duration_seconds,
                sample_rate=self.sample_rate,
                use_tts=True
            )
        
        if audio:
            return AudioTrack(
                audio=audio,
                name="subliminal",
                volume_db=self.mix_levels.subliminal
            )
        
        return None
    
    def mix_tracks(
        self,
        voice_track: AudioTrack,
        binaural_track: AudioTrack,
        ambient_track: AudioTrack,
        subliminal_track: Optional[AudioTrack] = None,
        intro_silence_ms: int = 3000,
        outro_silence_ms: int = 5000,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> AudioSegment:
        """
        Mix all tracks together into final audio.
        
        Args:
            voice_track: Main voice narration
            binaural_track: Binaural beats
            ambient_track: Ambient music
            subliminal_track: Optional subliminal affirmations
            intro_silence_ms: Silence before voice starts
            outro_silence_ms: Silence after voice ends
            progress_callback: Optional progress callback
        
        Returns:
            Mixed AudioSegment
        """
        logger.info("Mixing tracks...")
        
        if progress_callback:
            progress_callback(0.1)
        
        # Calculate total duration
        voice_duration = voice_track.duration_ms
        total_duration = intro_silence_ms + voice_duration + outro_silence_ms
        
        # Apply volume adjustments
        voice_audio = voice_track.audio + voice_track.volume_db
        binaural_audio = binaural_track.audio + binaural_track.volume_db
        ambient_audio = ambient_track.audio + ambient_track.volume_db
        
        if progress_callback:
            progress_callback(0.3)
        
        # Ensure all tracks are same length
        # Extend binaural and ambient if needed
        if len(binaural_audio) < total_duration:
            binaural_audio = binaural_audio * (total_duration // len(binaural_audio) + 1)
        binaural_audio = binaural_audio[:total_duration]
        
        if len(ambient_audio) < total_duration:
            ambient_audio = ambient_audio * (total_duration // len(ambient_audio) + 1)
        ambient_audio = ambient_audio[:total_duration]
        
        if progress_callback:
            progress_callback(0.5)
        
        # Start with binaural as base
        mixed = binaural_audio
        
        # Overlay ambient
        mixed = mixed.overlay(ambient_audio)
        
        if progress_callback:
            progress_callback(0.6)
        
        # Overlay subliminal if present
        if subliminal_track:
            subliminal_audio = subliminal_track.audio + subliminal_track.volume_db
            if len(subliminal_audio) < total_duration:
                subliminal_audio = subliminal_audio * (total_duration // len(subliminal_audio) + 1)
            subliminal_audio = subliminal_audio[:total_duration]
            mixed = mixed.overlay(subliminal_audio)
        
        if progress_callback:
            progress_callback(0.7)
        
        # Overlay voice at intro offset
        mixed = mixed.overlay(voice_audio, position=intro_silence_ms)
        
        if progress_callback:
            progress_callback(0.8)
        
        # Apply final fade in/out
        mixed = mixed.fade_in(duration=2000)
        mixed = mixed.fade_out(duration=3000)
        
        if progress_callback:
            progress_callback(1.0)
        
        logger.info(f"Mix complete: {len(mixed)/1000:.1f}s")
        return mixed
    
    def build(
        self,
        voice_path: Path,
        output_path: Path,
        theta_frequency: Optional[float] = None,
        duration: Optional[float] = None,
        ambient_path: Optional[Path] = None,
        subliminal_path: Optional[Path] = None,
        subliminal_text: Optional[str] = None,
        subliminal_from_voice: bool = False,
        intro_silence_seconds: Optional[float] = None,
        outro_silence_seconds: Optional[float] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Path:
        """
        Build complete hypnosis audio track.
        
        The recommended workflow:
        1. Record your affirmation script
        2. Use subliminal_from_voice=True to automatically create subliminal layer
        3. The voice and subliminal will contain the same affirmations
        4. Listen daily for best results
        
        Args:
            voice_path: Path to voice recording
            output_path: Path for output file
            theta_frequency: Binaural beat frequency (uses preset if None)
            duration: Optional override for total duration
            ambient_path: Path to ambient music file
            subliminal_path: Path to subliminal audio file
            subliminal_text: Text for TTS subliminal (if available)
            subliminal_from_voice: Create subliminal from voice track (RECOMMENDED)
            intro_silence_seconds: Seconds before voice (uses preset if None)
            outro_silence_seconds: Seconds after voice (uses preset if None)
            progress_callback: Callback with (stage_name, progress 0-1)
        
        Returns:
            Path to output file
        """
        # Use preset defaults if not specified
        intro_silence = intro_silence_seconds if intro_silence_seconds is not None else self._preset["intro_silence"]
        outro_silence = outro_silence_seconds if outro_silence_seconds is not None else self._preset["outro_silence"]
        
        def report(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)
            logger.debug(f"{stage}: {progress*100:.0f}%")
        
        # Load voice
        report("Loading voice", 0.0)
        voice_track = self.load_voice(voice_path)
        report("Loading voice", 1.0)
        
        # Calculate durations
        voice_duration = voice_track.duration_seconds
        total_duration = duration or (intro_silence + voice_duration + outro_silence)
        
        # Generate binaural beats
        report("Generating binaural beats", 0.0)
        binaural_track = self.generate_binaural_track(
            total_duration,
            theta_frequency,
            progress_callback=lambda p: report("Generating binaural beats", p)
        )
        
        # Load/generate ambient
        report("Preparing ambient music", 0.0)
        ambient_track = self.load_ambient_track(
            ambient_path,
            total_duration
        )
        report("Preparing ambient music", 1.0)
        
        # Handle subliminal track
        subliminal_track = None
        
        # Priority: subliminal_from_voice > subliminal_path > subliminal_text
        if subliminal_from_voice:
            report("Creating subliminal from voice", 0.0)
            subliminal_track = self.create_subliminal_from_voice(
                voice_track,
                total_duration
            )
            report("Creating subliminal from voice", 1.0)
        elif subliminal_path or subliminal_text:
            report("Preparing subliminal track", 0.0)
            subliminal_track = self.load_subliminal_track(
                subliminal_path,
                total_duration,
                subliminal_text
            )
            report("Preparing subliminal track", 1.0)
        
        # Mix tracks
        report("Mixing tracks", 0.0)
        final_audio = self.mix_tracks(
            voice_track,
            binaural_track,
            ambient_track,
            subliminal_track,
            intro_silence_ms=int(intro_silence * 1000),
            outro_silence_ms=int(outro_silence * 1000),
            progress_callback=lambda p: report("Mixing tracks", p)
        )
        
        # Export
        report("Exporting", 0.0)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        output_format = output_path.suffix.lower().lstrip('.')
        if output_format == 'mp3':
            final_audio.export(
                str(output_path),
                format='mp3',
                bitrate='320k',
                parameters=['-q:a', '0']
            )
        elif output_format == 'wav':
            final_audio.export(str(output_path), format='wav')
        elif output_format == 'flac':
            final_audio.export(str(output_path), format='flac')
        else:
            final_audio.export(str(output_path), format=output_format)
        
        report("Exporting", 1.0)
        
        logger.info(f"Audio saved to: {output_path}")
        return output_path


def create_test_voice(output_path: Path, duration_seconds: float = 10.0):
    """Create a test voice track (silence with markers) for testing."""
    sample_rate = 44100
    num_samples = int(duration_seconds * sample_rate)
    
    # Create mostly silent audio with periodic beeps
    samples = np.zeros(num_samples, dtype=np.float32)
    
    # Add short beeps every 2 seconds
    beep_duration = int(0.1 * sample_rate)
    beep_freq = 440  # A4
    t_beep = np.linspace(0, 0.1, beep_duration)
    beep = np.sin(2 * np.pi * beep_freq * t_beep) * 0.3
    
    for i in range(0, num_samples, 2 * sample_rate):
        end = min(i + beep_duration, num_samples)
        samples[i:end] = beep[:end-i]
    
    # Convert to 16-bit stereo
    samples_int = (samples * 32767).astype(np.int16)
    stereo = np.empty(num_samples * 2, dtype=np.int16)
    stereo[0::2] = samples_int
    stereo[1::2] = samples_int
    
    # Write WAV
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(stereo.tobytes())
    
    return output_path


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Create test voice
    test_voice = Path("/tmp/test_voice.wav")
    create_test_voice(test_voice, duration_seconds=30)
    
    # Build hypnosis track
    builder = HypnosisAudioBuilder()
    output = builder.build(
        voice_path=test_voice,
        output_path=Path("/tmp/test_hypnosis.mp3"),
        theta_frequency=6.0
    )
    
    print(f"Test output created: {output}")
