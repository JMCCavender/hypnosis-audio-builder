#!/usr/bin/env python3
"""
Subliminal Music Builder

Layer subliminal affirmations and binaural beats into any music track.
Perfect for programming your subconscious while enjoying Bach, lo-fi, or any music.

Usage:
    python subliminal_music_builder.py --music bach.mp3 --voice affirmations.wav -o output.mp3
    python subliminal_music_builder.py --music bach.mp3 --voice affirmations.wav --energy --theta 6 -o output.mp3
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio_builder import (
    HypnosisAudioBuilder,
    BinauralBeatGenerator,
    BinauralConfig,
    SubliminalProcessor,
    MixLevels,
    AudioTrack,
)
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter
import numpy as np
import io
import wave


class SubliminalMusicBuilder:
    """
    Build music tracks with embedded subliminal messages and binaural beats.

    Layers:
    1. Music (Bach, lo-fi, ambient, etc.) - main audible layer
    2. Binaural beats - theta waves for subconscious access
    3. Subliminal voice - your affirmations at sub-perceptual volume
    4. Optional: Energy beats - subtle rhythmic pulse for focus
    """

    # Mix levels for music-based tracks (music is primary)
    DEFAULT_LEVELS = {
        'music': 0,           # Full volume
        'binaural': -18,      # Subtle underneath
        'subliminal': -32,    # Below conscious hearing
        'energy': -25,        # Subtle pulse
    }

    # Energy beat patterns (BPM -> pattern)
    ENERGY_PATTERNS = {
        'focus': {'bpm': 60, 'style': 'pulse'},      # Calm focus
        'flow': {'bpm': 90, 'style': 'pulse'},       # Creative flow
        'drive': {'bpm': 120, 'style': 'kick'},      # Energetic
        'intense': {'bpm': 140, 'style': 'kick'},    # High energy
    }

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.binaural_generator = BinauralBeatGenerator(BinauralConfig())
        self.subliminal_processor = SubliminalProcessor()

    def build(
        self,
        music_path: Path,
        voice_path: Path,
        output_path: Path,
        music_level: float = 0,
        subliminal_level: float = -32,
        binaural_level: float = -18,
        theta_freq: float = 6.0,
        carrier_freq: float = 200.0,
        energy_style: str = None,
        energy_level: float = -25,
        progress_callback=None
    ) -> Path:
        """
        Build a music track with subliminal messaging.

        Args:
            music_path: Path to music file (Bach, lo-fi, etc.)
            voice_path: Path to voice recording of affirmations
            output_path: Output file path
            music_level: Music volume in dB (default 0)
            subliminal_level: Subliminal voice level in dB (default -32)
            binaural_level: Binaural beats level in dB (default -18)
            theta_freq: Theta frequency for binaural beats (default 6.0 Hz)
            carrier_freq: Carrier frequency for binaural beats (default 200 Hz)
            energy_style: Optional energy beat style ('focus', 'flow', 'drive', 'intense')
            energy_level: Energy beat level in dB (default -25)
            progress_callback: Optional callback(stage, progress)

        Returns:
            Path to output file
        """
        def report(stage, progress):
            if progress_callback:
                progress_callback(stage, progress)
            print(f"\r{stage}: {'█' * int(progress * 40)}{'░' * (40 - int(progress * 40))} {int(progress * 100)}%", end='', flush=True)

        # Load music
        report("Loading music", 0)
        music = AudioSegment.from_file(str(music_path))
        if music.channels == 1:
            music = music.set_channels(2)
        music = normalize(music)
        duration_ms = len(music)
        duration_sec = duration_ms / 1000
        report("Loading music", 1)
        print()

        # Load and process subliminal voice
        report("Processing subliminal voice", 0)
        voice = AudioSegment.from_file(str(voice_path))
        if voice.channels == 1:
            voice = voice.set_channels(2)
        voice = normalize(voice)

        # Apply subliminal processing (EQ for softer sound)
        voice = low_pass_filter(voice, 4000)
        voice = high_pass_filter(voice, 100)

        # Loop voice to match music duration
        voice_with_gap = voice + AudioSegment.silent(duration=500)
        loops_needed = int(duration_ms / len(voice_with_gap)) + 1
        subliminal = (voice_with_gap * loops_needed)[:duration_ms]
        report("Processing subliminal voice", 1)
        print()

        # Generate binaural beats
        report("Generating binaural beats", 0)
        self.binaural_generator.config.base_frequency = carrier_freq
        self.binaural_generator.config.theta_frequency = theta_freq
        binaural = self.binaural_generator.generate(
            duration_sec,
            progress_callback=lambda p: report("Generating binaural beats", p)
        )
        print()

        # Generate energy beats if requested
        energy_track = None
        if energy_style and energy_style in self.ENERGY_PATTERNS:
            report("Generating energy beats", 0)
            pattern = self.ENERGY_PATTERNS[energy_style]
            energy_track = self._generate_energy_beats(
                duration_sec,
                pattern['bpm'],
                pattern['style']
            )
            report("Generating energy beats", 1)
            print()

        # Mix everything
        report("Mixing tracks", 0)

        # Start with binaural as base (ensures stereo)
        mixed = binaural + binaural_level
        report("Mixing tracks", 0.2)

        # Add music
        music_adjusted = music + music_level
        mixed = mixed.overlay(music_adjusted)
        report("Mixing tracks", 0.4)

        # Add subliminal
        subliminal_adjusted = subliminal + subliminal_level
        mixed = mixed.overlay(subliminal_adjusted)
        report("Mixing tracks", 0.6)

        # Add energy beats if present
        if energy_track:
            energy_adjusted = energy_track + energy_level
            mixed = mixed.overlay(energy_adjusted)
        report("Mixing tracks", 0.8)

        # Final processing
        mixed = mixed.fade_in(2000).fade_out(3000)
        report("Mixing tracks", 1)
        print()

        # Export
        report("Exporting", 0)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_format = output_path.suffix.lower().lstrip('.')
        if output_format == 'mp3':
            mixed.export(str(output_path), format='mp3', bitrate='320k')
        else:
            mixed.export(str(output_path), format=output_format)

        report("Exporting", 1)
        print()

        return output_path

    def _generate_energy_beats(
        self,
        duration_sec: float,
        bpm: int,
        style: str
    ) -> AudioSegment:
        """Generate subtle rhythmic energy beats."""
        sample_rate = self.sample_rate
        num_samples = int(duration_sec * sample_rate)

        # Calculate beat timing
        beat_interval = 60.0 / bpm  # seconds per beat
        samples_per_beat = int(beat_interval * sample_rate)

        # Generate beat sound based on style
        if style == 'pulse':
            # Soft sine pulse
            beat_duration = 0.1  # 100ms
            beat_samples = int(beat_duration * sample_rate)
            t = np.linspace(0, beat_duration, beat_samples)
            beat = np.sin(2 * np.pi * 60 * t)  # 60 Hz pulse
            # Apply envelope
            envelope = np.sin(np.pi * t / beat_duration)
            beat *= envelope * 0.5
        else:  # kick style
            # Subtle kick drum sound
            beat_duration = 0.15
            beat_samples = int(beat_duration * sample_rate)
            t = np.linspace(0, beat_duration, beat_samples)
            # Frequency sweep from 150Hz to 50Hz
            freq_sweep = 150 * np.exp(-t * 20) + 50
            phase = np.cumsum(2 * np.pi * freq_sweep / sample_rate)
            beat = np.sin(phase)
            # Apply envelope
            envelope = np.exp(-t * 30)
            beat *= envelope * 0.6

        # Create full track
        samples = np.zeros(num_samples)
        beat_positions = range(0, num_samples, samples_per_beat)

        for pos in beat_positions:
            end = min(pos + len(beat), num_samples)
            samples[pos:end] += beat[:end - pos]

        # Convert to stereo
        samples_int = (samples * 32767).astype(np.int16)
        stereo = np.empty(num_samples * 2, dtype=np.int16)
        stereo[0::2] = samples_int
        stereo[1::2] = samples_int

        # Convert to AudioSegment
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(stereo.tobytes())

        buffer.seek(0)
        return AudioSegment.from_wav(buffer)


def main():
    parser = argparse.ArgumentParser(
        description='Layer subliminal messages and binaural beats into music',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: Bach + subliminal affirmations
  python subliminal_music_builder.py --music bach.mp3 --voice affirmations.wav -o output.mp3

  # With energy beats for focus
  python subliminal_music_builder.py --music bach.mp3 --voice affirmations.wav --energy focus -o output.mp3

  # Custom levels and theta frequency
  python subliminal_music_builder.py --music bach.mp3 --voice affirmations.wav \\
      --subliminal-level -30 --theta 4 -o deep_programming.mp3

  # High energy programming
  python subliminal_music_builder.py --music electronic.mp3 --voice affirmations.wav \\
      --energy drive --theta 7.5 -o energy.mp3

Energy styles:
  focus   - 60 BPM soft pulse (calm concentration)
  flow    - 90 BPM soft pulse (creative flow state)
  drive   - 120 BPM subtle kick (energetic focus)
  intense - 140 BPM subtle kick (high energy)
        """
    )

    parser.add_argument('--music', '-m', type=Path, required=True,
                        help='Music file (Bach, lo-fi, ambient, etc.)')
    parser.add_argument('--voice', '-v', type=Path, required=True,
                        help='Voice recording of affirmations')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output file path')

    # Level controls
    parser.add_argument('--music-level', type=float, default=0,
                        help='Music volume in dB (default: 0)')
    parser.add_argument('--subliminal-level', type=float, default=-32,
                        help='Subliminal voice level in dB (default: -32)')
    parser.add_argument('--binaural-level', type=float, default=-18,
                        help='Binaural beats level in dB (default: -18)')

    # Binaural config
    parser.add_argument('--theta', type=float, default=6.0,
                        help='Theta frequency in Hz (default: 6.0)')
    parser.add_argument('--carrier', type=float, default=200,
                        help='Carrier frequency in Hz (default: 200)')

    # Energy beats
    parser.add_argument('--energy', '-e', choices=['focus', 'flow', 'drive', 'intense'],
                        help='Add subtle energy beats')
    parser.add_argument('--energy-level', type=float, default=-25,
                        help='Energy beat level in dB (default: -25)')

    args = parser.parse_args()

    # Validate inputs
    if not args.music.exists():
        print(f"Error: Music file not found: {args.music}")
        sys.exit(1)
    if not args.voice.exists():
        print(f"Error: Voice file not found: {args.voice}")
        sys.exit(1)

    # Print config
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           SUBLIMINAL MUSIC BUILDER                           ║
╠══════════════════════════════════════════════════════════════╣
║  Music:      {str(args.music)[:45]:<45} ║
║  Voice:      {str(args.voice)[:45]:<45} ║
║  Output:     {str(args.output)[:45]:<45} ║
╠══════════════════════════════════════════════════════════════╣
║  Music Level:      {args.music_level:>6} dB                              ║
║  Subliminal Level: {args.subliminal_level:>6} dB                              ║
║  Binaural Level:   {args.binaural_level:>6} dB                              ║
║  Theta Frequency:  {args.theta:>6} Hz                              ║
║  Energy Style:     {str(args.energy or 'None'):>6}                                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    builder = SubliminalMusicBuilder()

    try:
        output = builder.build(
            music_path=args.music,
            voice_path=args.voice,
            output_path=args.output,
            music_level=args.music_level,
            subliminal_level=args.subliminal_level,
            binaural_level=args.binaural_level,
            theta_freq=args.theta,
            carrier_freq=args.carrier,
            energy_style=args.energy,
            energy_level=args.energy_level
        )

        print(f"""
✓ Subliminal music track created: {output}

Layers:
  • Music at {args.music_level} dB (audible)
  • Binaural beats at {args.binaural_level} dB ({args.theta} Hz theta)
  • Subliminal voice at {args.subliminal_level} dB (sub-perceptual)
  {f'• Energy beats at {args.energy_level} dB ({args.energy} style)' if args.energy else ''}

🎧 Use headphones for binaural effect. Enjoy your Bach + programming!
""")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
