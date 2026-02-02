#!/usr/bin/env python3
"""
Hypnosis Audio Builder - Command Line Interface

Provides a full-featured CLI for creating self-hypnosis audio tracks
with binaural beats, ambient music, and subliminal messaging.

Usage:
    python hypnosis_audio_builder.py --voice script.wav --output hypnosis.mp3

Author: Claude (Anthropic)
License: MIT
"""

import argparse
import logging
import sys
import json
import webbrowser
import yaml
from pathlib import Path
from typing import Optional

from src.audio_builder import (
    HypnosisAudioBuilder,
    MixLevels,
    BinauralConfig,
    ScriptValidator,
    create_test_voice
)


def setup_logging(verbose: bool = False, debug: bool = False):
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def create_progress_bar(width: int = 40):
    """Create a simple progress bar printer."""
    current_stage = [None]
    
    def update(stage: str, progress: float):
        if stage != current_stage[0]:
            if current_stage[0] is not None:
                print()  # New line for new stage
            current_stage[0] = stage
            print(f"\n{stage}:", end=" ")
        
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percent = int(progress * 100)
        print(f"\r{stage}: [{bar}] {percent}%", end="", flush=True)
        
        if progress >= 1.0:
            print()  # New line when complete
    
    return update


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML or JSON file."""
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        if config_path.suffix in {'.yaml', '.yml'}:
            return yaml.safe_load(f) or {}
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def merge_config_with_args(config: dict, args: argparse.Namespace) -> argparse.Namespace:
    """Merge config file values with command line arguments."""
    # Command line args take precedence
    for key, value in config.items():
        if hasattr(args, key):
            arg_value = getattr(args, key)
            # Only use config value if arg wasn't explicitly set
            if arg_value is None:
                setattr(args, key, value)
    
    return args


def validate_brainwave_frequency(value: str) -> float:
    """Validate brainwave frequency is in valid range (theta through beta)."""
    freq = float(value)
    if not 4.0 <= freq <= 30.0:
        raise argparse.ArgumentTypeError(
            f"Brainwave frequency must be between 4.0 and 30.0 Hz (got {freq})\n"
            "  Theta: 4-8 Hz (meditation, subconscious)\n"
            "  Alpha: 8-13 Hz (relaxed focus)\n"
            "  Beta: 13-30 Hz (active concentration)"
        )
    return freq


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create self-hypnosis audio tracks with binaural beats and ambient music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Self-Hypnosis Audio Creation Workflow:
  1. Write your script (present tense, sensory, emotional language)
  2. Record yourself speaking the affirmations
  3. Run this tool to add theta waves, ambient music, and subliminals
  4. Listen daily - morning and night for best results

The theta waves open access to the subconscious mind.
The subliminal layer reinforces the same affirmations beneath conscious awareness.

Examples:
  Basic usage with auto-subliminal from voice:
    %(prog)s --voice morning_energy.wav --subliminal-from-voice --output morning.mp3

  Morning session preset:
    %(prog)s --voice script.wav --session morning --subliminal-from-voice -o morning.mp3

  Night/sleep session:
    %(prog)s --voice script.wav --session night --subliminal-from-voice -o night.mp3

  Validate your affirmation script:
    %(prog)s --validate-script affirmations.txt

  With ambient music:
    %(prog)s --voice script.wav --ambient-music meditation.mp3 --output final.mp3

  Full configuration:
    %(prog)s \\
      --voice morning_energy.wav \\
      --theta-freq 6.0 \\
      --duration auto \\
      --ambient-music meditation_01.mp3 \\
      --subliminal-from-voice \\
      --output morning_energy_final.mp3

  List available session presets:
    %(prog)s --list-presets

  Batch processing:
    %(prog)s --batch scripts_folder/ --output-dir processed/ --subliminal-from-voice
        """
    )
    
    # Required arguments
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--voice", "-v",
        type=Path,
        help="Path to voice narration file (WAV, MP3, FLAC)"
    )
    input_group.add_argument(
        "--batch", "-b",
        type=Path,
        help="Directory containing voice files for batch processing"
    )
    input_group.add_argument(
        "--validate-script",
        type=Path,
        metavar="FILE",
        help="Validate an affirmation script file (checks for best practices)"
    )
    
    # Session presets
    session_group = parser.add_argument_group("Session Presets")
    session_group.add_argument(
        "--session", "-S",
        choices=["morning", "night", "sleep", "subliminal", "standard", "focus", "active"],
        default="standard",
        help="Session type preset (default: standard). Use --list-presets for details"
    )
    session_group.add_argument(
        "--list-presets",
        action="store_true",
        help="List available session presets and exit"
    )
    session_group.add_argument(
        "--open-frequency-guide",
        action="store_true",
        help="Open the interactive brainwave frequency explorer in your browser"
    )
    
    # Audio generation options
    audio_group = parser.add_argument_group("Audio Generation")
    audio_group.add_argument(
        "--theta-freq", "-t",
        type=validate_brainwave_frequency,
        metavar="FREQ",
        help="Brainwave frequency in Hz (4.0-30.0). Theta: 4-8, Alpha: 8-13, Beta: 13-30. Overrides session preset."
    )
    audio_group.add_argument(
        "--base-freq",
        type=float,
        default=200.0,
        metavar="FREQ",
        help="Binaural beat carrier frequency in Hz (default: 200.0)"
    )
    audio_group.add_argument(
        "--duration", "-d",
        type=str,
        default="auto",
        help="Track duration: 'auto' (from voice), or seconds (default: auto)"
    )
    audio_group.add_argument(
        "--ambient-music", "-a",
        type=Path,
        metavar="FILE",
        help="Path to ambient music file (optional)"
    )
    audio_group.add_argument(
        "--no-ambient-placeholder",
        action="store_true",
        help="Disable ambient placeholder generation when no music provided"
    )
    
    # Subliminal options - enhanced
    subliminal_group = parser.add_argument_group("Subliminal Track (Recommended)")
    subliminal_group.add_argument(
        "--subliminal-from-voice", "--sfv",
        action="store_true",
        help="RECOMMENDED: Create subliminal track from voice (same affirmations)"
    )
    subliminal_group.add_argument(
        "--subliminal", "-s",
        type=Path,
        metavar="FILE",
        help="Path to pre-recorded subliminal audio file"
    )
    subliminal_group.add_argument(
        "--subliminal-text",
        type=Path,
        metavar="FILE",
        help="Path to text file with affirmations for TTS subliminal"
    )
    
    # Timing options
    timing_group = parser.add_argument_group("Timing")
    timing_group.add_argument(
        "--intro-silence",
        type=float,
        metavar="SECS",
        help="Seconds of ambient/binaural before voice starts (overrides preset)"
    )
    timing_group.add_argument(
        "--outro-silence",
        type=float,
        metavar="SECS",
        help="Seconds of ambient/binaural after voice ends (overrides preset)"
    )
    
    # Volume options
    volume_group = parser.add_argument_group("Volume Levels (dB) - Override Preset")
    volume_group.add_argument(
        "--voice-volume",
        type=float,
        metavar="DB",
        help="Voice volume in dB (default from preset)"
    )
    volume_group.add_argument(
        "--binaural-volume",
        type=float,
        metavar="DB",
        help="Binaural beats volume in dB (default from preset)"
    )
    volume_group.add_argument(
        "--ambient-volume",
        type=float,
        metavar="DB",
        help="Ambient music volume in dB (default from preset)"
    )
    volume_group.add_argument(
        "--subliminal-volume",
        type=float,
        metavar="DB",
        help="Subliminal track volume in dB (default: -35)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (MP3, WAV, or FLAC)"
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch processing"
    )
    output_group.add_argument(
        "--output-format",
        choices=["mp3", "wav", "flac"],
        default="mp3",
        help="Output format (default: mp3)"
    )
    output_group.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        metavar="RATE",
        help="Output sample rate in Hz (default: 44100)"
    )
    
    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config", "-c",
        type=Path,
        metavar="FILE",
        help="Path to YAML or JSON configuration file"
    )
    config_group.add_argument(
        "--save-config",
        type=Path,
        metavar="FILE",
        help="Save current settings to config file"
    )
    
    # Preview and testing
    preview_group = parser.add_argument_group("Preview & Testing")
    preview_group.add_argument(
        "--preview",
        action="store_true",
        help="Generate a 30-second preview instead of full track"
    )
    preview_group.add_argument(
        "--preview-duration",
        type=float,
        default=30.0,
        metavar="SECS",
        help="Preview duration in seconds (default: 30)"
    )
    preview_group.add_argument(
        "--test",
        action="store_true",
        help="Run with test audio to verify setup"
    )
    
    # General options
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose output"
    )
    general_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    general_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    general_group.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.1.0"
    )
    
    return parser.parse_args()


def process_single_file(
    voice_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    builder: HypnosisAudioBuilder,
    progress_callback=None
):
    """Process a single voice file into hypnosis audio."""
    # Parse duration
    duration = None
    if args.duration != "auto":
        try:
            duration = float(args.duration)
        except ValueError:
            raise ValueError(f"Invalid duration: {args.duration}")
    
    # Load subliminal text if provided
    subliminal_text = None
    if args.subliminal_text and args.subliminal_text.exists():
        subliminal_text = args.subliminal_text.read_text()
    
    # Build the track
    return builder.build(
        voice_path=voice_path,
        output_path=output_path,
        theta_frequency=args.theta_freq,
        duration=duration,
        ambient_path=args.ambient_music,
        subliminal_path=args.subliminal,
        subliminal_text=subliminal_text,
        subliminal_from_voice=args.subliminal_from_voice,
        intro_silence_seconds=args.intro_silence,
        outro_silence_seconds=args.outro_silence,
        progress_callback=progress_callback
    )


def batch_process(
    batch_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    builder: HypnosisAudioBuilder
):
    """Process all audio files in a directory."""
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    voice_files = [
        f for f in batch_dir.iterdir()
        if f.suffix.lower() in supported_formats
    ]
    
    if not voice_files:
        print(f"No audio files found in {batch_dir}")
        return []
    
    print(f"Found {len(voice_files)} files to process")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, voice_file in enumerate(voice_files, 1):
        print(f"\n[{i}/{len(voice_files)}] Processing: {voice_file.name}")
        
        output_name = voice_file.stem + "_hypnosis." + args.output_format
        output_path = output_dir / output_name
        
        try:
            progress = create_progress_bar() if not args.quiet else None
            result = process_single_file(
                voice_file, 
                output_path, 
                args, 
                builder,
                progress_callback=progress
            )
            results.append(result)
            print(f"  ✓ Created: {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    return results


def save_config_file(args: argparse.Namespace, output_path: Path):
    """Save current settings to a config file."""
    config = {
        "theta_freq": args.theta_freq,
        "base_freq": args.base_freq,
        "intro_silence": args.intro_silence,
        "outro_silence": args.outro_silence,
        "voice_volume": args.voice_volume,
        "binaural_volume": args.binaural_volume,
        "ambient_volume": args.ambient_volume,
        "subliminal_volume": args.subliminal_volume,
        "output_format": args.output_format,
        "sample_rate": args.sample_rate,
    }
    
    # Add optional paths if set
    if args.ambient_music:
        config["ambient_music"] = str(args.ambient_music)
    if args.subliminal:
        config["subliminal"] = str(args.subliminal)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.suffix in {'.yaml', '.yml'}:
            yaml.dump(config, f, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.debug)
    logger = logging.getLogger(__name__)
    
    # Handle --list-presets
    if args.list_presets:
        print(HypnosisAudioBuilder.list_presets())
        return 0

    # Handle --open-frequency-guide
    if args.open_frequency_guide:
        guide_path = Path(__file__).parent / "brainwave-explorer-808.html"
        if guide_path.exists():
            webbrowser.open(f"file://{guide_path.resolve()}")
            print(f"Opening brainwave frequency explorer: {guide_path}")
        else:
            print(f"Error: Frequency guide not found at {guide_path}")
            return 1
        return 0
    
    # Handle --validate-script
    if args.validate_script:
        if not args.validate_script.exists():
            print(f"Error: Script file not found: {args.validate_script}")
            return 1
        
        text = args.validate_script.read_text()
        analysis = ScriptValidator.validate(text)
        print(ScriptValidator.format_report(analysis))
        return 0 if analysis["valid"] else 1
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
    
    # Save config if requested
    if args.save_config:
        save_config_file(args, args.save_config)
        if not (args.voice or args.batch or args.test):
            return 0
    
    # Validate required inputs
    if not args.test and not args.voice and not args.batch:
        print("Error: --voice or --batch is required (or use --test for testing)")
        print("Use --help for usage information")
        return 1
    
    # Get preset for this session type
    preset = HypnosisAudioBuilder.SESSION_PRESETS.get(
        args.session, 
        HypnosisAudioBuilder.SESSION_PRESETS["standard"]
    )
    
    # Build mix levels - use preset as base, override with explicit args
    preset_mix = preset["mix"]
    mix_levels = MixLevels(
        voice=args.voice_volume if args.voice_volume is not None else preset_mix.voice,
        binaural=args.binaural_volume if args.binaural_volume is not None else preset_mix.binaural,
        ambient=args.ambient_volume if args.ambient_volume is not None else preset_mix.ambient,
        subliminal=args.subliminal_volume if args.subliminal_volume is not None else preset_mix.subliminal
    )
    
    # Create builder with session type
    builder = HypnosisAudioBuilder(
        mix_levels=mix_levels,
        sample_rate=args.sample_rate,
        session_type=args.session
    )
    
    # Override binaural config if theta-freq specified
    if args.theta_freq is not None:
        binaural_config = BinauralConfig(
            base_frequency=args.base_freq,
            theta_frequency=args.theta_freq,
            sample_rate=args.sample_rate
        )
        builder.set_binaural_config(binaural_config)
    
    try:
        # Test mode
        if args.test:
            print("Running test mode...")
            test_voice = Path("/tmp/hypnosis_test_voice.wav")
            test_output = Path("/tmp/hypnosis_test_output.mp3")
            
            print("Creating test voice track...")
            create_test_voice(test_voice, duration_seconds=15)
            
            args.voice = test_voice
            args.output = test_output
            args.duration = "auto"
            args.subliminal_from_voice = True  # Test subliminal feature
        
        # Progress callback
        progress = create_progress_bar() if not args.quiet else None
        
        # Batch processing
        if args.batch:
            if not args.output_dir:
                args.output_dir = args.batch / "processed"
            
            results = batch_process(args.batch, args.output_dir, args, builder)
            print(f"\nBatch processing complete: {len(results)} files created")
            return 0
        
        # Single file processing
        if args.voice:
            # Determine output path
            if not args.output:
                suffix = f".{args.session}" if args.session != "standard" else ""
                args.output = args.voice.with_suffix(f"{suffix}.hypnosis.{args.output_format}")
            
            print(f"Session:  {args.session} ({preset['description']})")
            print(f"Input:    {args.voice}")
            print(f"Output:   {args.output}")
            print(f"Theta:    {builder.binaural_generator.config.theta_frequency} Hz")
            if args.subliminal_from_voice:
                print(f"Subliminal: from voice (same affirmations)")
            print()
            
            result = process_single_file(
                args.voice,
                args.output,
                args,
                builder,
                progress_callback=progress
            )
            
            print(f"\n✓ Hypnosis audio created: {result}")
            print(f"\nRecommended: Listen daily, morning and night, with headphones.")
            return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
