#!/usr/bin/env python3
"""
Album Build Module - Batch pairing of songs (ambient) with vocals.

Pairs multiple songs from an album folder with multiple vocal recordings,
cycling vocals top-to-bottom when there are more songs than vocals.
Each pairing produces a separate output file.

Usage flow:
    1. Discover audio files in songs dir and vocals dir
    2. Pair them top-to-bottom (cycling vocals if needed)
    3. Build each pair using HypnosisAudioBuilder
    4. Output individual tracks to output directory
"""

import itertools
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .audio_builder import HypnosisAudioBuilder, MixLevels

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}


def discover_audio_files(directory: Path) -> List[Path]:
    """
    Discover audio files in a directory, sorted alphabetically.

    Args:
        directory: Path to scan for audio files.

    Returns:
        Sorted list of audio file paths.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]
    return sorted(files, key=lambda f: f.name)


def pair_songs_with_vocals(
    songs: List[Path],
    vocals: List[Path],
) -> List[Tuple[Path, Path]]:
    """
    Pair songs with vocals top-to-bottom, cycling vocals if needed.

    Args:
        songs: Ordered list of song (ambient) file paths.
        vocals: Ordered list of vocal file paths.

    Returns:
        List of (song, vocal) tuples. Empty if either list is empty.
    """
    if not songs or not vocals:
        return []

    vocal_cycle = itertools.cycle(vocals)
    return [(song, next(vocal_cycle)) for song in songs]


def generate_album_output_path(
    song_path: Path,
    vocal_path: Path,
    output_dir: Path,
    output_format: str,
    track_number: int,
) -> Path:
    """
    Generate output file path for an album track.

    Format: {track_number:02d}_{song_stem}_{vocal_stem}.{format}

    Args:
        song_path: Path to the song file.
        vocal_path: Path to the vocal file.
        output_dir: Output directory.
        output_format: File extension without dot (e.g. "mp3").
        track_number: 1-based track number for ordering.

    Returns:
        Full output Path.
    """
    filename = f"{track_number:02d}_{song_path.stem}_{vocal_path.stem}.{output_format}"
    return output_dir / filename


def album_build(
    songs_dir: Path,
    output_dir: Path,
    output_format: str = "mp3",
    session_type: str = "standard",
    vocals_dir: Optional[Path] = None,
    vocal_files: Optional[List[Path]] = None,
    subliminal_from_voice: bool = False,
    mix_levels: Optional[MixLevels] = None,
    sample_rate: int = 44100,
    quiet: bool = False,
) -> List[Path]:
    """
    Build an album by pairing songs with vocals.

    Args:
        songs_dir: Directory containing song (ambient) files.
        output_dir: Directory for output files.
        output_format: Output audio format.
        session_type: Session preset name.
        vocals_dir: Directory containing vocal files (provide this or vocal_files).
        vocal_files: Explicit list of vocal file paths.
        subliminal_from_voice: Create subliminal layer from voice.
        mix_levels: Optional volume overrides.
        sample_rate: Output sample rate.
        quiet: Suppress progress output.

    Returns:
        List of output file paths that were created.
    """
    # Discover songs
    songs = discover_audio_files(songs_dir)
    if not songs:
        if not quiet:
            print(f"No audio files found in songs directory: {songs_dir}")
        return []

    # Resolve vocals
    if vocal_files:
        vocals = list(vocal_files)
    elif vocals_dir:
        vocals = discover_audio_files(vocals_dir)
    else:
        vocals = []

    if not vocals:
        if not quiet:
            print("No vocal files provided")
        return []

    # Pair them
    pairs = pair_songs_with_vocals(songs, vocals)

    if not quiet:
        print(f"Album build: {len(songs)} songs × {len(vocals)} vocals → {len(pairs)} tracks")
        for i, (song, vocal) in enumerate(pairs, 1):
            print(f"  {i:02d}. {song.name} + {vocal.name}")
        print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build each track
    results = []
    for i, (song_path, vocal_path) in enumerate(pairs, 1):
        output_path = generate_album_output_path(
            song_path, vocal_path, output_dir, output_format, i
        )

        if not quiet:
            print(f"[{i}/{len(pairs)}] Building: {output_path.name}")

        try:
            builder = HypnosisAudioBuilder(
                mix_levels=mix_levels,
                sample_rate=sample_rate,
                session_type=session_type,
            )

            builder.build(
                voice_path=vocal_path,
                output_path=output_path,
                ambient_path=song_path,
                subliminal_from_voice=subliminal_from_voice,
            )

            results.append(output_path)
            if not quiet:
                print(f"  ✓ {output_path.name}")
        except Exception as e:
            if not quiet:
                print(f"  ✗ Error on track {i}: {e}")
            logger.error(f"Album build error on track {i}: {e}", exc_info=True)

    if not quiet:
        print(f"\nAlbum complete: {len(results)}/{len(pairs)} tracks created")

    return results
