#!/usr/bin/env python3
"""
Test suite for Album Build feature.

Album Build pairs multiple songs (ambient tracks) with multiple vocals,
cycling vocals top-to-bottom if there are more songs than vocals.
Each pairing produces a separate output file.

Run with: python -m pytest tests/test_album_build.py -v
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_builder import create_test_voice, HypnosisAudioBuilder

# Shared test path constants
SONG1 = Path("song1.mp3")
SONG2 = Path("song2.mp3")
SONG3 = Path("song3.mp3")
SONG_MP3 = Path("song.mp3")
SONG_WAV = Path("song.wav")
VOCAL1 = Path("vocal1.wav")
VOCAL2 = Path("vocal2.wav")
VOCAL3 = Path("vocal3.wav")
VOCAL_WAV = Path("vocal.wav")
V1_WAV = Path("v1.wav")
VOCALS_DIR_PATH = "/vocals"
SYS_ARGV = "sys.argv"


class TestPairSongsWithVocals(unittest.TestCase):
    """Test the pairing logic that maps songs to vocals with cycling."""

    def test_equal_counts(self):
        """When songs == vocals, each pairs 1:1."""
        from src.album_builder import pair_songs_with_vocals

        songs = [SONG1, SONG2, SONG3]
        vocals = [VOCAL1, VOCAL2, VOCAL3]

        pairs = pair_songs_with_vocals(songs, vocals)

        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0], (SONG1, VOCAL1))
        self.assertEqual(pairs[1], (SONG2, VOCAL2))
        self.assertEqual(pairs[2], (SONG3, VOCAL3))

    def test_more_songs_than_vocals_cycles(self):
        """When more songs than vocals, vocals cycle from the beginning."""
        from src.album_builder import pair_songs_with_vocals

        songs = [Path(f"song{i}.mp3") for i in range(5)]
        vocals = [VOCAL1, VOCAL2]

        pairs = pair_songs_with_vocals(songs, vocals)

        self.assertEqual(len(pairs), 5)
        # Vocals cycle: v1, v2, v1, v2, v1
        self.assertEqual(pairs[0][1], VOCAL1)
        self.assertEqual(pairs[1][1], VOCAL2)
        self.assertEqual(pairs[2][1], VOCAL1)
        self.assertEqual(pairs[3][1], VOCAL2)
        self.assertEqual(pairs[4][1], VOCAL1)

    def test_more_vocals_than_songs(self):
        """When more vocals than songs, extra vocals are unused."""
        from src.album_builder import pair_songs_with_vocals

        songs = [SONG1]
        vocals = [VOCAL1, VOCAL2, VOCAL3]

        pairs = pair_songs_with_vocals(songs, vocals)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], (SONG1, VOCAL1))

    def test_single_vocal_repeats_for_all_songs(self):
        """A single vocal repeats across all songs."""
        from src.album_builder import pair_songs_with_vocals

        songs = [Path(f"song{i}.mp3") for i in range(4)]
        vocals = [VOCAL_WAV]

        pairs = pair_songs_with_vocals(songs, vocals)

        self.assertEqual(len(pairs), 4)
        for pair in pairs:
            self.assertEqual(pair[1], VOCAL_WAV)

    def test_empty_songs_returns_empty(self):
        """No songs means no pairs."""
        from src.album_builder import pair_songs_with_vocals

        pairs = pair_songs_with_vocals([], [VOCAL_WAV])
        self.assertEqual(pairs, [])

    def test_empty_vocals_returns_empty(self):
        """No vocals means no pairs."""
        from src.album_builder import pair_songs_with_vocals

        pairs = pair_songs_with_vocals([SONG_MP3], [])
        self.assertEqual(pairs, [])

    def test_preserves_song_order(self):
        """Songs maintain their sorted order in the pairing."""
        from src.album_builder import pair_songs_with_vocals

        songs = [Path("c_song.mp3"), Path("a_song.mp3"), Path("b_song.mp3")]
        vocals = [V1_WAV, Path("v2.wav"), Path("v3.wav")]

        pairs = pair_songs_with_vocals(songs, vocals)

        # Should preserve the order passed in
        self.assertEqual(pairs[0][0], Path("c_song.mp3"))
        self.assertEqual(pairs[1][0], Path("a_song.mp3"))
        self.assertEqual(pairs[2][0], Path("b_song.mp3"))


class TestDiscoverAudioFiles(unittest.TestCase):
    """Test discovery of audio files from a directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discovers_supported_formats(self):
        """Finds wav, mp3, flac, ogg, m4a files."""
        from src.album_builder import discover_audio_files

        for name in ["track.wav", "track.mp3", "track.flac", "track.ogg", "track.m4a"]:
            (Path(self.temp_dir) / name).touch()

        files = discover_audio_files(Path(self.temp_dir))
        self.assertEqual(len(files), 5)

    def test_ignores_non_audio_files(self):
        """Skips txt, jpg, pdf, etc."""
        from src.album_builder import discover_audio_files

        (Path(self.temp_dir) / "notes.txt").touch()
        (Path(self.temp_dir) / "cover.jpg").touch()
        (Path(self.temp_dir) / "song.mp3").touch()

        files = discover_audio_files(Path(self.temp_dir))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "song.mp3")

    def test_returns_sorted_by_name(self):
        """Files are returned sorted alphabetically for predictable pairing."""
        from src.album_builder import discover_audio_files

        for name in ["03_track.mp3", "01_track.mp3", "02_track.mp3"]:
            (Path(self.temp_dir) / name).touch()

        files = discover_audio_files(Path(self.temp_dir))
        names = [f.name for f in files]
        self.assertEqual(names, ["01_track.mp3", "02_track.mp3", "03_track.mp3"])

    def test_empty_directory(self):
        """Empty directory returns empty list."""
        from src.album_builder import discover_audio_files

        files = discover_audio_files(Path(self.temp_dir))
        self.assertEqual(files, [])

    def test_nonexistent_directory_raises(self):
        """Non-existent directory raises FileNotFoundError."""
        from src.album_builder import discover_audio_files

        with self.assertRaises(FileNotFoundError):
            discover_audio_files(Path("/nonexistent/dir"))


class TestGenerateAlbumOutputPath(unittest.TestCase):
    """Test output filename generation for album tracks."""

    def test_combines_song_and_vocal_names(self):
        """Output name includes both song stem and vocal stem."""
        from src.album_builder import generate_album_output_path

        result = generate_album_output_path(
            song_path=Path("ambient_ocean.mp3"),
            vocal_path=Path("morning_energy.wav"),
            output_dir=Path("/output"),
            output_format="mp3",
            track_number=1,
        )

        self.assertEqual(result.parent, Path("/output"))
        self.assertIn("ambient_ocean", result.stem)
        self.assertIn("morning_energy", result.stem)
        self.assertEqual(result.suffix, ".mp3")

    def test_track_number_prefix(self):
        """Output filename starts with zero-padded track number."""
        from src.album_builder import generate_album_output_path

        result = generate_album_output_path(
            song_path=SONG_MP3,
            vocal_path=VOCAL_WAV,
            output_dir=Path("/out"),
            output_format="wav",
            track_number=3,
        )

        self.assertTrue(result.name.startswith("03_"))

    def test_format_respected(self):
        """Output uses the specified format extension."""
        from src.album_builder import generate_album_output_path

        result = generate_album_output_path(
            song_path=SONG_MP3,
            vocal_path=VOCAL_WAV,
            output_dir=Path("/out"),
            output_format="flac",
            track_number=1,
        )

        self.assertEqual(result.suffix, ".flac")


class TestAlbumBuild(unittest.TestCase):
    """Integration tests for the full album build pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.songs_dir = Path(self.temp_dir) / "songs"
        self.vocals_dir = Path(self.temp_dir) / "vocals"
        self.output_dir = Path(self.temp_dir) / "output"
        self.songs_dir.mkdir()
        self.vocals_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_audio(self, directory: Path, name: str):
        """Create a test audio file long enough for crossfade looping."""
        path = directory / name
        create_test_voice(path, duration_seconds=10.0)
        return path

    def _run_album_build(self, **kwargs):
        """Run album_build with common defaults."""
        from src.album_builder import album_build

        defaults = {
            "songs_dir": self.songs_dir,
            "vocals_dir": self.vocals_dir,
            "output_dir": self.output_dir,
            "output_format": "wav",
            "session_type": "standard",
        }
        defaults.update(kwargs)
        return album_build(**defaults)

    def test_album_build_creates_output_files(self):
        """Album build creates one output file per song."""
        # Create 3 songs and 2 vocals
        for i in range(3):
            self._create_test_audio(self.songs_dir, f"song_{i+1}.wav")
        for i in range(2):
            self._create_test_audio(self.vocals_dir, f"vocal_{i+1}.wav")

        results = self._run_album_build()

        self.assertEqual(len(results), 3)
        for path in results:
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_album_build_with_single_vocal(self):
        """Single vocal is used for all songs."""
        for i in range(2):
            self._create_test_audio(self.songs_dir, f"song_{i+1}.wav")
        self._create_test_audio(self.vocals_dir, "my_voice.wav")

        results = self._run_album_build()

        self.assertEqual(len(results), 2)

    def test_album_build_creates_output_dir(self):
        """Output directory is created if it doesn't exist."""
        self._create_test_audio(self.songs_dir, "song.wav")
        self._create_test_audio(self.vocals_dir, "vocal.wav")

        new_output = Path(self.temp_dir) / "new_dir" / "album"

        results = self._run_album_build(output_dir=new_output)

        self.assertTrue(new_output.exists())
        self.assertEqual(len(results), 1)

    def test_album_build_no_songs_returns_empty(self):
        """Empty songs directory returns empty results."""
        self._create_test_audio(self.vocals_dir, "vocal.wav")

        results = self._run_album_build()

        self.assertEqual(results, [])

    def test_album_build_no_vocals_returns_empty(self):
        """Empty vocals directory returns empty results."""
        self._create_test_audio(self.songs_dir, "song.wav")

        results = self._run_album_build()

        self.assertEqual(results, [])

    def test_album_build_with_vocal_files_list(self):
        """Can pass explicit list of vocal file paths instead of a directory."""
        self._create_test_audio(self.songs_dir, "song_1.wav")
        v1 = self._create_test_audio(self.vocals_dir, "vocal_1.wav")

        results = self._run_album_build(vocal_files=[v1], vocals_dir=None)

        self.assertEqual(len(results), 1)

    def test_album_build_subliminal_from_voice(self):
        """Album build supports subliminal-from-voice option."""
        self._create_test_audio(self.songs_dir, "song.wav")
        self._create_test_audio(self.vocals_dir, "vocal.wav")

        results = self._run_album_build(subliminal_from_voice=True)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].exists())

    def test_album_build_forwards_entrainment_options(self):
        """entrainment_mode and waveform are passed through to the builder."""
        from src.album_builder import album_build
        from src.audio_builder import IsochronicToneGenerator

        self._create_test_audio(self.songs_dir, "song.wav")
        self._create_test_audio(self.vocals_dir, "vocal.wav")

        seen = {}
        original_build = HypnosisAudioBuilder.build

        def spy_build(self, *args, **kwargs):
            seen["mode"] = self.entrainment_mode
            seen["generator"] = type(self.binaural_generator)
            seen["waveform"] = self.binaural_generator.config.waveform
            return original_build(self, *args, **kwargs)

        with patch.object(HypnosisAudioBuilder, "build", spy_build):
            results = album_build(
                songs_dir=self.songs_dir,
                vocals_dir=self.vocals_dir,
                output_dir=self.output_dir,
                output_format="wav",
                entrainment_mode="isochronic",
                waveform="sawtooth",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(seen["mode"], "isochronic")
        self.assertIs(seen["generator"], IsochronicToneGenerator)
        self.assertEqual(seen["waveform"], "sawtooth")


class TestAlbumCLIArgs(unittest.TestCase):
    """Test CLI argument parsing for album mode."""

    def test_album_arg_exists(self):
        """--album argument is recognized by the parser."""
        from hypnosis_audio_builder import parse_args

        with patch(SYS_ARGV, ["prog", "--album", "/some/dir", "--vocals-dir", VOCALS_DIR_PATH]):
            args = parse_args()
            self.assertEqual(args.album, Path("/some/dir"))

    def test_vocals_dir_arg(self):
        """--vocals-dir argument is recognized."""
        from hypnosis_audio_builder import parse_args

        with patch(SYS_ARGV, ["prog", "--album", "/songs", "--vocals-dir", VOCALS_DIR_PATH]):
            args = parse_args()
            self.assertEqual(args.vocals_dir, Path(VOCALS_DIR_PATH))

    def test_vocals_arg_multiple_files(self):
        """--vocals accepts multiple file paths."""
        from hypnosis_audio_builder import parse_args

        with patch(SYS_ARGV, [
            "prog", "--album", "/songs",
            "--vocals", "v1.wav", "v2.wav", "v3.wav"
        ]):
            args = parse_args()
            self.assertEqual(len(args.vocals), 3)
            self.assertEqual(args.vocals[0], V1_WAV)


if __name__ == "__main__":
    unittest.main(verbosity=2)
