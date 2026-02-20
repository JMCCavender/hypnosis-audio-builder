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
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_builder import create_test_voice


class TestPairSongsWithVocals(unittest.TestCase):
    """Test the pairing logic that maps songs to vocals with cycling."""

    def test_equal_counts(self):
        """When songs == vocals, each pairs 1:1."""
        from src.album_builder import pair_songs_with_vocals

        songs = [Path("song1.mp3"), Path("song2.mp3"), Path("song3.mp3")]
        vocals = [Path("vocal1.wav"), Path("vocal2.wav"), Path("vocal3.wav")]

        pairs = pair_songs_with_vocals(songs, vocals)

        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0], (Path("song1.mp3"), Path("vocal1.wav")))
        self.assertEqual(pairs[1], (Path("song2.mp3"), Path("vocal2.wav")))
        self.assertEqual(pairs[2], (Path("song3.mp3"), Path("vocal3.wav")))

    def test_more_songs_than_vocals_cycles(self):
        """When more songs than vocals, vocals cycle from the beginning."""
        from src.album_builder import pair_songs_with_vocals

        songs = [Path(f"song{i}.mp3") for i in range(5)]
        vocals = [Path("vocal1.wav"), Path("vocal2.wav")]

        pairs = pair_songs_with_vocals(songs, vocals)

        self.assertEqual(len(pairs), 5)
        # Vocals cycle: v1, v2, v1, v2, v1
        self.assertEqual(pairs[0][1], Path("vocal1.wav"))
        self.assertEqual(pairs[1][1], Path("vocal2.wav"))
        self.assertEqual(pairs[2][1], Path("vocal1.wav"))
        self.assertEqual(pairs[3][1], Path("vocal2.wav"))
        self.assertEqual(pairs[4][1], Path("vocal1.wav"))

    def test_more_vocals_than_songs(self):
        """When more vocals than songs, extra vocals are unused."""
        from src.album_builder import pair_songs_with_vocals

        songs = [Path("song1.mp3")]
        vocals = [Path("vocal1.wav"), Path("vocal2.wav"), Path("vocal3.wav")]

        pairs = pair_songs_with_vocals(songs, vocals)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], (Path("song1.mp3"), Path("vocal1.wav")))

    def test_single_vocal_repeats_for_all_songs(self):
        """A single vocal repeats across all songs."""
        from src.album_builder import pair_songs_with_vocals

        songs = [Path(f"song{i}.mp3") for i in range(4)]
        vocals = [Path("vocal.wav")]

        pairs = pair_songs_with_vocals(songs, vocals)

        self.assertEqual(len(pairs), 4)
        for pair in pairs:
            self.assertEqual(pair[1], Path("vocal.wav"))

    def test_empty_songs_returns_empty(self):
        """No songs means no pairs."""
        from src.album_builder import pair_songs_with_vocals

        pairs = pair_songs_with_vocals([], [Path("vocal.wav")])
        self.assertEqual(pairs, [])

    def test_empty_vocals_returns_empty(self):
        """No vocals means no pairs."""
        from src.album_builder import pair_songs_with_vocals

        pairs = pair_songs_with_vocals([Path("song.mp3")], [])
        self.assertEqual(pairs, [])

    def test_preserves_song_order(self):
        """Songs maintain their sorted order in the pairing."""
        from src.album_builder import pair_songs_with_vocals

        songs = [Path("c_song.mp3"), Path("a_song.mp3"), Path("b_song.mp3")]
        vocals = [Path("v1.wav"), Path("v2.wav"), Path("v3.wav")]

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
            song_path=Path("song.mp3"),
            vocal_path=Path("vocal.wav"),
            output_dir=Path("/out"),
            output_format="wav",
            track_number=3,
        )

        self.assertTrue(result.name.startswith("03_"))

    def test_format_respected(self):
        """Output uses the specified format extension."""
        from src.album_builder import generate_album_output_path

        result = generate_album_output_path(
            song_path=Path("song.mp3"),
            vocal_path=Path("vocal.wav"),
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

    def test_album_build_creates_output_files(self):
        """Album build creates one output file per song."""
        from src.album_builder import album_build

        # Create 3 songs and 2 vocals
        for i in range(3):
            self._create_test_audio(self.songs_dir, f"song_{i+1}.wav")
        for i in range(2):
            self._create_test_audio(self.vocals_dir, f"vocal_{i+1}.wav")

        results = album_build(
            songs_dir=self.songs_dir,
            vocals_dir=self.vocals_dir,
            output_dir=self.output_dir,
            output_format="wav",
            session_type="standard",
        )

        self.assertEqual(len(results), 3)
        for path in results:
            self.assertTrue(path.exists())
            self.assertTrue(path.stat().st_size > 0)

    def test_album_build_with_single_vocal(self):
        """Single vocal is used for all songs."""
        from src.album_builder import album_build

        for i in range(2):
            self._create_test_audio(self.songs_dir, f"song_{i+1}.wav")
        self._create_test_audio(self.vocals_dir, "my_voice.wav")

        results = album_build(
            songs_dir=self.songs_dir,
            vocals_dir=self.vocals_dir,
            output_dir=self.output_dir,
            output_format="wav",
            session_type="standard",
        )

        self.assertEqual(len(results), 2)

    def test_album_build_creates_output_dir(self):
        """Output directory is created if it doesn't exist."""
        from src.album_builder import album_build

        self._create_test_audio(self.songs_dir, "song.wav")
        self._create_test_audio(self.vocals_dir, "vocal.wav")

        new_output = Path(self.temp_dir) / "new_dir" / "album"

        results = album_build(
            songs_dir=self.songs_dir,
            vocals_dir=self.vocals_dir,
            output_dir=new_output,
            output_format="wav",
            session_type="standard",
        )

        self.assertTrue(new_output.exists())
        self.assertEqual(len(results), 1)

    def test_album_build_no_songs_returns_empty(self):
        """Empty songs directory returns empty results."""
        from src.album_builder import album_build

        self._create_test_audio(self.vocals_dir, "vocal.wav")

        results = album_build(
            songs_dir=self.songs_dir,
            vocals_dir=self.vocals_dir,
            output_dir=self.output_dir,
            output_format="wav",
            session_type="standard",
        )

        self.assertEqual(results, [])

    def test_album_build_no_vocals_returns_empty(self):
        """Empty vocals directory returns empty results."""
        from src.album_builder import album_build

        self._create_test_audio(self.songs_dir, "song.wav")

        results = album_build(
            songs_dir=self.songs_dir,
            vocals_dir=self.vocals_dir,
            output_dir=self.output_dir,
            output_format="wav",
            session_type="standard",
        )

        self.assertEqual(results, [])

    def test_album_build_with_vocal_files_list(self):
        """Can pass explicit list of vocal file paths instead of a directory."""
        from src.album_builder import album_build

        self._create_test_audio(self.songs_dir, "song_1.wav")
        v1 = self._create_test_audio(self.vocals_dir, "vocal_1.wav")

        results = album_build(
            songs_dir=self.songs_dir,
            vocal_files=[v1],
            output_dir=self.output_dir,
            output_format="wav",
            session_type="standard",
        )

        self.assertEqual(len(results), 1)

    def test_album_build_subliminal_from_voice(self):
        """Album build supports subliminal-from-voice option."""
        from src.album_builder import album_build

        self._create_test_audio(self.songs_dir, "song.wav")
        self._create_test_audio(self.vocals_dir, "vocal.wav")

        results = album_build(
            songs_dir=self.songs_dir,
            vocals_dir=self.vocals_dir,
            output_dir=self.output_dir,
            output_format="wav",
            session_type="standard",
            subliminal_from_voice=True,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].exists())


class TestAlbumCLIArgs(unittest.TestCase):
    """Test CLI argument parsing for album mode."""

    def test_album_arg_exists(self):
        """--album argument is recognized by the parser."""
        from hypnosis_audio_builder import parse_args

        with patch("sys.argv", ["prog", "--album", "/some/dir", "--vocals-dir", "/vocals"]):
            args = parse_args()
            self.assertEqual(args.album, Path("/some/dir"))

    def test_vocals_dir_arg(self):
        """--vocals-dir argument is recognized."""
        from hypnosis_audio_builder import parse_args

        with patch("sys.argv", ["prog", "--album", "/songs", "--vocals-dir", "/vocals"]):
            args = parse_args()
            self.assertEqual(args.vocals_dir, Path("/vocals"))

    def test_vocals_arg_multiple_files(self):
        """--vocals accepts multiple file paths."""
        from hypnosis_audio_builder import parse_args

        with patch("sys.argv", [
            "prog", "--album", "/songs",
            "--vocals", "v1.wav", "v2.wav", "v3.wav"
        ]):
            args = parse_args()
            self.assertEqual(len(args.vocals), 3)
            self.assertEqual(args.vocals[0], Path("v1.wav"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
