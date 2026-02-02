#!/usr/bin/env python3
"""
Test suite for Hypnosis Audio Builder

Run with: python -m pytest tests/test_audio_builder.py -v
Or:       python tests/test_audio_builder.py
"""

import sys
import unittest
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_builder import (
    HypnosisAudioBuilder,
    MixLevels,
    BinauralConfig,
    BinauralBeatGenerator,
    AmbientMusicManager,
    create_test_voice,
)


class TestBinauralConfig(unittest.TestCase):
    """Test binaural beat configuration."""
    
    def test_default_config(self):
        config = BinauralConfig()
        self.assertEqual(config.base_frequency, 200.0)
        self.assertEqual(config.theta_frequency, 6.0)
        self.assertEqual(config.sample_rate, 44100)
    
    def test_frequencies(self):
        config = BinauralConfig(base_frequency=200.0, theta_frequency=6.0)
        self.assertEqual(config.left_frequency, 200.0)
        self.assertEqual(config.right_frequency, 206.0)  # 200 + 6
    
    def test_custom_theta(self):
        config = BinauralConfig(theta_frequency=4.0)
        self.assertEqual(config.right_frequency, 204.0)


class TestMixLevels(unittest.TestCase):
    """Test mix level configuration."""
    
    def test_default_levels(self):
        levels = MixLevels()
        self.assertEqual(levels.voice, 0.0)
        self.assertEqual(levels.binaural, -10.0)
        self.assertEqual(levels.ambient, -20.0)
        self.assertEqual(levels.subliminal, -35.0)
    
    def test_custom_levels(self):
        levels = MixLevels(voice=-3.0, binaural=-15.0)
        self.assertEqual(levels.voice, -3.0)
        self.assertEqual(levels.binaural, -15.0)


class TestBinauralBeatGenerator(unittest.TestCase):
    """Test binaural beat generation."""
    
    def setUp(self):
        self.config = BinauralConfig()
        self.generator = BinauralBeatGenerator(self.config)
    
    def test_generate_short_audio(self):
        audio = self.generator.generate(duration_seconds=1.0)
        self.assertIsNotNone(audio)
        # Should be approximately 1 second (with some tolerance for processing)
        self.assertAlmostEqual(len(audio), 1000, delta=50)
    
    def test_generate_stereo(self):
        audio = self.generator.generate(duration_seconds=1.0)
        self.assertEqual(audio.channels, 2)
    
    def test_progress_callback(self):
        progress_values = []
        
        def callback(progress):
            progress_values.append(progress)
        
        audio = self.generator.generate(
            duration_seconds=1.0,
            progress_callback=callback
        )
        
        # Should have received progress updates
        self.assertTrue(len(progress_values) > 0)
        # Final progress should be 1.0
        self.assertEqual(progress_values[-1], 1.0)


class TestAmbientMusicManager(unittest.TestCase):
    """Test ambient music management."""
    
    def setUp(self):
        self.manager = AmbientMusicManager()
    
    def test_generate_placeholder(self):
        audio = self.manager.generate_placeholder(duration_seconds=5.0)
        self.assertIsNotNone(audio)
        # Should be approximately 5 seconds
        self.assertAlmostEqual(len(audio), 5000, delta=100)
    
    def test_placeholder_is_stereo(self):
        audio = self.manager.generate_placeholder(duration_seconds=1.0)
        self.assertEqual(audio.channels, 2)


class TestCreateTestVoice(unittest.TestCase):
    """Test the test voice creation utility."""
    
    def test_create_test_voice(self):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result = create_test_voice(output_path, duration_seconds=2.0)
            self.assertTrue(result.exists())
            self.assertTrue(result.stat().st_size > 0)
        finally:
            output_path.unlink(missing_ok=True)


class TestHypnosisAudioBuilder(unittest.TestCase):
    """Test the main audio builder class."""
    
    def setUp(self):
        self.builder = HypnosisAudioBuilder()
        
        # Create a test voice file
        self.temp_dir = tempfile.mkdtemp()
        self.test_voice_path = Path(self.temp_dir) / "test_voice.wav"
        create_test_voice(self.test_voice_path, duration_seconds=3.0)
    
    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_voice(self):
        track = self.builder.load_voice(self.test_voice_path)
        self.assertEqual(track.name, "voice")
        self.assertIsNotNone(track.audio)
    
    def test_generate_binaural_track(self):
        track = self.builder.generate_binaural_track(
            duration_seconds=2.0,
            theta_frequency=6.0
        )
        self.assertEqual(track.name, "binaural")
        self.assertIsNotNone(track.audio)
    
    def test_load_ambient_placeholder(self):
        track = self.builder.load_ambient_track(
            ambient_path=None,
            duration_seconds=2.0,
            use_placeholder=True
        )
        self.assertEqual(track.name, "ambient")
        self.assertIsNotNone(track.audio)
    
    def test_full_build(self):
        output_path = Path(self.temp_dir) / "output.mp3"
        
        result = self.builder.build(
            voice_path=self.test_voice_path,
            output_path=output_path,
            theta_frequency=6.0
        )
        
        self.assertTrue(result.exists())
        self.assertTrue(result.stat().st_size > 0)
    
    def test_custom_mix_levels(self):
        custom_levels = MixLevels(
            voice=-3.0,
            binaural=-15.0,
            ambient=-25.0,
            subliminal=-40.0
        )
        
        builder = HypnosisAudioBuilder(mix_levels=custom_levels)
        self.assertEqual(builder.mix_levels.voice, -3.0)
        self.assertEqual(builder.mix_levels.binaural, -15.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_missing_voice_file(self):
        builder = HypnosisAudioBuilder()
        
        with self.assertRaises(FileNotFoundError):
            builder.load_voice(Path("/nonexistent/file.wav"))
    
    def test_invalid_theta_frequency(self):
        # The validation is done at CLI level, but config accepts any float
        config = BinauralConfig(theta_frequency=100.0)
        # Generator should still work, just won't produce theta waves
        generator = BinauralBeatGenerator(config)
        audio = generator.generate(duration_seconds=1.0)
        self.assertIsNotNone(audio)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
