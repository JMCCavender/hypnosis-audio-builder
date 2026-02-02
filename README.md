# Hypnosis Audio Builder

A Python-based audio production tool for creating self-hypnosis audio tracks with binaural beats (theta, alpha, and beta waves), ambient music, and subliminal messaging.

## The Self-Hypnosis Creation Workflow

### Step 1 — Write Your Script

Write down the traits, self-image, beliefs, and identity you want to embody.

**Rules for Writing Effective Affirmations:**

| Rule | Why | Example |
|------|-----|---------|
| **Present tense** | Write as if it already exists | "I am confident" not "I will be confident" |
| **Sensory & descriptive** | Your subconscious thinks in images and feelings | "I feel warmth radiating from my chest" |
| **Emotional words** | Words that carry feeling | confident, powerful, calm, magnetic, certain |
| **No negations** | Subconscious doesn't process "not" well | "I am calm" not "I am not anxious" |
| **Absolute statements** | No maybes, hopes, or tries | "I am" not "I hope to be" |

**Example Script:**
```
I walk into every room knowing I belong.
People feel my presence before I speak.
I am calm, grounded, certain.
Opportunities flow toward me effortlessly.
```

### Step 2 — Layer in Hypnotic Techniques (Optional)

- **Hypnotic language patterns**: "imagine", "notice", "feel", "allow", "become"
- **Embedded commands**: subtle directives hidden in sentences
- **Double binds**: "you can feel this shift now or notice it later"
- **A trigger word**: something to re-access this state later

### Step 3 — Record and Layer

1. Record yourself speaking the script
2. Run through this tool to add theta waves, ambient music, and subliminals

### Step 4 — Listen Daily

**Morning and night. Every day.** The theta waves open the door. The subliminal script walks through it.

---

## Quick Start

### Installation

```bash
cd hypnosis-audio-builder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg  # macOS
```

### Basic Usage (Recommended)

```bash
python hypnosis_audio_builder.py \
  --voice morning_affirmations.wav \
  --subliminal-from-voice \
  --output morning_hypnosis.mp3
```

The `--subliminal-from-voice` flag creates the subliminal layer from your voice — same affirmations to conscious and subconscious mind.

### Session Presets

```bash
# Theta Wave Sessions (subconscious access)
# Morning (energizing, 6 Hz theta)
python hypnosis_audio_builder.py --voice script.wav --session morning --subliminal-from-voice -o morning.mp3

# Night (relaxing, 5.5 Hz theta)
python hypnosis_audio_builder.py --voice script.wav --session night --subliminal-from-voice -o night.mp3

# Sleep (deep programming, 4 Hz theta)
python hypnosis_audio_builder.py --voice script.wav --session sleep --subliminal-from-voice -o sleep.mp3

# Alpha/Beta Wave Sessions (focus and active thinking)
# Focus (relaxed focus, 10 Hz alpha)
python hypnosis_audio_builder.py --voice script.wav --session focus --subliminal-from-voice -o focus.mp3

# Active (concentration, 15 Hz beta)
python hypnosis_audio_builder.py --voice script.wav --session active --subliminal-from-voice -o active.mp3

# Subliminal-only (voice inaudible)
python hypnosis_audio_builder.py --voice script.wav --session subliminal -o subliminal_only.mp3

# List all presets
python hypnosis_audio_builder.py --list-presets
```

### Validate Your Script

```bash
python hypnosis_audio_builder.py --validate-script affirmations.txt
```

### Test Setup

```bash
python hypnosis_audio_builder.py --test
```

---

## Audio Layers

| Layer | Volume | Purpose |
|-------|--------|---------|
| **Voice** | 0 dB | Main affirmations to conscious mind |
| **Binaural Beats** | -10 dB | Brainwave entrainment (theta/alpha/beta) |
| **Ambient Music** | -20 dB | Calming atmosphere |
| **Subliminal** | -35 dB | Same affirmations beneath conscious hearing |

### Brainwave Frequencies

| Range | Hz | State | Best For |
|-------|-----|-------|----------|
| **Theta** | 4.0 | Deep meditation | Overnight, deep programming |
| **Theta** | 5.5 | Creativity | Visualization, imagery |
| **Theta** | 6.0 | Learning | Default, general use |
| **Theta** | 7.5 | Light meditation | Relaxation |
| **Alpha** | 10.0 | Relaxed focus | Learning, calm awareness |
| **Beta** | 15.0 | Active thinking | Concentration, mental clarity |

**Use headphones!** Binaural beats require separate frequencies per ear.

---

## Command Reference

```bash
# Core options
--voice, -v FILE          Voice recording
--session, -S TYPE        morning/night/sleep/focus/active/subliminal/standard
--subliminal-from-voice   Create subliminal from voice (RECOMMENDED)
--output, -o FILE         Output file

# Customize
--theta-freq FREQ         Override frequency (4-30 Hz: theta/alpha/beta)
--ambient-music FILE      Custom ambient
--validate-script FILE    Check affirmation script

# Info
--list-presets            Show all presets
--test                    Test with generated audio
--help                    Full help
```

---

## Examples

### Morning + Night Pair
```bash
python hypnosis_audio_builder.py --voice script.wav --session morning --subliminal-from-voice -o morning.mp3
python hypnosis_audio_builder.py --voice script.wav --session night --subliminal-from-voice -o night.mp3
```

### With Custom Ambient
```bash
python hypnosis_audio_builder.py \
  --voice confidence.wav \
  --session morning \
  --subliminal-from-voice \
  --ambient-music peaceful_piano.mp3 \
  -o confidence_morning.mp3
```

### Batch Process
```bash
python hypnosis_audio_builder.py \
  --batch ./recordings/ \
  --output-dir ./processed/ \
  --session morning \
  --subliminal-from-voice
```

---

## Python API

```python
from src import HypnosisAudioBuilder, ScriptValidator
from pathlib import Path

# Validate script
analysis = ScriptValidator.validate(Path("script.txt").read_text())
print(ScriptValidator.format_report(analysis))

# Build with morning preset
builder = HypnosisAudioBuilder(session_type="morning")
builder.build(
    voice_path=Path("voice.wav"),
    output_path=Path("output.mp3"),
    subliminal_from_voice=True
)
```

---

## Best Practices

### Writing
- ✅ "I am confident" | ❌ "I will be confident"
- ✅ "I feel calm" | ❌ "I am not anxious"
- ✅ "Success flows to me" | ❌ "I hope to succeed"

### Listening
- Morning and night, same track
- 21-30 days for best results
- **Headphones required** for binaural effect
- Never while driving

---

## Troubleshooting

**ffmpeg not found**: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu)

**Binaural beats not noticeable**: They're subtle by design. Use headphones.

**Script validation issues**: Rewrite negations and future tense.

---

*Each morning you wake up slightly more like the person you programmed yourself to become.*
