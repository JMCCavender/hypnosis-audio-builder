# Classical Music Library

A curated set of **optimistic, positive, public-domain classical music** for
building hypnosis/affirmation tracks — free to use, no royalties, no attribution
strings for the compositions themselves.

## What "license-free" means here

Two separate copyrights apply to any piece of recorded music:

1. **The composition** — every piece in this library was written by a composer
   who died well over 70 years ago (Vivaldi, Bach, Mozart, Beethoven, Handel,
   Pachelbel, Mendelssohn, Dvořák, Grieg). These compositions are firmly in the
   **public domain** worldwide.
2. **The recording** (the specific performance). This has its own copyright. The
   recordings referenced here come from **[Musopen](https://musopen.org/)**,
   a non‑profit that raised money to commission and **release recordings into the
   public domain**, hosted on the **[Internet Archive Musopen collection](https://archive.org/details/musopen)**.

> Always confirm the license note on a recording's source page before commercial
> use. Public-domain *compositions* are unrestricted; double-check the *recording*.

## The catalog

`classical-library.json` is the single source of truth (also mirrored in the
in-browser mixer). Each entry has a title, composer, mood tags, a source page,
and an `archive_query` used by the downloader.

Moods: `bright`, `uplifting`, `triumphant`, `energetic`, `playful`, `serene`.

## Getting the music

### Option A — one command (recommended)

```bash
python fetch_music.py            # download the whole library into music/
python fetch_music.py --list     # preview the catalog without downloading
python fetch_music.py --limit 3  # just the first few
```

The script resolves a real public-domain recording for each piece from the
Internet Archive and saves numbered files into `music/`. Run it in an
environment with internet access to `archive.org`.

### Option B — in the mixer UI

Open `subliminal-mixer-808.html` and use the **Classical Music Library** panel:

- **SOURCE ↗** opens the recording's page (Musopen / archive.org) to download.
- Paste a **direct audio URL** into the "Load URL" box to pull it straight in
  (works when the host allows cross-origin requests, e.g. archive.org).
- After downloading files locally, use **Load Songs** to add them.

### Option C — manual

Click the **SOURCE** links in `classical-library.json`, download the recordings,
and drop them in this folder.

## Using the music

```bash
# Single track
python hypnosis_audio_builder.py --voice affirmations.wav \
  --ambient-music music/01_antonio-vivaldi_spring-la-primavera-op-8-no-1.mp3 \
  --subliminal-from-voice -o morning.mp3

# Whole album (pairs every song with your vocals)
python hypnosis_audio_builder.py --album music/ --vocals affirmations.wav \
  --subliminal-from-voice
```
