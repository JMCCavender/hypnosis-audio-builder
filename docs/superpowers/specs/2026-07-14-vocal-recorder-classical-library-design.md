# Design: Vocal Recorder + Classical Library for the Subliminal Mixer

**Date:** 2026-07-14
**Target file:** `subliminal-mixer-808.html` (single self-contained page, no build step, no server)
**Status:** Approved design — ready for implementation plan

## Summary

Add two additive features to the browser-based Subliminal-808 mixer:

1. **Vocal Recorder** — record affirmations directly in the browser via the microphone,
   preview the take, keep-or-redo, add approved takes to the vocals, and download each
   take as a WAV.
2. **Classical Library** — a curated, browse-and-download panel of public-domain / CC0
   classical tracks suitable for hypnosis backgrounds. No audio is bundled; the panel
   links to vetted sources.

Both features are purely additive. Neither changes the existing play/render/export
audio engine. The page remains a single self-contained HTML file that works by
double-clicking (opened via `file://` or `open`).

## Existing architecture (verified)

The mixer is a **fully client-side Web Audio application**:

- Audio files are loaded through `loadAudioFile(file, type)` (line ~1348), which reads an
  `ArrayBuffer`, calls `audioCtx.decodeAudioData()`, and stores the result in a module
  buffer — `musicBuffer` or `voiceBuffer` (line ~1094).
- Live playback and the final export both build a graph from those buffers. Export uses
  an `OfflineAudioContext`, renders, and converts to WAV via `bufferToWav(buffer)`
  (line ~1828), then triggers a download.
- Markup anchors: the **Vocals (Affirmations)** panel is a `.file-box.voice` containing
  `#voiceInput` (a `<input type="file" multiple>`) and `#voiceBtn` "Load Vocals"
  (lines ~898-905). An **ALBUM TRACKS** section (`#trackListSection`, hidden by default,
  lines ~910-927) exists in markup.

**Known constraint — album multi-track is only partially wired.** The file inputs carry
`multiple`, and the ALBUM TRACKS UI exists, but the change handlers currently consume
only `e.target.files[0]` (lines ~1893-1899) into the single `voiceBuffer`. So at runtime
the mixer effectively tracks **one** vocal buffer today. The recorder must integrate
through the same vocal-add entry point rather than assume a working multi-track list.

## Feature 1 — Vocal Recorder

### Placement
Inside the existing `.file-box.voice` panel, add a red **● Record** button in the
`.file-row`, next to **Load Vocals**. No new page or layout change.

### User flow
```
[● Record]  ──click──▶  mic permission prompt (first time only)
                        │
   RECORDING ● 0:07     │  live input-level meter (catches silence/clipping)
   [■ Stop]             │
                        ▼ Stop
   PREVIEW: ▸ play take │  "Use this take"   "Re-record"   "Discard"
                        ▼ Use this take
   → added to vocals (plays in live mix)  +  [⤓ Download take]
```

- **Preview → keep or redo:** nothing is added to the vocals until the user clicks
  **Use this take**. **Re-record** discards and restarts capture. **Discard** cancels.
- **Multi-take intent:** each approved take is intended to append (album-friendly). Because
  the album multi-track model is only partially wired (see constraint above), the recorder
  hooks into the same entry point uploads use. If, at implementation time, append-to-list
  is not functional, an approved take **replaces** the current vocal buffer (same behavior
  as uploading a new file today), and this limitation is surfaced to the user with a short
  note rather than silently dropping takes. Wiring up true multi-take append beyond the
  existing upload behavior is out of scope for this spec.
- **Download take:** offers the take as a **WAV** file (e.g. `affirmation-take-1.wav`),
  compatible with the Python CLI and universally decodable.

### Technical approach
- Capture with `MediaRecorder` over a `getUserMedia({ audio: true })` stream. Collect
  `dataavailable` chunks into a `Blob` (MediaRecorder's native container, typically
  webm/opus).
- **Preview** plays the take from an object URL (no re-encoding needed).
- **Use this take:** decode the recorded `Blob`'s `ArrayBuffer` with
  `audioCtx.decodeAudioData()` into an `AudioBuffer`, then feed it through the same path
  `loadAudioFile` uses to populate the vocal buffer. Reuse existing code; do not fork the
  load logic.
- **Download take:** convert the decoded `AudioBuffer` to WAV using the existing
  `bufferToWav()` function so the saved file is clean WAV rather than webm.
- **Live level meter:** an `AnalyserNode` on the input stream drives a small meter during
  recording so the user can see they are not capturing silence or clipping.

### Isolation
Implement as a small self-contained recorder module (a cohesive group of functions:
`startRecording`, `stopRecording`, `getTakeBlob`/`getTakeBuffer`, plus its UI wiring).
It communicates with the rest of the app only through: (a) the existing vocal-add entry
point, and (b) the existing `bufferToWav()` helper. It does not touch the play/render/
export graph.

### Error handling
- **Permission denied / no microphone:** inline message in the vocals panel
  ("Microphone access needed to record"); no crash, no alert spam.
- **No `MediaRecorder` support:** hide or disable the Record button with a short note.
- **Empty/near-silent take:** optional non-blocking warning before "Use this take".
- **Decode failure of the recorded blob:** inline error, offer Re-record.

## Feature 2 — Classical Library (browse + download)

### Placement
A collapsible **"Classical Library — public domain"** section near the
**Songs (Album Tracks)** area, opened by a **Browse Classical** button next to
**Load Songs**.

### Contents
A curated list of ~12 calming, genuinely public-domain / CC0 pieces suited to hypnosis
backgrounds (e.g. Satie *Gymnopédies*, Debussy *Clair de Lune*, Chopin nocturnes, slow
Bach movements). Each row shows: title · composer, source/performer, approximate length,
license label (Public Domain / CC0), and a **Get track ↗** link to the source download
page (e.g. Musopen, Archive.org).

### Data model
The track list is a small JavaScript array (or embedded JSON) inside the HTML so the file
stays self-contained and the list is trivial to extend. Each entry:
`{ title, composer, source, sourceUrl, approxLength, license }`.

### User flow
Click **Get track ↗** → source page opens in a new browser tab → user downloads the file
once → loads it via the existing **Load Songs**. Two steps, but always works and stays
fully legal. No audio is bundled in the repo.

### Legal
Only Public-Domain or CC0 sources. Each link is verified during implementation to resolve
to a PD/CC0 recording, and the license is shown inline. If a candidate track's license
cannot be confirmed, it is dropped from the list.

## Shared principles

- **Additive only.** New self-contained modules touch the existing engine solely through
  its current entry points (vocal-add + `bufferToWav`). The play/render/export path is
  unchanged.
- **Single self-contained HTML file.** No server, no build step; still opens by
  double-clicking.
- **No new dependencies.** Uses built-in browser APIs (`MediaRecorder`, `getUserMedia`,
  `AnalyserNode`, Web Audio).

## Decisions locked in

- Downloaded takes are **WAV** (via existing `bufferToWav`), not webm — for CLI
  compatibility.
- The library ships as **~12 curated links**, extensible later. **No audio bundled.**
- The recorder shows a **live input-level meter** during capture.
- Take flow is **preview → keep or redo**; only approved takes are added.

## Testing / verification

The mixer has no automated test harness today, and this remains a single HTML file.
Verification is by driving the app in a browser:

1. Record a take → live meter moves → Stop → preview plays → **Use this take** → the vocal
   appears and plays in the live mix → **Export** produces a WAV containing the vocal.
2. **Download take** produces a valid, playable WAV (verify with `ffprobe`).
3. Permission-denied path shows the inline message and does not crash.
4. Classical Library opens, rows render, and **Get track ↗** opens the correct source tab.

Adding an automated test setup is out of scope unless separately requested.

## Out of scope

- Completing/repairing the partially-wired album multi-track model beyond current upload
  behavior.
- Bundling or redistributing any audio files.
- One-click streaming of remote library tracks into the mixer (rejected due to
  cross-origin reliability and `file://` limitations).
- Server-side or Python-CLI changes.
