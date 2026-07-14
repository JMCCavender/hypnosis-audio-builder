# Vocal Recorder + Classical Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-browser microphone vocal recorder and a curated public-domain Baroque classical library to the self-contained Subliminal-808 mixer (`subliminal-mixer-808.html`).

**Architecture:** Both features are additive modules inside the single existing HTML file. The recorder captures audio with `MediaRecorder`, previews it, and on approval decodes it into the mixer's existing `voiceBuffer` (the same slot file-upload fills) and can download it as WAV via the existing `bufferToWav()`. The classical library is a static data array rendered into a browse-and-download panel that links out to vetted Public-Domain sources; no audio is bundled. Nothing in the play/render/export engine changes.

**Tech Stack:** Vanilla JS, Web Audio API (`AudioContext`, `OfflineAudioContext`, `AnalyserNode`), `MediaRecorder`, `getUserMedia`. No new dependencies, no build step.

## Global Constraints

- **Single self-contained file:** all changes are to `subliminal-mixer-808.html`. No new runtime files, no server dependency, no external libraries.
- **No audio bundled:** the classical library ships links only, to Public-Domain / CC0 sources.
- **Additive only:** do not modify the existing play/render/export graph. Integrate only through the existing vocal-load path (`voiceBuffer`) and `bufferToWav()`.
- **Library repertoire:** Baroque era (~1600–1750) only, slow ~60 BPM largo/adagio movements.
- **Downloaded takes are WAV** (via existing `bufferToWav`), not the raw MediaRecorder container.
- **Secure-context requirement:** the recorder needs `getUserMedia`, which requires a secure context. All recorder verification is done by serving the folder over `http://localhost` (see Verification Harness), never `file://`.

## Verification Harness (no automated test runner)

This is a single static HTML file with no JS test framework, and adding one is out of scope (per spec). Each task therefore ends with a **browser-driven verification** step: concrete actions and the exact observable result expected.

Standard setup used by verification steps:

```bash
# From the repo root, serve the folder (localhost = secure context, required for mic):
python3 -m http.server 8765
# Then open in Chrome: http://localhost:8765/subliminal-mixer-808.html
```

Console is inspected via the browser devtools console (or the claude-in-chrome `read_console_messages` tool when driving automatically). Downloaded WAVs are validated with `ffprobe <file>`.

## File Structure

- **Modify only:** `subliminal-mixer-808.html`
  - Markup: add a Record button + recorder sub-panel inside `.file-box.voice` (~line 898-905); add a "Browse Classical" button near `.file-box.music` (~line 886-894) and a collapsible classical-library section.
  - CSS: add styles for recorder states (idle/recording/preview) and the library panel, colocated with existing styles (before the closing `</style>`, ~line 251 area is the file-input block; append new rules near related selectors).
  - JS: add a **recorder module** (capture/preview/keep-redo/download) and a **classical library module** (data array + render + toggle), wired in the existing event-binding block (~line 1882-1935), reusing `loadAudioFile`'s decode path and `bufferToWav()`.

All code lives in the one file; "Create/Modify" paths below all refer to `subliminal-mixer-808.html` with the anchoring line ranges noted.

---

### Task 1: Recorder UI + capture + preview

Adds the Record button, the recorder sub-panel, live timer + input-level meter during capture, and preview playback of the just-recorded take. No integration into the mix yet.

**Files:**
- Modify: `subliminal-mixer-808.html` — markup in `.file-box.voice` (~898-905); CSS near file-input styles (~251); JS recorder module appended before the final event-binding block (~1880); event wiring in the binding block (~1897).

**Interfaces:**
- Consumes: existing `audioCtx` / `initAudio()` (~line 1349 shows `if (!audioCtx) initAudio();`); existing `#voiceName` / `#ledDisplay` display elements.
- Produces (used by Task 2):
  - `recorderState` object: `{ mediaRecorder, stream, chunks: [], takeBlob: null, timerId, analyser, rafId }`
  - `async function startRecording()` — requests mic, starts capture, drives timer + meter.
  - `function stopRecording()` — stops capture, assembles `recorderState.takeBlob`, shows preview.
  - `function resetRecorder()` — tears down stream/analyser/timer, clears state, returns UI to idle.
  - DOM ids: `#recordBtn`, `#recorderPanel`, `#recTimer`, `#recMeterFill`, `#takePreview` (an `<audio>`), `#useTakeBtn`, `#reRecordBtn`, `#discardTakeBtn`, `#downloadTakeBtn`.

- [ ] **Step 1: Add the recorder markup**

Inside the `.file-box.voice` block, after the existing `.file-row` (the one containing `#voiceInput` and `#voiceBtn`, ~line 905), add the Record button into that same `.file-row` and a recorder panel below it:

```html
<!-- add as the last child of the existing .file-row in .file-box.voice -->
<button class="file-btn record-btn" id="recordBtn">● Record</button>

<!-- add immediately after that .file-row, still inside .file-box.voice -->
<div class="recorder-panel" id="recorderPanel" data-mode="idle" style="display:none">
    <div class="rec-status-row">
        <span class="rec-dot"></span>
        <span class="rec-timer" id="recTimer">0:00</span>
        <div class="rec-meter"><div class="rec-meter-fill" id="recMeterFill" style="width:0%"></div></div>
        <button class="rec-stop-btn" id="recStopBtn">■ Stop</button>
    </div>
    <div class="rec-preview-row" id="recPreviewRow" style="display:none">
        <audio class="take-preview" id="takePreview" controls></audio>
        <div class="rec-preview-actions">
            <button class="file-btn" id="useTakeBtn">Use this take</button>
            <button class="file-btn" id="reRecordBtn">Re-record</button>
            <button class="file-btn" id="downloadTakeBtn">⤓ Download take</button>
            <button class="file-btn ghost" id="discardTakeBtn">Discard</button>
        </div>
    </div>
    <div class="rec-error" id="recError" style="display:none"></div>
</div>
```

- [ ] **Step 2: Add the recorder CSS**

Append near the other file-input styles (before `</style>`):

```css
.record-btn { background:#7a1f1f; }
.record-btn.armed { background:#c0392b; }
.recorder-panel { margin-top:8px; padding:8px; border:1px solid #333; border-radius:6px; background:#141414; }
.rec-status-row, .rec-preview-actions { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.rec-dot { width:10px; height:10px; border-radius:50%; background:#c0392b; animation:recpulse 1s infinite; }
@keyframes recpulse { 0%,100%{opacity:1;} 50%{opacity:.3;} }
.rec-timer { font-variant-numeric:tabular-nums; min-width:44px; }
.rec-meter { flex:1; height:8px; background:#000; border-radius:4px; overflow:hidden; min-width:80px; }
.rec-meter-fill { height:100%; width:0%; background:linear-gradient(90deg,#2ecc71,#f1c40f,#e74c3c); transition:width .05s linear; }
.rec-preview-row { margin-top:8px; }
.take-preview { width:100%; }
.file-btn.ghost { background:transparent; border:1px solid #444; }
.rec-error { margin-top:6px; color:#ef4444; font-size:12px; }
```

- [ ] **Step 3: Add the recorder module JS (capture + preview only)**

Append this module before the final event-binding block (~line 1880):

```javascript
// ===== VOCAL RECORDER =====
const recorderState = { mediaRecorder:null, stream:null, chunks:[], takeBlob:null, timerId:null, analyser:null, rafId:null, startMs:0 };

function showRecError(msg) {
    const el = document.getElementById('recError');
    el.textContent = msg; el.style.display = 'block';
}

function drawMeter() {
    if (!recorderState.analyser) return;
    const buf = new Uint8Array(recorderState.analyser.fftSize);
    recorderState.analyser.getByteTimeDomainData(buf);
    let peak = 0;
    for (let i=0;i<buf.length;i++){ const v = Math.abs(buf[i]-128)/128; if (v>peak) peak=v; }
    document.getElementById('recMeterFill').style.width = Math.min(100, peak*140) + '%';
    recorderState.rafId = requestAnimationFrame(drawMeter);
}

function tickTimer() {
    const secs = Math.floor((performance.now() - recorderState.startMs)/1000);
    document.getElementById('recTimer').textContent = Math.floor(secs/60) + ':' + String(secs%60).padStart(2,'0');
}

async function startRecording() {
    if (!audioCtx) initAudio();
    document.getElementById('recError').style.display = 'none';
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio:true });
    } catch (err) {
        showRecError('Microphone access needed to record (' + err.name + ').');
        return;
    }
    recorderState.stream = stream;
    recorderState.chunks = [];
    recorderState.takeBlob = null;
    const mr = new MediaRecorder(stream);
    recorderState.mediaRecorder = mr;
    mr.ondataavailable = (e) => { if (e.data && e.data.size) recorderState.chunks.push(e.data); };
    mr.onstop = onRecordingStopped;
    mr.start();

    // level meter
    const src = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    recorderState.analyser = analyser;
    drawMeter();

    // timer + UI
    recorderState.startMs = performance.now();
    document.getElementById('recTimer').textContent = '0:00';
    recorderState.timerId = setInterval(tickTimer, 250);

    const panel = document.getElementById('recorderPanel');
    panel.style.display = 'block';
    panel.dataset.mode = 'recording';
    document.getElementById('recPreviewRow').style.display = 'none';
    document.getElementById('recordBtn').classList.add('armed');
    document.getElementById('ledDisplay').textContent = 'RECORDING';
}

function stopRecording() {
    if (recorderState.mediaRecorder && recorderState.mediaRecorder.state !== 'inactive') {
        recorderState.mediaRecorder.stop();
    }
}

function onRecordingStopped() {
    // stop meter + timer
    if (recorderState.rafId) cancelAnimationFrame(recorderState.rafId);
    if (recorderState.timerId) clearInterval(recorderState.timerId);
    recorderState.rafId = null; recorderState.timerId = null;
    if (recorderState.stream) recorderState.stream.getTracks().forEach(t => t.stop());
    document.getElementById('recMeterFill').style.width = '0%';
    document.getElementById('recordBtn').classList.remove('armed');

    recorderState.takeBlob = new Blob(recorderState.chunks, { type: recorderState.chunks[0] ? recorderState.chunks[0].type : 'audio/webm' });
    const url = URL.createObjectURL(recorderState.takeBlob);
    const audio = document.getElementById('takePreview');
    audio.src = url;

    document.getElementById('recorderPanel').dataset.mode = 'preview';
    document.getElementById('recPreviewRow').style.display = 'block';
    document.getElementById('ledDisplay').textContent = 'PREVIEW TAKE';
}

function resetRecorder() {
    if (recorderState.rafId) cancelAnimationFrame(recorderState.rafId);
    if (recorderState.timerId) clearInterval(recorderState.timerId);
    if (recorderState.stream) recorderState.stream.getTracks().forEach(t => t.stop());
    recorderState.mediaRecorder = null; recorderState.stream = null; recorderState.chunks = [];
    recorderState.takeBlob = null; recorderState.analyser = null; recorderState.rafId = null; recorderState.timerId = null;
    const panel = document.getElementById('recorderPanel');
    panel.dataset.mode = 'idle';
    panel.style.display = 'none';
    document.getElementById('recordBtn').classList.remove('armed');
}
```

- [ ] **Step 4: Wire the capture/preview buttons**

In the final event-binding block (near the existing `voiceInput` listener, ~line 1897), add:

```javascript
document.getElementById('recordBtn').addEventListener('click', () => {
    const panel = document.getElementById('recorderPanel');
    if (panel.dataset.mode === 'recording') { stopRecording(); }
    else { startRecording(); }
});
document.getElementById('recStopBtn').addEventListener('click', stopRecording);
document.getElementById('reRecordBtn').addEventListener('click', () => { resetRecorder(); startRecording(); });
document.getElementById('discardTakeBtn').addEventListener('click', resetRecorder);
```

- [ ] **Step 5: Verify capture + preview in the browser**

Run the Verification Harness, open `http://localhost:8765/subliminal-mixer-808.html`. Then:
1. Click **● Record** → grant mic permission → observe: button turns red (`.armed`), timer counts up, the level meter moves when you speak, LED shows `RECORDING`.
2. Click **■ Stop** (or Record again) → observe: preview `<audio>` appears with the take, LED shows `PREVIEW TAKE`.
3. Press play on the preview `<audio>` → you hear your recording.
4. Click **Discard** → panel hides, returns to idle.
5. Console shows no errors (`read_console_messages` / devtools).

Expected: all five observations hold. If mic permission is blocked, confirm you are on `http://localhost`, not `file://`.

- [ ] **Step 6: Commit**

```bash
git add subliminal-mixer-808.html
git commit -m "feat(mixer): add vocal recorder capture + preview UI"
```

---

### Task 2: Approve take → add to mix, download WAV, and error handling

Wires **Use this take** to decode the take into the mixer's `voiceBuffer` (reusing the existing load path), **Download take** to save WAV via `bufferToWav()`, and hardens the error/edge cases (unsupported browser, empty take).

**Files:**
- Modify: `subliminal-mixer-808.html` — JS recorder module (append functions); event-binding block; a small guard near page init for `MediaRecorder` support.

**Interfaces:**
- Consumes (from Task 1): `recorderState`, `resetRecorder()`, DOM ids `#useTakeBtn`, `#downloadTakeBtn`.
- Consumes (existing engine): the vocal-load path in `loadAudioFile` (~1348-1400) — specifically the lines that set `voiceBuffer = buffer;`, add the `voice-on` LED class, set `#voiceName`, and flip `#ledDisplay` to `READY`. `bufferToWav(buffer)` (~1828). `audioCtx.decodeAudioData`.
- Produces:
  - `async function decodeTakeToBuffer()` → returns an `AudioBuffer` decoded from `recorderState.takeBlob`.
  - `function applyVoiceBuffer(buffer, label)` → sets `voiceBuffer`, updates the same UI `loadAudioFile` updates for the voice case (extracted/shared), so recorder and upload stay DRY.

- [ ] **Step 1: Extract the shared "voice buffer applied" UI update (DRY)**

In `loadAudioFile` (~1367-1382), the `type === 'voice'` branch sets `voiceBuffer`, adds `voice-on`, sets `#voiceName`, and updates `#ledDisplay`. Extract that into a reusable function so the recorder reuses identical behavior. Add near the recorder module:

```javascript
function applyVoiceBuffer(buffer, label) {
    voiceBuffer = buffer;
    document.getElementById('voiceLed').classList.add('voice-on');
    const nameEl = document.getElementById('voiceName');
    nameEl.textContent = `${label} (${buffer.duration.toFixed(1)}s)`;
    nameEl.classList.remove('empty');
    const display = document.getElementById('ledDisplay');
    if (musicBuffer && voiceBuffer) display.textContent = 'READY';
    else display.textContent = 'LOAD 2ND FILE';
    if (isPlaying) { stopAudio(); startAudio(); }
    updateTimeDisplay();
}
```

Then in `loadAudioFile`'s `else` (voice) branch, replace the inline voice-buffer/`voiceName`/led updates with `applyVoiceBuffer(buffer, file.name);` (leave the `music` branch untouched). Verify the `music`/`voice` split still sets `musicBuffer` inline for music.

- [ ] **Step 2: Add decode + approve + download functions**

Append to the recorder module:

```javascript
async function decodeTakeToBuffer() {
    if (!audioCtx) initAudio();
    const arrayBuffer = await recorderState.takeBlob.arrayBuffer();
    return await audioCtx.decodeAudioData(arrayBuffer);
}

let takeCounter = 0;
async function useTake() {
    if (!recorderState.takeBlob) return;
    try {
        const buffer = await decodeTakeToBuffer();
        takeCounter += 1;
        applyVoiceBuffer(buffer, `Recorded take ${takeCounter}`);
        resetRecorder();
    } catch (err) {
        showRecError('Could not process the recording: ' + err.message + '. Try Re-record.');
    }
}

async function downloadTake() {
    if (!recorderState.takeBlob) return;
    try {
        const buffer = await decodeTakeToBuffer();
        const wavBlob = bufferToWav(buffer);
        const url = URL.createObjectURL(wavBlob);
        const a = document.createElement('a');
        a.href = url; a.download = `affirmation-take-${takeCounter + 1}.wav`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (err) {
        showRecError('Could not export WAV: ' + err.message);
    }
}
```

- [ ] **Step 3: Wire approve + download buttons, and guard unsupported browsers**

In the event-binding block add:

```javascript
document.getElementById('useTakeBtn').addEventListener('click', useTake);
document.getElementById('downloadTakeBtn').addEventListener('click', downloadTake);

// Hide recording UI if the browser lacks MediaRecorder/getUserMedia
if (!window.MediaRecorder || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const rb = document.getElementById('recordBtn');
    rb.disabled = true;
    rb.title = 'Recording not supported in this browser';
    rb.textContent = '● Record (unsupported)';
}
```

- [ ] **Step 4: Verify approve, mix, and download in the browser**

Serve via harness, open on localhost. Load a music file via **Load Songs** first (so the mix has two layers). Then:
1. Record → Stop → **Use this take** → observe: `#voiceName` shows `Recorded take 1 (N.Ns)`, voice LED lights, LED display shows `READY`, recorder panel hides.
2. Click **Play** (main mixer) → you hear the music with your recorded vocal layered (as the subliminal voice).
3. Click **Export** → a `subliminal-mix.wav` downloads. Run `ffprobe subliminal-mix.wav` → confirm it is a valid WAV with non-zero duration.
4. Record again → **Download take** → `affirmation-take-2.wav` downloads. Run `ffprobe affirmation-take-2.wav` → valid WAV, duration ≈ your recording length.

Expected: all hold. (Note per spec: a second "Use this take" replaces the single `voiceBuffer` — this is the accepted current-model behavior, not multi-track append.)

- [ ] **Step 5: Verify the error path**

In the browser, deny mic permission (or use the site permission toggle to block the mic), then click **● Record**. Expected: inline `#recError` shows "Microphone access needed to record (NotAllowedError)." and nothing crashes; console has no unhandled exception.

- [ ] **Step 6: Commit**

```bash
git add subliminal-mixer-808.html
git commit -m "feat(mixer): approve take into mix, WAV download, recorder error handling"
```

---

### Task 3: Classical Library data + browse panel

Adds the Baroque track dataset, a **Browse Classical** button, and a collapsible panel that renders each track as a row with a **Get track ↗** link. Data only + rendering; link URLs are finalized in Task 4.

**Files:**
- Modify: `subliminal-mixer-808.html` — markup near `.file-box.music` (~886-894); CSS; JS library module; event wiring.

**Interfaces:**
- Consumes: nothing from the engine (pure UI/data).
- Produces:
  - `const CLASSICAL_LIBRARY` array of `{ title, composer, source, sourceUrl, approxLength, tempo, license }`.
  - `function renderClassicalLibrary()` — populates `#classicalList`.
  - `function toggleClassicalLibrary()` — shows/hides `#classicalSection`.
  - DOM ids: `#browseClassicalBtn`, `#classicalSection`, `#classicalList`.

- [ ] **Step 1: Add the Browse button + panel markup**

In the `.file-box.music` `.file-row` (after `#musicName`, ~line 894), add:

```html
<button class="file-btn" id="browseClassicalBtn">Browse Classical</button>
```

After the whole `.file-section` div (~line 908, before `<!-- Track List (Album Mode) -->`), add:

```html
<div class="classical-section" id="classicalSection" style="display:none">
    <div class="classical-header">
        <span class="classical-title">🎻 Classical Library — Baroque, public domain (~60 BPM)</span>
        <span class="classical-hint">Open a source, download the track, then load it via “Load Songs”.</span>
    </div>
    <div class="classical-list" id="classicalList"></div>
</div>
```

- [ ] **Step 2: Add the CSS**

```css
.classical-section { margin:10px 0; padding:10px; border:1px solid #333; border-radius:6px; background:#141414; }
.classical-header { display:flex; flex-direction:column; gap:2px; margin-bottom:8px; }
.classical-title { font-weight:600; }
.classical-hint { font-size:12px; opacity:.7; }
.classical-list { display:flex; flex-direction:column; gap:4px; }
.classical-row { display:grid; grid-template-columns:2fr 1.4fr auto auto auto; gap:8px; align-items:center; padding:6px 8px; background:#0e0e0e; border-radius:4px; font-size:13px; }
.classical-row .cl-license { font-size:11px; opacity:.7; }
.classical-row a.cl-get { white-space:nowrap; color:#4aa3ff; text-decoration:none; }
.classical-row a.cl-get:hover { text-decoration:underline; }
@media (max-width:640px){ .classical-row{ grid-template-columns:1fr auto; } .classical-row .cl-source,.classical-row .cl-len{ display:none; } }
```

- [ ] **Step 3: Add the dataset + render module**

Append a JS module. Populate with these curated Baroque largo/adagio pieces. `sourceUrl` uses deterministic Musopen search URLs as the starting point; Task 4 verifies/pins each to a confirmed Public-Domain download page.

```javascript
// ===== CLASSICAL LIBRARY (Baroque, public domain, ~60 BPM) =====
const CLASSICAL_LIBRARY = [
    { title:"Air on the G String (Suite No. 3)", composer:"J.S. Bach", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Air%20on%20the%20G%20String", approxLength:"5:00", tempo:"~56 BPM", license:"Public Domain" },
    { title:"Adagio in G minor", composer:"Albinoni / Giazotto", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Albinoni%20Adagio", approxLength:"8:00", tempo:"~52 BPM", license:"Public Domain" },
    { title:"Ombra mai fù (Largo from Xerxes)", composer:"G.F. Handel", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Handel%20Largo%20Xerxes", approxLength:"3:30", tempo:"~54 BPM", license:"Public Domain" },
    { title:"Canon in D", composer:"J. Pachelbel", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Pachelbel%20Canon", approxLength:"5:00", tempo:"~64 BPM", license:"Public Domain" },
    { title:"Oboe Concerto in D minor — Adagio", composer:"A. Marcello", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Marcello%20Oboe%20Concerto", approxLength:"4:00", tempo:"~56 BPM", license:"Public Domain" },
    { title:"Sheep May Safely Graze (BWV 208)", composer:"J.S. Bach", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Sheep%20May%20Safely%20Graze", approxLength:"5:00", tempo:"~60 BPM", license:"Public Domain" },
    { title:"Winter (The Four Seasons) — Largo", composer:"A. Vivaldi", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Vivaldi%20Winter%20Largo", approxLength:"2:00", tempo:"~56 BPM", license:"Public Domain" },
    { title:"Concerto Grosso Op. 6 No. 8 (Christmas) — Largo", composer:"A. Corelli", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Corelli%20Christmas%20Concerto", approxLength:"3:00", tempo:"~58 BPM", license:"Public Domain" },
    { title:"Sarabande in D minor (HWV 437)", composer:"G.F. Handel", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Handel%20Sarabande", approxLength:"3:00", tempo:"~60 BPM", license:"Public Domain" },
    { title:"Concerto for Oboe & Violin (BWV 1060) — Adagio", composer:"J.S. Bach", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Bach%20BWV%201060%20Adagio", approxLength:"4:30", tempo:"~54 BPM", license:"Public Domain" },
    { title:"Viola Concerto in G — Largo", composer:"G.P. Telemann", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Telemann%20Viola%20Concerto", approxLength:"3:00", tempo:"~58 BPM", license:"Public Domain" },
    { title:"Jesu, Joy of Man's Desiring (BWV 147)", composer:"J.S. Bach", source:"Musopen", sourceUrl:"https://musopen.org/music/?q=Jesu%20Joy%20of%20Man%27s%20Desiring", approxLength:"3:00", tempo:"~62 BPM", license:"Public Domain" }
];

function renderClassicalLibrary() {
    const list = document.getElementById('classicalList');
    list.innerHTML = '';
    CLASSICAL_LIBRARY.forEach(t => {
        const row = document.createElement('div');
        row.className = 'classical-row';
        const title = document.createElement('span');
        title.innerHTML = `<strong>${t.title}</strong> · ${t.composer}`;
        const source = document.createElement('span');
        source.className = 'cl-source'; source.textContent = t.source;
        const len = document.createElement('span');
        len.className = 'cl-len'; len.textContent = `${t.approxLength} · ${t.tempo}`;
        const lic = document.createElement('span');
        lic.className = 'cl-license'; lic.textContent = t.license;
        const link = document.createElement('a');
        link.className = 'cl-get'; link.href = t.sourceUrl; link.target = '_blank';
        link.rel = 'noopener noreferrer'; link.textContent = 'Get track ↗';
        row.append(title, source, len, lic, link);
        list.appendChild(row);
    });
}

function toggleClassicalLibrary() {
    const sec = document.getElementById('classicalSection');
    const showing = sec.style.display !== 'none';
    if (showing) { sec.style.display = 'none'; }
    else { if (!document.getElementById('classicalList').children.length) renderClassicalLibrary(); sec.style.display = 'block'; }
}
```

- [ ] **Step 4: Wire the toggle button**

```javascript
document.getElementById('browseClassicalBtn').addEventListener('click', toggleClassicalLibrary);
```

- [ ] **Step 5: Verify the panel renders**

Open the page (harness or even `file://` is fine — no mic here). Click **Browse Classical**. Expected: the section expands showing 12 rows, each with title · composer, source, length · tempo, license, and a **Get track ↗** link. Click the button again → it collapses. Click a **Get track ↗** link → a new tab opens to that source URL.

- [ ] **Step 6: Commit**

```bash
git add subliminal-mixer-808.html
git commit -m "feat(mixer): add Baroque public-domain classical library panel"
```

---

### Task 4: Verify library links and licenses

Confirms each `sourceUrl` resolves to a genuinely Public-Domain / CC0 recording and pins direct download/track pages where possible. This is a required gate before the library ships (spec: "each link is verified … the license is shown inline").

**Files:**
- Modify: `subliminal-mixer-808.html` — `CLASSICAL_LIBRARY` entries only (URLs, and any dropped/replaced tracks).

**Interfaces:**
- Consumes/Produces: the `CLASSICAL_LIBRARY` array from Task 3.

- [ ] **Step 1: Check each source link**

For each of the 12 entries, open `sourceUrl` and confirm: (a) the page loads, (b) it offers a downloadable recording of that piece, (c) the recording's license is Public Domain or CC0 (Musopen labels PD recordings explicitly; Archive.org shows the license on the item page). Prefer replacing the Musopen *search* URL with the specific *piece/recording page* URL when one is confirmed. Use `WebFetch` on each URL to read the page and confirm availability + license.

- [ ] **Step 2: Fix, replace, or drop**

For any entry whose recording is not confirmed PD/CC0 or whose link is dead: pin it to a confirmed alternative (another PD recording of the same or a comparable Baroque ~60 BPM piece), or remove it. Keep the list at ~10–12 confirmed entries. Update `sourceUrl`, `source`, and `license` to match what you verified. Do not leave any entry pointing at an unverified or non-PD recording.

- [ ] **Step 3: Re-verify render**

Reload the page, open **Browse Classical**, click through 2–3 **Get track ↗** links to confirm they now land on the confirmed pages. Confirm every visible `license` label matches the verified license.

- [ ] **Step 4: Commit**

```bash
git add subliminal-mixer-808.html
git commit -m "chore(mixer): verify and pin public-domain classical library sources"
```

---

## Self-Review

**Spec coverage:**
- Recorder placement in `.file-box.voice` → Task 1 Step 1. ✓
- Preview → keep/redo/discard flow → Task 1 (preview) + Task 2 (useTake) + reRecord/discard wiring Task 1 Step 4. ✓
- Live input-level meter → Task 1 (`drawMeter`). ✓
- Approved take into existing vocal path (DRY via `applyVoiceBuffer`) → Task 2 Steps 1-2. ✓
- Download take as WAV via `bufferToWav` → Task 2 Step 2 (`downloadTake`). ✓
- Multi-take = replace fallback, surfaced → Task 2 Step 4 note. ✓
- Recorder error handling (permission, unsupported, decode) → Task 2 Steps 3/5 + `showRecError`. ✓
- Secure-context/localhost requirement → Global Constraints + Verification Harness. ✓
- Classical library: Baroque, ~60 BPM, ~12 entries, browse+download, no audio bundled → Task 3. ✓
- License verification gate → Task 4. ✓
- Additive only / single file / no deps → Global Constraints; all tasks modify only the one file. ✓

**Placeholder scan:** No "TBD/TODO/handle appropriately" — all code is concrete. The only intentionally-deferred content is the exact pinned track URLs, which are a *verification* deliverable (Task 4) with a concrete starting URL per entry, not a placeholder. ✓

**Type consistency:** `recorderState`, `resetRecorder()`, `applyVoiceBuffer(buffer, label)`, `decodeTakeToBuffer()`, `bufferToWav(buffer)`, `CLASSICAL_LIBRARY` fields (`title, composer, source, sourceUrl, approxLength, tempo, license`) are used consistently across tasks and match the render function. ✓

## Notes for the implementer

- The mixer stores the vocal as the **subliminal voice** layer (`voiceBuffer`), mixed at low dB. A recorded affirmation therefore plays as the near-inaudible subliminal track by design — that is correct behavior, not a bug.
- Do not refactor the existing `music` branch of `loadAudioFile`; only the `voice` branch is extracted to `applyVoiceBuffer` for reuse.
- Keep everything in `subliminal-mixer-808.html`. No new files.
