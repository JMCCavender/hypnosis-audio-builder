#!/usr/bin/env python3
"""
fetch_music.py - Download license-free classical music for the Hypnosis Audio Builder.

Reads music/classical-library.json and downloads a public-domain recording for
each catalogued piece from the Internet Archive's Musopen collection, saving the
files into music/ so they can be used as ambient/song tracks (Load Songs in the
mixer, or --album / --ambient-music on the CLI).

All catalogued pieces are public-domain COMPOSITIONS. The recordings resolved
here come from Musopen's public-domain release hosted on the Internet Archive
(https://archive.org/details/musopen). Always confirm the specific recording's
license note on its source page before commercial use.

Usage:
    python fetch_music.py                 # download the whole catalog into music/
    python fetch_music.py --list          # just list the catalog, download nothing
    python fetch_music.py --limit 3       # download only the first 3 pieces
    python fetch_music.py --out music/    # choose the output directory

Requires only the Python standard library. Run it in an environment with
outbound internet access to archive.org.
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

CATALOG_PATH = Path(__file__).parent / "music" / "classical-library.json"

ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"
ARCHIVE_METADATA = "https://archive.org/metadata"
ARCHIVE_DOWNLOAD = "https://archive.org/download"

# Preferred audio formats in descending order (archive.org "format" values)
AUDIO_FORMAT_PRIORITY = [
    "VBR MP3",
    "128Kbps MP3",
    "64Kbps MP3",
    "MP3",
    "Ogg Vorbis",
    "Flac",
    "24bit Flac",
]

USER_AGENT = "hypnosis-audio-builder/fetch_music (+https://github.com/JMCCavender/hypnosis-audio-builder)"


def slugify(text: str, max_len: int = 50) -> str:
    """Turn a title/composer into a filesystem-safe slug."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len] or "track"


def pick_audio_file(files: list) -> Optional[dict]:
    """
    Choose the best audio file from an archive.org metadata 'files' list.

    Args:
        files: list of file dicts from the archive.org metadata API.

    Returns:
        The chosen file dict, or None if no audio file is present.
    """
    audio = [f for f in files if f.get("format") in AUDIO_FORMAT_PRIORITY
             or str(f.get("name", "")).lower().endswith((".mp3", ".ogg", ".flac"))]
    if not audio:
        return None

    def rank(f: dict) -> int:
        fmt = f.get("format", "")
        return AUDIO_FORMAT_PRIORITY.index(fmt) if fmt in AUDIO_FORMAT_PRIORITY else len(AUDIO_FORMAT_PRIORITY)

    return sorted(audio, key=rank)[0]


def build_download_url(identifier: str, filename: str) -> str:
    """Build the archive.org direct-download URL for a file within an item."""
    return f"{ARCHIVE_DOWNLOAD}/{identifier}/{urllib.parse.quote(filename)}"


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def resolve_identifier(archive_query: str, timeout: int = 30) -> Optional[str]:
    """Find the best-matching Internet Archive item identifier for a query."""
    params = urllib.parse.urlencode({
        "q": archive_query,
        "fl[]": "identifier",
        "rows": "5",
        "sort[]": "downloads desc",
        "output": "json",
    }, doseq=True)
    try:
        data = _get_json(f"{ARCHIVE_SEARCH}?{params}", timeout=timeout)
    except Exception as e:  # noqa: BLE001 - network best-effort
        print(f"    ! search failed: {e}")
        return None
    docs = data.get("response", {}).get("docs", [])
    return docs[0].get("identifier") if docs else None


def resolve_file_url(identifier: str, timeout: int = 30) -> Optional[str]:
    """Resolve a direct audio download URL for an archive.org item."""
    try:
        meta = _get_json(f"{ARCHIVE_METADATA}/{identifier}", timeout=timeout)
    except Exception as e:  # noqa: BLE001
        print(f"    ! metadata failed: {e}")
        return None
    chosen = pick_audio_file(meta.get("files", []))
    if not chosen:
        return None
    return build_download_url(identifier, chosen["name"])


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Stream a URL to a local file. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    ! download failed: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def load_catalog(path: Path = CATALOG_PATH) -> list:
    """Load the track list from the JSON catalog."""
    data = json.loads(path.read_text())
    return data.get("tracks", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Download license-free classical music into music/")
    parser.add_argument("--list", action="store_true", help="List the catalog and exit")
    parser.add_argument("--out", type=Path, default=CATALOG_PATH.parent, help="Output directory (default: music/)")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N pieces")
    parser.add_argument("--ext", default="mp3", help="Preferred extension for saved files (informational)")
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        print(f"Catalog not found: {CATALOG_PATH}")
        return 1

    tracks = load_catalog()
    if args.limit:
        tracks = tracks[:args.limit]

    if args.list:
        print(f"Classical Music Library ({len(tracks)} pieces):\n")
        for i, t in enumerate(tracks, 1):
            moods = ", ".join(t.get("moods", []))
            print(f"  {i:02d}. {t['composer']} - {t['title']}  [{moods}]")
            print(f"      Source: {t.get('source', {}).get('page', 'n/a')}")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(tracks)} pieces into {out_dir}/\n")
    ok = 0
    for i, t in enumerate(tracks, 1):
        label = f"{t['composer']} - {t['title']}"
        print(f"[{i}/{len(tracks)}] {label}")

        url = (t.get("source") or {}).get("download")
        if not url and t.get("archive_query"):
            identifier = resolve_identifier(t["archive_query"])
            if identifier:
                print(f"    archive item: {identifier}")
                url = resolve_file_url(identifier)

        if not url:
            page = (t.get("source") or {}).get("page", "n/a")
            print(f"    ! could not resolve a downloadable recording automatically.")
            print(f"      Download manually from: {page}")
            continue

        ext = Path(urllib.parse.urlparse(url).path).suffix or f".{args.ext}"
        dest = out_dir / f"{i:02d}_{slugify(t['composer'])}_{slugify(t['title'])}{ext}"
        print(f"    -> {dest.name}")
        if download_file(url, dest):
            ok += 1
            print(f"    done ({dest.stat().st_size // 1024} KB)")

    print(f"\nDownloaded {ok}/{len(tracks)} pieces into {out_dir}/")
    if ok < len(tracks):
        print("For any that failed, use the SOURCE links (musopen.org / archive.org) to download manually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
