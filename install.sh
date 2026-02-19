#!/usr/bin/env bash
#
# Hypnosis Audio Builder - Mac Installer
#
# Usage:
#   ./install.sh          # Install everything
#   ./install.sh --check  # Check prerequisites only
#
set -euo pipefail

APP_NAME="Hypnosis Audio Builder"
PYTHON_MIN="3.9"
VENV_DIR="venv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()    { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
success() { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn()    { printf "\033[1;33m==>\033[0m %s\n" "$*"; }
fail()    { printf "\033[1;31m==>\033[0m %s\n" "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
check_python() {
    info "Checking Python..."

    # Try python3 first, fall back to python
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        fail "Python not found. Install Python $PYTHON_MIN+ from https://www.python.org/downloads/"
    fi

    # Verify version
    PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
        fail "Python $PY_VERSION found, but $PYTHON_MIN+ is required"
    fi

    success "Python $PY_VERSION found ($PYTHON_CMD)"
}

check_ffmpeg() {
    info "Checking ffmpeg..."

    if command -v ffmpeg &>/dev/null; then
        FF_VERSION=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
        success "ffmpeg $FF_VERSION found"
        return 0
    fi

    warn "ffmpeg not found (required for MP3 support)"
    return 1
}

install_ffmpeg() {
    if command -v ffmpeg &>/dev/null; then
        return 0
    fi

    info "Installing ffmpeg..."

    if command -v brew &>/dev/null; then
        brew install ffmpeg
        success "ffmpeg installed via Homebrew"
    else
        warn "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for Apple Silicon Macs
        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi

        brew install ffmpeg
        success "Homebrew and ffmpeg installed"
    fi
}

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
setup_venv() {
    info "Setting up Python virtual environment..."

    if [ -d "$VENV_DIR" ]; then
        warn "Virtual environment already exists. Reusing it."
    else
        $PYTHON_CMD -m venv "$VENV_DIR"
        success "Virtual environment created in ./$VENV_DIR/"
    fi

    # Activate
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    success "Virtual environment activated"
}

install_package() {
    info "Installing $APP_NAME and dependencies..."

    # Upgrade pip first
    pip install --upgrade pip --quiet

    # Install the package in editable mode so changes take effect immediately
    pip install -e ".[dev]" --quiet

    success "$APP_NAME installed successfully"
}

verify_install() {
    info "Verifying installation..."

    # Check the CLI command works
    if hypnosis --version &>/dev/null; then
        VERSION=$(hypnosis --version 2>&1)
        success "CLI command 'hypnosis' works ($VERSION)"
    else
        # Fall back to module invocation
        if python -m hypnosis_audio_builder --version &>/dev/null; then
            success "Module invocation works (python -m hypnosis_audio_builder)"
        else
            warn "CLI verification had issues, but installation completed"
        fi
    fi

    # Run the built-in test
    info "Running built-in audio test..."
    if hypnosis --test &>/dev/null; then
        success "Audio generation test passed"
    else
        warn "Audio test had issues. Check that ffmpeg is installed for MP3 support."
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║     Hypnosis Audio Builder v1.1.0    ║"
    echo "  ║          Mac Installer                ║"
    echo "  ╚══════════════════════════════════════╝"
    echo ""

    # Navigate to the script's directory (repo root)
    cd "$(dirname "$0")"

    # Check-only mode
    if [ "${1:-}" = "--check" ]; then
        check_python
        check_ffmpeg || true
        echo ""
        info "Check complete. Run ./install.sh to install."
        exit 0
    fi

    # Step 1: Python
    check_python

    # Step 2: ffmpeg
    if ! check_ffmpeg; then
        read -rp "   Install ffmpeg via Homebrew? [Y/n] " REPLY
        case "${REPLY:-Y}" in
            [Nn]*) warn "Skipping ffmpeg. MP3 export will not work without it." ;;
            *)     install_ffmpeg ;;
        esac
    fi

    # Step 3: Virtual environment
    setup_venv

    # Step 4: Install package
    install_package

    # Step 5: Verify
    verify_install

    # Done
    echo ""
    echo "  ──────────────────────────────────────"
    echo ""
    success "Installation complete!"
    echo ""
    echo "  To get started:"
    echo ""
    echo "    source venv/bin/activate"
    echo "    hypnosis --test"
    echo "    hypnosis --help"
    echo ""
    echo "  Quick example:"
    echo ""
    echo "    hypnosis --voice recording.wav --subliminal-from-voice -o output.mp3"
    echo ""
    echo "  ──────────────────────────────────────"
    echo ""
}

main "$@"
