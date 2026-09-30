#!/usr/bin/env sh
# Start OR Lens from a read-only Codex plugin package.
# Protocol output belongs exclusively to the Python MCP server; this script uses stderr.
set -eu

fail() {
  printf '%s\n' "OR Lens: $*" >&2
  exit 1
}

plugin_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
case "$(uname -s)" in
  Darwin|Linux) ;;
  *) fail "only macOS and Linux are supported by this local bootstrap" ;;
esac
case "$(uname -m)" in
  arm64|aarch64|x86_64) ;;
  *) fail "unsupported CPU architecture: $(uname -m)" ;;
esac

cache_base=${XDG_CACHE_HOME:-"${HOME:-}/.cache"}
[ -n "$cache_base" ] || fail "HOME or XDG_CACHE_HOME is required for the local runtime"
runtime_base=${OR_LENS_RUNTIME_DIR:-"$cache_base/or-lens"}
mkdir -p "$runtime_base"

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1"
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    fail "shasum or sha256sum is required to prepare the local runtime"
  fi
}
sha256_stream() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  else
    fail "shasum or sha256sum is required to prepare the local runtime"
  fi
}

fingerprint=$(
  (
    cd "$plugin_root"
    printf '%s\n' .codex-plugin/plugin.json mcp.json pyproject.toml uv.lock README.md
    find server scripts skills -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print
  ) | LC_ALL=C sort | while IFS= read -r path; do
    sha256_file "$plugin_root/$path"
  done | sha256_stream | awk '{print $1}'
)
[ -n "$fingerprint" ] || fail "could not fingerprint the packaged application"
runtime_root="$runtime_base/$fingerprint"
runtime_source="$runtime_root/source"
runtime_venv="$runtime_source/.venv"
ready_file="$runtime_root/ready"
lock_dir="$runtime_root/bootstrap.lock"

acquire_lock() {
  attempts=0
  while ! mkdir "$lock_dir" 2>/dev/null; do
    if [ -r "$lock_dir/pid" ]; then
      lock_pid=$(cat "$lock_dir/pid" 2>/dev/null || true)
      case "$lock_pid" in
        *[!0-9]*|'') ;;
        *)
          if ! kill -0 "$lock_pid" 2>/dev/null; then
            printf '%s\n' "OR Lens: recovering a stale bootstrap lock from process $lock_pid." >&2
            rm -f "$lock_dir/pid"
            rmdir "$lock_dir" 2>/dev/null || true
            continue
          fi
          ;;
      esac
    fi
    attempts=$((attempts + 1))
    [ "$attempts" -lt 300 ] || fail "another OR Lens setup did not finish within five minutes; remove $lock_dir only after confirming it is not running"
    sleep 1
  done
  printf '%s\n' "$$" > "$lock_dir/pid"
}
release_lock() {
  rm -f "$lock_dir/pid"
  rmdir "$lock_dir" 2>/dev/null || true
}

if [ ! -x "$runtime_venv/bin/or-lens" ] || [ ! -f "$ready_file" ]; then
  mkdir -p "$runtime_root"
  acquire_lock
  trap 'release_lock; exit 130' INT TERM
  trap release_lock EXIT
  if [ ! -x "$runtime_venv/bin/or-lens" ] || [ ! -f "$ready_file" ]; then
    mkdir -p "$runtime_root"
    rm -rf "$runtime_source"
    rm -f "$ready_file"
    mkdir -p "$runtime_source"
    if find "$plugin_root/server" "$plugin_root/scripts" "$plugin_root/skills" -type l -print -quit | grep -q .; then
      fail "the plugin package contains a symbolic link in an allowlisted directory"
    fi
    source_archive="$runtime_root/source.tar.$$"
    (
      cd "$plugin_root"
      tar -cf "$source_archive" .codex-plugin mcp.json pyproject.toml uv.lock README.md scripts server skills
    ) || fail "could not stage the read-only plugin package"
    (
      cd "$runtime_source"
      tar -xf "$source_archive"
    ) || fail "could not unpack the local runtime"
    rm -f "$source_archive"

    if [ -n "${UV:-}" ]; then
      uv_bin=$(command -v "$UV" || true)
      [ -n "$uv_bin" ] || fail "UV does not name an executable: $UV"
    else
      uv_bin=$(command -v uv || true)
    fi
    if [ -z "$uv_bin" ]; then
      command -v curl >/dev/null 2>&1 || fail "uv is missing and curl is unavailable; install uv from https://docs.astral.sh/uv/getting-started/installation/"
      installer="$runtime_root/uv-installer.sh"
      printf '%s\n' 'OR Lens: downloading the official uv installer for first use.' >&2
      curl --fail --location --proto '=https' --tlsv1.2 \
        https://astral.sh/uv/install.sh --output "$installer" \
        || fail "could not download uv; install it from https://docs.astral.sh/uv/getting-started/installation/"
      UV_UNMANAGED_INSTALL="$runtime_root/bin" sh "$installer" >&2 \
        || fail "the official uv installer failed"
      rm -f "$installer"
      uv_bin="$runtime_root/bin/uv"
    fi
    [ -x "$uv_bin" ] || fail "uv executable is unavailable after installation"

    printf '%s\n' 'OR Lens: creating its local, lockfile-pinned Python runtime.' >&2
    (
      cd "$runtime_source"
      UV_CACHE_DIR="$runtime_root/uv-cache" \
      UV_PYTHON_INSTALL_DIR="$runtime_root/python" \
      "$uv_bin" sync --frozen --no-dev --python 3.12 >&2
    ) || fail "could not create the local runtime; verify internet access and retry"
    [ -x "$runtime_venv/bin/or-lens" ] || fail "uv finished without installing the OR Lens command"
    : > "$ready_file"
  fi
  release_lock
  trap - EXIT INT TERM
fi

exec "$runtime_venv/bin/or-lens" desktop
