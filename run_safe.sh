#!/usr/bin/env bash
# Enter the project, activate its venv, and preserve the terminal mode.

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir" || exit 2

venv_path="${CS437_VENV:-${HOME}/env}"
if ! test -f "$venv_path/bin/activate"; then
  echo "Virtual environment not found: $venv_path" >&2
  echo "Set CS437_VENV if it is stored somewhere else." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$venv_path/bin/activate"

if ! test -t 0; then
  exec python "$@"
fi

tty_state=$(stty -g)
restore_terminal() {
  stty "$tty_state" 2>/dev/null || stty sane 2>/dev/null || true
}
trap restore_terminal EXIT HUP INT TERM

python "$@"
status=$?
restore_terminal
exit "$status"
