#!/usr/bin/env bash
set -Eeuo pipefail

IMPLEMENTATION="pytorch"
DISPLAY_NAME="PyTorch"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ $# -ne 1 ]]; then echo "Usage: $0 <problem_name>"; exit 2; fi
problem="$1"
[[ "$problem" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Error: unsafe problem name: $problem" >&2; exit 2; }
[[ -f "$ENV_FILE" ]] || { echo "Error: missing $ENV_FILE" >&2; exit 2; }
set -a; source "$ENV_FILE"; set +a
for variable in LOCAL_PROJECT_PATH REMOTE_OS REMOTE_HOST REMOTE_USER REMOTE_ROOT REMOTE_PYTHON; do
    [[ -n "${!variable:-}" ]] || { echo "Error: set $variable in scripts/.env" >&2; exit 2; }
done
REMOTE_OS="${REMOTE_OS,,}"
[[ "$REMOTE_OS" == "windows" || "$REMOTE_OS" == "linux" ]] || { echo "Error: REMOTE_OS must be windows or linux" >&2; exit 2; }
if [[ "$LOCAL_PROJECT_PATH" = /* ]]; then PROJECT_DIR="$LOCAL_PROJECT_PATH"; else PROJECT_DIR="$(cd -- "$SCRIPT_DIR/$LOCAL_PROJECT_PATH" && pwd)"; fi
source_file="$PROJECT_DIR/src/$IMPLEMENTATION/$problem.py"
test_file="$PROJECT_DIR/test/$IMPLEMENTATION/$problem.py"
[[ -f "$source_file" && -f "$test_file" ]] || { echo "Error: source or test file not found for $problem" >&2; exit 2; }
for command_name in ssh scp tee; do command -v "$command_name" >/dev/null 2>&1 || { echo "Error: missing command: $command_name" >&2; exit 127; }; done

COMMON_OPTIONS=(-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new)
SSH_OPTIONS=(-p "${REMOTE_PORT:-22}" "${COMMON_OPTIONS[@]}")
SCP_OPTIONS=(-P "${REMOTE_PORT:-22}" "${COMMON_OPTIONS[@]}")
if [[ -n "${SSH_KEY_PATH:-}" ]]; then SSH_OPTIONS+=(-i "$SSH_KEY_PATH"); SCP_OPTIONS+=(-i "$SSH_KEY_PATH"); fi
if [[ -n "${REMOTE_PASSWORD:-}" ]]; then
    command -v sshpass >/dev/null 2>&1 || { echo "Error: password authentication requires sshpass" >&2; exit 127; }
    export SSHPASS="$REMOTE_PASSWORD"; SSH=(sshpass -e ssh); SCP=(sshpass -e scp)
else
    SSH=(ssh); SCP=(scp)
fi

REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
timestamp="$(date +%Y%m%d_%H%M%S)"; execution_time="$(date '+%Y-%m-%d %H:%M:%S %Z')"; run_id="${timestamp}_$$"
remote_dir="${REMOTE_ROOT}/${problem}_${run_id}"; remote_dir_win="${remote_dir//\//\\}"
result_dir="$PROJECT_DIR/result/$IMPLEMENTATION"; result_file="$result_dir/${problem}.txt"; summary_file="$result_dir/history.md"
mkdir -p "$result_dir"
if [[ ! -s "$summary_file" ]]; then printf '# PyTorch Run History\n\n| Execution Time | Problem | Platform | Status | Average | Maximum | Minimum |\n| --- | --- | --- | :---: | ---: | ---: | ---: |\n' > "$summary_file"; fi

cleanup_remote() {
    if [[ "$REMOTE_OS" == "windows" ]]; then
        "${SSH[@]}" "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "powershell.exe -NoProfile -NonInteractive -Command \"Remove-Item -LiteralPath '$remote_dir' -Recurse -Force\""
    else
        "${SSH[@]}" "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "rm -rf -- '$remote_dir'"
    fi
}

run_remote() {
    echo "[1/4] Creating remote directories..."
    if [[ "$REMOTE_OS" == "windows" ]]; then
        "${SSH[@]}" "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "powershell.exe -NoProfile -NonInteractive -Command \"[void](New-Item -ItemType Directory -Force -Path '$remote_dir/src/$IMPLEMENTATION','$remote_dir/test/$IMPLEMENTATION')\"" || return 1
    else
        "${SSH[@]}" "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "mkdir -p '$remote_dir/src/$IMPLEMENTATION' '$remote_dir/test/$IMPLEMENTATION'" || return 1
    fi
    echo "[2/4] Uploading source and test files..."
    "${SCP[@]}" "${SCP_OPTIONS[@]}" "$source_file" "$REMOTE_TARGET:$remote_dir/src/$IMPLEMENTATION/$problem.py" || return 1
    "${SCP[@]}" "${SCP_OPTIONS[@]}" "$test_file" "$REMOTE_TARGET:$remote_dir/test/$IMPLEMENTATION/$problem.py" || return 1
    echo "[3/4] Running..."
    if [[ "$REMOTE_OS" == "windows" ]]; then
        "${SSH[@]}" "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "cd /d \"$remote_dir_win\" && echo === GPU === && nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader && set PYTHONPATH=$remote_dir_win && \"$REMOTE_PYTHON\" \"test\\$IMPLEMENTATION\\$problem.py\"" || return 1
    else
        "${SSH[@]}" "${SSH_OPTIONS[@]}" "$REMOTE_TARGET" "cd '$remote_dir' && echo '=== GPU ===' && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader && PYTHONPATH='$remote_dir' \"$REMOTE_PYTHON\" 'test/$IMPLEMENTATION/$problem.py'" || return 1
    fi
}

{ echo "============================================================"; echo "$DISPLAY_NAME problem: $problem"; echo "============================================================"; } | tee "$result_file"
status="PASS"; if ! run_remote 2>&1 | tee -a "$result_file"; then status="FAIL"; fi
if [[ "${KEEP_REMOTE:-0}" == "1" ]]; then echo "[4/4] Remote files retained at: $remote_dir" | tee -a "$result_file"; else echo "[4/4] Cleaning remote files..." | tee -a "$result_file"; cleanup_remote 2>&1 | tee -a "$result_file" || true; fi
echo "Result: $status - $problem" | tee -a "$result_file"
perf_line="$(tr -d '\r' < "$result_file" | awk '/^\[PERF\]/{line=$0} END{print line}')"
average="N/A"; maximum="N/A"; minimum="N/A"
platform="$(tr -d '\r' < "$result_file" | awk '/^=== GPU ===/{getline; sub(/,.*/, ""); print; exit}')"
[[ -n "$platform" ]] || platform="N/A"
if [[ -n "$perf_line" ]]; then
    [[ "$perf_line" =~ avg=([0-9]+([.][0-9]+)?)[[:space:]]ms ]] && average="${BASH_REMATCH[1]} ms"
    [[ "$perf_line" =~ max=([0-9]+([.][0-9]+)?)[[:space:]]ms ]] && maximum="${BASH_REMATCH[1]} ms"
    [[ "$perf_line" =~ min=([0-9]+([.][0-9]+)?)[[:space:]]ms ]] && minimum="${BASH_REMATCH[1]} ms"
fi
printf '| %s | `%s` | %s | **%s** | %s | %s | %s |\n' "$execution_time" "$problem" "$platform" "$status" "$average" "$maximum" "$minimum" >> "$summary_file"
[[ "$status" == "PASS" ]]
