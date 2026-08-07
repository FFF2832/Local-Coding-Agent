#!/usr/bin/env bash
#
# OpenCode offline deployment script (x86 Linux, connects to local Ollama)
# Built for company x86 machines. Shares the same underlying logic as the
# MAXER-5000 (Jetson ARM64) version, with the following differences:
#   - Adds the Ollama installation step itself (assumes a fresh machine)
#   - GPU detection uses nvidia-smi (not tegrastats, which is Jetson-only)
#   - Flash Attention is only recommended when an NVIDIA GPU is detected
#   - No unified memory architecture; VRAM is a resource separate from
#     system RAM and must be considered independently
#   - Base model auto-detection: no need to manually specify which model
#     to use, the script automatically picks the first non-embedding
#     model from 'ollama list'. To pin a specific model, prefix the call:
#     OPENCODE_BASE_MODEL="gemma4:latest" ./opencode-install-x86.sh model
#     (this is not remembered between calls; re-add the prefix every time,
#     or export it once in your shell session)
#
# Usage: ./opencode-install-x86.sh [check|ollama|prereq|system|model|configure|ollama-tune|verify|all]
#   check        Run environment checks only, no changes made
#   ollama       Install Ollama itself (if not already installed)
#   prereq       Install opencode-ai (npm) + approve its postinstall script
#   system       Install system-level dependencies: clipboard tools (xclip/wl-clipboard), python3-venv
#   model        Build a large-context-window variant of the specified base model
#   configure    Write global opencode.jsonc + auth.json (points to local Ollama) + TUI settings
#   ollama-tune  Tune Ollama's systemd settings: context ceiling, Flash Attention (GPU-dependent), KV cache quantization
#   verify       Run the full verification pass and print a summary table
#   all          Run everything above in order (default)
#
# Design principles:
#   - Every step is idempotent: re-running won't reinstall or overwrite existing setup
#   - Destructive operations (overwriting existing config content, system service
#     settings) are never auto-applied; the script only prints the suggested command
#   - Every step prints a clear [OK] / [FAIL] / [SKIP] / [ACTION NEEDED] marker
#   - Does not use `go install sst/opencode`; the npm package `opencode-ai` and
#     charmbracelet's older Go-based tool are two different projects, this
#     script only handles the former
#   - Config files live at the "global" path (~/.config/opencode/), so opencode
#     works the same regardless of which folder it's started from

set -uo pipefail

BASE_MODEL="${OPENCODE_BASE_MODEL:-}"
NUM_CTX="${OPENCODE_NUM_CTX:-32768}"
OLLAMA_URL="http://localhost:11434"
OLLAMA_OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OLLAMA_OVERRIDE_FILE="$OLLAMA_OVERRIDE_DIR/override.conf"
GLOBAL_CONFIG_FILE="$HOME/.config/opencode/opencode.jsonc"
TUI_CONFIG_FILE="$HOME/.config/opencode/tui.json"
AUTH_CONFIG_FILE="$HOME/.local/share/opencode/auth.json"

# Known name prefixes that aren't suitable as a coding-agent chat model
# (embedding models, etc.) — excluded during auto-detection.
EXCLUDE_MODEL_PATTERNS="nomic-embed|mxbai-embed|all-minilm|bge-"

# Auto-detect the base model: prefer the value from the OPENCODE_BASE_MODEL
# env var if set; otherwise pick the first model from `ollama list` that
# isn't on the exclusion list. This avoids having to type
# OPENCODE_BASE_MODEL= every time, while still allowing manual override
# (an explicit value always takes priority).
detect_base_model() {
  if [ -n "$BASE_MODEL" ]; then
    echo "$BASE_MODEL"
    return 0
  fi
  local detected
  detected="$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -Ev "$EXCLUDE_MODEL_PATTERNS" | grep -v -- '-32k$' | head -1)"
  echo "$detected"
}

log_ok()     { echo "[OK] $1"; }
log_fail()   { echo "[FAIL] $1"; }
log_skip()   { echo "[SKIP] $1"; }
log_action() { echo "[ACTION NEEDED] $1"; }
log_info()   { echo "[INFO] $1"; }

# ── GPU detection ─────────────────────────────────────────

has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

check_gpu() {
  if has_nvidia_gpu; then
    local gpu_info
    gpu_info="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
    log_ok "NVIDIA GPU detected: $gpu_info"
    return 0
  elif lspci 2>/dev/null | grep -qi nvidia; then
    log_fail "lspci sees an NVIDIA device, but nvidia-smi failed to run -- the driver may not be installed, so CUDA acceleration will not be active"
    return 1
  else
    log_info "No NVIDIA GPU detected, inference will run in CPU mode (significantly slower; prefer smaller models)"
    return 1
  fi
}

# ── Environment checks ────────────────────────────────────

check_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    log_ok "Node.js $(node -v) / npm $(npm -v)"
    return 0
  else
    log_fail "Node.js or npm is not installed (opencode-ai is installed via npm). Install via apt: sudo apt install -y nodejs npm, or use nvm for a newer version"
    return 1
  fi
}

check_ollama_installed() {
  if command -v ollama >/dev/null 2>&1; then
    log_ok "Ollama is installed ($(ollama --version 2>/dev/null | head -1))"
    return 0
  else
    log_fail "Ollama is not installed yet"
    return 1
  fi
}

check_ollama_service() {
  if curl -s "$OLLAMA_URL" >/dev/null 2>&1; then
    log_ok "Ollama service is running ($OLLAMA_URL)"
    return 0
  else
    log_fail "Ollama service is not responding, run 'sudo systemctl start ollama' or 'ollama serve'"
    return 1
  fi
}

check_opencode() {
  if command -v opencode >/dev/null 2>&1; then
    log_ok "OpenCode $(opencode --version 2>/dev/null || echo '(failed to read version)')"
    return 0
  else
    log_fail "opencode command not found (common cause: the npm postinstall script was not approved)"
    return 1
  fi
}

check_context_model() {
  if ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -qx "$CONTEXT_MODEL"; then
    log_ok "Local model $CONTEXT_MODEL already exists"
    return 0
  else
    log_fail "Local model $CONTEXT_MODEL does not exist (the large-context variant of base model $BASE_MODEL has not been built yet)"
    return 1
  fi
}

check_clipboard() {
  if command -v xclip >/dev/null 2>&1 || command -v xsel >/dev/null 2>&1 || command -v wl-copy >/dev/null 2>&1; then
    log_ok "Clipboard tool is installed"
    return 0
  else
    log_fail "No clipboard tool found (none of xclip/xsel/wl-clipboard are installed), OpenCode TUI copy/paste will not work"
    return 1
  fi
}

check_python_venv() {
  if python3 -m venv --help >/dev/null 2>&1; then
    log_ok "python3-venv is available"
    return 0
  else
    log_fail "python3-venv is not installed; the agent running 'python3 -m venv' will fail and require sudo to fix"
    return 1
  fi
}

check_context_length_env() {
  if [ -f "$OLLAMA_OVERRIDE_FILE" ] && grep -q "OLLAMA_CONTEXT_LENGTH=$NUM_CTX" "$OLLAMA_OVERRIDE_FILE" 2>/dev/null; then
    log_ok "OLLAMA_CONTEXT_LENGTH global ceiling = $NUM_CTX (matches the individual model's num_ctx)"
    return 0
  else
    log_fail "OLLAMA_CONTEXT_LENGTH is not set or does not match $NUM_CTX -- this is a global ceiling that overrides individual models' num_ctx, and is the most common cause of long conversations breaking down or getting truncated"
    return 1
  fi
}

check_global_config() {
  if [ ! -f "$GLOBAL_CONFIG_FILE" ]; then
    log_fail "$GLOBAL_CONFIG_FILE does not exist"
    return 1
  fi
  if grep -q '"@ai-sdk/openai-compatible"' "$GLOBAL_CONFIG_FILE" 2>/dev/null; then
    log_ok "Global config provider key format is correct"
  else
    log_fail "Global config exists, but the provider does not include '@ai-sdk/openai-compatible'"
    return 1
  fi
  if grep -q "\"$CONTEXT_MODEL\"" "$GLOBAL_CONFIG_FILE" 2>/dev/null; then
    log_ok "Global config already includes model: $CONTEXT_MODEL"
  else
    log_fail "model: $CONTEXT_MODEL not found in the global config"
    return 1
  fi
  return 0
}

check_auth_json() {
  if [ -f "$AUTH_CONFIG_FILE" ] && grep -q '"ollama"' "$AUTH_CONFIG_FILE" 2>/dev/null; then
    log_ok "auth.json already has the ollama placeholder key set"
    return 0
  else
    log_fail "$AUTH_CONFIG_FILE is not set up; some versions will prompt for a manual API key entry as a result"
    return 1
  fi
}

run_check() {
  echo "===== Environment Check ====="
  local gpu_ok=0 node_ok=0 ollama_bin_ok=0 ollama_ok=0 opencode_ok=0 model_ok=0 config_ok=0 \
        clip_ok=0 venv_ok=0 ctxlen_ok=0 auth_ok=0
  check_gpu               && gpu_ok=1
  check_node               && node_ok=1
  check_ollama_installed   && ollama_bin_ok=1
  if [ "$ollama_bin_ok" -eq 1 ]; then
    check_ollama_service  && ollama_ok=1
  else
    log_skip "Ollama service check (not installed yet)"
  fi
  check_opencode           && opencode_ok=1
  if [ "$ollama_ok" -eq 1 ]; then
    check_context_model   && model_ok=1
  else
    log_skip "Local model check (Ollama service not ready)"
  fi
  check_global_config      && config_ok=1
  check_clipboard          && clip_ok=1
  check_python_venv        && venv_ok=1
  check_context_length_env && ctxlen_ok=1
  check_auth_json          && auth_ok=1

  echo "===== Summary ====="
  if [ "$node_ok" -eq 1 ] && [ "$ollama_bin_ok" -eq 1 ] && [ "$ollama_ok" -eq 1 ] && \
     [ "$opencode_ok" -eq 1 ] && [ "$model_ok" -eq 1 ] && [ "$config_ok" -eq 1 ] && \
     [ "$clip_ok" -eq 1 ] && [ "$venv_ok" -eq 1 ] && [ "$ctxlen_ok" -eq 1 ] && [ "$auth_ok" -eq 1 ]; then
    log_ok "Environment is fully ready. Run opencode from any folder to use it"
    [ "$gpu_ok" -eq 0 ] && log_info "Currently running in CPU inference mode, significantly slower than with a GPU -- prefer smaller models"
    return 0
  fi
  log_info "Not ready yet. Run the stages in order: ollama -> prereq -> system -> model -> configure -> ollama-tune"
  return 1
}

# ── Install Ollama itself ─────────────────────────────────

install_ollama() {
  echo "===== Installing Ollama ====="
  if check_ollama_installed; then
    log_skip "Ollama is already installed"
  else
    log_info "Running the official install script..."
    if curl -fsSL https://ollama.com/install.sh | sh >/tmp/ollama_install.log 2>&1; then
      log_ok "Ollama installation complete"
    else
      log_fail "Installation failed, see /tmp/ollama_install.log (if your company network has a proxy restriction, additional setup may be needed)"
      return 1
    fi
  fi

  if check_ollama_service; then
    log_skip "Ollama service is already running"
  else
    log_info "Starting the Ollama service..."
    sudo systemctl enable --now ollama >/tmp/ollama_start.log 2>&1 || true
    sleep 2
    if check_ollama_service; then
      log_ok "Ollama service started successfully"
    else
      log_fail "Service failed to start, see /tmp/ollama_start.log, or run 'ollama serve' manually to check the error output"
      return 1
    fi
  fi

  check_gpu
}

# ── OpenCode installation ─────────────────────────────────

install_prereq() {
  echo "===== Installing OpenCode (opencode-ai) ====="

  if ! check_node; then
    log_fail "Install Node.js first, then re-run './opencode-install-x86.sh prereq'"
    return 1
  fi

  if check_opencode; then
    log_skip "opencode-ai is already installed"
  else
    log_info "Installing opencode-ai globally via npm..."
    if npm i -g opencode-ai@latest >/tmp/opencode_install.log 2>&1; then
      log_ok "npm package installation complete"
    else
      log_fail "npm install failed, see /tmp/opencode_install.log"
      return 1
    fi

    log_info "Approving the postinstall script..."
    npm approve-scripts --allow-scripts-pending >/tmp/opencode_approve.log 2>&1 || true

    local npm_bin
    npm_bin="$(npm prefix -g)/bin"
    if ! echo "$PATH" | grep -q "$npm_bin"; then
      if ! grep -q "$npm_bin" "$HOME/.bashrc" 2>/dev/null; then
        echo "export PATH=\"$npm_bin:\$PATH\"" >> "$HOME/.bashrc"
      fi
      export PATH="$npm_bin:$PATH"
    fi

    if check_opencode; then
      log_ok "opencode command is callable"
    else
      log_fail "opencode command still not found, see /tmp/opencode_install.log and /tmp/opencode_approve.log"
      return 1
    fi
  fi
}

# ── System-level dependencies ─────────────────────────────

install_system_deps() {
  echo "===== System dependencies: clipboard tools + python3-venv ====="

  if check_clipboard; then
    log_skip "Clipboard tool is already installed"
  else
    log_info "Installing clipboard tools..."
    if sudo apt install -y xclip wl-clipboard >/tmp/opencode_clipboard.log 2>&1; then
      log_ok "xclip / wl-clipboard installation complete"
    else
      log_fail "Installation failed, see /tmp/opencode_clipboard.log"
    fi
  fi

  if check_python_venv; then
    log_skip "python3-venv is already available"
  else
    log_info "Installing python3-venv..."
    local py_ver
    py_ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3")"
    if sudo apt install -y "python${py_ver}-venv" >/tmp/opencode_venv.log 2>&1 || sudo apt install -y python3-venv >/tmp/opencode_venv.log 2>&1; then
      log_ok "python3-venv installation complete"
    else
      log_fail "Installation failed, see /tmp/opencode_venv.log"
    fi
  fi
}

# ── Local model: build the large-context-window variant ───

setup_model() {
  echo "===== Building local model $CONTEXT_MODEL ====="

  if ! check_ollama_service; then
    log_fail "Ollama service is not ready, run './opencode-install-x86.sh ollama' first"
    return 1
  fi

  if ! ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -qx "$BASE_MODEL"; then
    log_info "Base model $BASE_MODEL has not been downloaded yet, running 'ollama pull $BASE_MODEL'..."
    if ! ollama pull "$BASE_MODEL" 2>&1 | tee /tmp/opencode_pull.log; then
      log_fail "Download failed, see /tmp/opencode_pull.log (check disk space: df -h /)"
      return 1
    fi
  fi

  if check_context_model; then
    log_skip "$CONTEXT_MODEL already exists, skipping creation"
    return 0
  fi

  log_info "Building a num_ctx=$NUM_CTX variant based on $BASE_MODEL (via a Modelfile, to avoid interactive-mode input timing issues)..."
  local modelfile
  modelfile="$(mktemp)"
  cat > "$modelfile" <<EOF
FROM $BASE_MODEL
PARAMETER num_ctx $NUM_CTX
EOF
  if ollama create "$CONTEXT_MODEL" -f "$modelfile" > /tmp/opencode_model_setup.log 2>&1; then
    rm -f "$modelfile"
    if check_context_model; then
      log_ok "$CONTEXT_MODEL created successfully"
    else
      log_fail "Command completed but $CONTEXT_MODEL does not appear in 'ollama list', see /tmp/opencode_model_setup.log"
      return 1
    fi
  else
    rm -f "$modelfile"
    log_fail "Creation failed, see /tmp/opencode_model_setup.log"
    return 1
  fi
}

# ── Configure OpenCode (global config) ────────────────────

configure_opencode() {
  echo "===== Configuring OpenCode (global) ====="

  if ! check_opencode; then
    log_fail "OpenCode is not ready, run './opencode-install-x86.sh prereq' first"
    return 1
  fi
  if ! check_context_model; then
    log_fail "Local model is not ready, run './opencode-install-x86.sh model' first"
    return 1
  fi

  mkdir -p "$HOME/.config/opencode"

  local should_write_config=1
  if [ -f "$GLOBAL_CONFIG_FILE" ]; then
    # Determine whether this is a "shell" config file (schema only, no real
    # provider content). A shell config is safe to overwrite; one with real
    # provider content is protected and left untouched.
    if grep -q '"provider"' "$GLOBAL_CONFIG_FILE" 2>/dev/null; then
      log_action "$GLOBAL_CONFIG_FILE already exists and contains provider settings, the script will not overwrite it (to avoid destroying any other model settings you added manually). To regenerate it, back up or delete the file manually first, then re-run './opencode-install-x86.sh configure'"
      should_write_config=0
    else
      log_info "$GLOBAL_CONFIG_FILE exists but is a shell config (no provider settings), safe to overwrite"
    fi
  fi

  if [ "$should_write_config" -eq 1 ]; then
    cat > "$GLOBAL_CONFIG_FILE" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "ollama/$CONTEXT_MODEL",
  "compaction": {
    "auto": true,
    "prune": true,
    "threshold": 0.8
  },
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "${OLLAMA_URL}/v1",
        "apiKey": "ollama-local"
      },
      "models": {
        "$CONTEXT_MODEL": {
          "name": "$CONTEXT_MODEL",
          "reasoning": true,
          "tools": true
        }
      }
    }
  }
}
EOF
    log_ok "$GLOBAL_CONFIG_FILE created (applies globally, includes auto-compaction settings; threshold support varies by version)"
  fi

  mkdir -p "$(dirname "$AUTH_CONFIG_FILE")"
  if [ -f "$AUTH_CONFIG_FILE" ] && grep -q '"ollama"' "$AUTH_CONFIG_FILE" 2>/dev/null; then
    log_skip "$AUTH_CONFIG_FILE already contains ollama settings"
  else
    cat > "$AUTH_CONFIG_FILE" <<'EOF'
{
  "ollama": {
    "type": "api",
    "key": "ollama"
  }
}
EOF
    log_ok "$AUTH_CONFIG_FILE created"
  fi

  if [ -f "$TUI_CONFIG_FILE" ]; then
    log_skip "$TUI_CONFIG_FILE already exists, not overwritten"
  else
    cat > "$TUI_CONFIG_FILE" <<'EOF'
{
  "mouse": true
}
EOF
    log_ok "$TUI_CONFIG_FILE created (mouse:true; copy/paste uses Shift+drag, requires xclip/wl-clipboard)"
  fi

  check_global_config
}

# ── Ollama service tuning ─────────────────────────────────

tune_ollama() {
  echo "===== Tuning Ollama systemd settings ====="

  sudo mkdir -p "$OLLAMA_OVERRIDE_DIR" 2>/dev/null || true

  local gpu_present=0
  has_nvidia_gpu && gpu_present=1

  if [ ! -f "$OLLAMA_OVERRIDE_FILE" ]; then
    log_info "$OLLAMA_OVERRIDE_FILE does not exist, creating it (requires sudo, may prompt for a password)..."
    local flash_line=""
    if [ "$gpu_present" -eq 1 ]; then
      flash_line='Environment="OLLAMA_FLASH_ATTENTION=1"'
    fi
    if sudo tee "$OLLAMA_OVERRIDE_FILE" > /tmp/opencode_tune.log 2>&1 <<CONF
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_CONTEXT_LENGTH=$NUM_CTX"
$flash_line
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_KEEP_ALIVE=-1"
CONF
    then
      log_ok "$OLLAMA_OVERRIDE_FILE created"
      sudo systemctl daemon-reload
      sudo systemctl restart ollama
      sleep 2
      if check_ollama_service; then
        log_ok "Ollama service restarted successfully with the new settings applied"
      else
        log_fail "Service did not respond after restart, check 'journalctl -u ollama -n 30 --no-pager' (a GPU-related setting may be incompatible with this hardware -- try removing the OLLAMA_FLASH_ATTENTION line first)"
        return 1
      fi
    else
      log_fail "Write failed, see /tmp/opencode_tune.log"
      return 1
    fi
    return 0
  fi

  log_info "Current contents of $OLLAMA_OVERRIDE_FILE:"
  cat "$OLLAMA_OVERRIDE_FILE"

  if check_context_length_env; then
    log_skip "OLLAMA_CONTEXT_LENGTH is already $NUM_CTX"
  else
    log_action "OLLAMA_CONTEXT_LENGTH needs to be manually changed to $NUM_CTX. Edit $OLLAMA_OVERRIDE_FILE, then run:
    sudo systemctl daemon-reload && sudo systemctl restart ollama"
  fi

  if [ "$gpu_present" -eq 1 ]; then
    if grep -q "OLLAMA_FLASH_ATTENTION=1" "$OLLAMA_OVERRIDE_FILE" 2>/dev/null; then
      log_ok "OLLAMA_FLASH_ATTENTION is enabled"
    else
      log_action "NVIDIA GPU detected, recommend adding Environment=\"OLLAMA_FLASH_ATTENTION=1\" (well-supported and worth enabling on x86+CUDA)"
    fi
    if grep -q "OLLAMA_KV_CACHE_TYPE=q8_0" "$OLLAMA_OVERRIDE_FILE" 2>/dev/null; then
      log_ok "OLLAMA_KV_CACHE_TYPE is quantized to q8_0"
    else
      log_action "Recommend adding Environment=\"OLLAMA_KV_CACHE_TYPE=q8_0\" (saves VRAM)"
    fi
  else
    log_info "No NVIDIA GPU detected, Flash Attention is not recommended (ineffective or potentially problematic in CPU mode)"
  fi

  if grep -q "OLLAMA_KEEP_ALIVE=-1" "$OLLAMA_OVERRIDE_FILE" 2>/dev/null; then
    log_ok "OLLAMA_KEEP_ALIVE is set to stay resident permanently"
  else
    log_action "Recommend adding Environment=\"OLLAMA_KEEP_ALIVE=-1\" (keeps the model resident in memory, avoiding reload after every idle period)"
  fi
}

# ── Verification ───────────────────────────────────────────

run_verify() {
  echo "===== Full Verification ====="
  run_check
  local check_result=$?

  echo "===== Next Steps (cannot be automated, needs manual confirmation) ====="
  log_info "1. Run opencode from any folder and confirm the bottom-left shows 'Ollama (local)', not the default 'OpenCode Zen'"
  log_info "2. Recommended: disconnect from the network and send a simple prompt, to confirm it still responds without internet access"
  log_info "3. Verify tool-calling stability with a simple prompt first (e.g. \"use bash to create test.txt, execute directly without a text explanation\")"
  log_info "4. For large files, avoid asking for a full file rewrite -- instead ask to read the relevant section first, then use targeted line-based edits"
  if has_nvidia_gpu; then
    log_info "5. Monitor the GPU with 'nvidia-smi -l 1' in real time, to confirm GPU usage actually rises during inference rather than silently falling back to CPU mode"
  else
    log_info "5. No GPU present, inference runs on CPU and will be noticeably slower than with a discrete GPU -- prefer models in the 3B-8B range"
  fi
  return $check_result
}

# ── Main flow ──────────────────────────────────────────────

main() {
  local cmd="${1:-all}"

  # Except for the 'ollama' sub-command, which doesn't depend on any
  # specific model, every other command auto-detects the base model first.
  if [ "$cmd" != "ollama" ]; then
    BASE_MODEL="$(detect_base_model)"
    if [ -z "$BASE_MODEL" ]; then
      if [ "$cmd" = "check" ]; then
        log_info "No usable base model detected (ollama list is empty, or only contains embedding models), model-related checks will show [FAIL]"
        CONTEXT_MODEL="(no base model detected)"
      else
        log_fail "No usable base model detected (ollama list is empty, or only contains embedding models). Run 'ollama pull <model>' manually first, or re-run with OPENCODE_BASE_MODEL=<model> specified"
        exit 1
      fi
    else
      CONTEXT_MODEL="${BASE_MODEL%%:*}:${BASE_MODEL##*:}-32k"
      log_info "Using base model: $BASE_MODEL (auto-detected as the first non-embedding model in ollama list when not manually specified; to pin a specific model, prefix with OPENCODE_BASE_MODEL=<model>)"
    fi
  fi

  case "$cmd" in
    check)       run_check ;;
    ollama)      install_ollama ;;
    prereq)      install_prereq ;;
    system)      install_system_deps ;;
    model)       setup_model ;;
    configure)   configure_opencode ;;
    ollama-tune) tune_ollama ;;
    verify)      run_verify ;;
    all)
      install_ollama && install_prereq && install_system_deps && setup_model && configure_opencode
      tune_ollama
      run_verify
      ;;
    *)
      echo "Usage: $0 [check|ollama|prereq|system|model|configure|ollama-tune|verify|all]"
      exit 2
      ;;
  esac
}

main "$@"
