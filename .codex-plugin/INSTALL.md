# Installing training-hub for Codex

Enable training-hub LLM training skills in Codex via native skill discovery.

## Prerequisites

- Git
- Python 3.11+ with uv or pip
- [Codex CLI](https://github.com/openai/codex) installed

## Installation

1. **Clone the training-hub repository:**
   ```bash
   git clone https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git ~/.codex/training-hub
   ```

2. **Install the Python library:**
   ```bash
   cd ~/.codex/training-hub && uv sync --extra dev
   ```
   Or with pip:
   ```bash
   pip install -e ~/.codex/training-hub
   ```
   For GPU support:
   ```bash
   pip install -e "~/.codex/training-hub[cuda]"
   ```

3. **Create the skills symlink:**
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/training-hub/skills ~/.agents/skills/training-hub
   ```

   **Windows (PowerShell):**
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills"
   cmd /c mklink /J "$env:USERPROFILE\.agents\skills\training-hub" "$env:USERPROFILE\.codex\training-hub\skills"
   ```

4. **Restart Codex** to discover the skills.

## Path Resolution

When skills reference `${CLAUDE_PLUGIN_ROOT}/scripts/...`, use the clone path instead:
```bash
~/.codex/training-hub/scripts/th_detect.sh
~/.codex/training-hub/scripts/th_train.sh
~/.codex/training-hub/scripts/th_estimate.sh
```

## Verify

```bash
ls -la ~/.agents/skills/training-hub
```

You should see a symlink pointing to the training-hub skills directory.

## Updating

```bash
cd ~/.codex/training-hub && git pull
```

Skills update instantly through the symlink.

## Uninstalling

```bash
rm ~/.agents/skills/training-hub
rm -rf ~/.codex/training-hub
```
