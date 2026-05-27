# Installing training-hub for Codex

## Via Marketplace (Recommended)

```bash
codex plugin marketplace add Red-Hat-AI-Innovation-Team/plugins
```

Then install the plugin from the marketplace. The Python library will need to be installed separately:

```bash
pip install training-hub
```

For GPU training, also install extras (sequentially):
```bash
pip install training-hub[lora]                          # LoRA support
pip install training-hub[grpo,lora]                     # GRPO support
pip install training-hub[cuda] --no-build-isolation     # CUDA kernels
```

## Manual Installation

If you prefer to install manually:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git ~/.codex/training-hub
   ```

2. **Install the Python library:**
   ```bash
   pip install -e ~/.codex/training-hub
   ```

3. **Create the skills symlink** (skills are in `.claude/skills/` — shared between Claude Code and Codex):
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/training-hub/.claude/skills ~/.agents/skills/training-hub
   ```

4. **Restart Codex** to discover the skills.

## Updating

Marketplace installs update automatically. For manual installs:
```bash
cd ~/.codex/training-hub && git pull
```

## Uninstalling

For manual installs:
```bash
rm ~/.agents/skills/training-hub
rm -rf ~/.codex/training-hub
```
