# Installing training-hub for OpenCode

## Prerequisites

- [OpenCode.ai](https://opencode.ai) installed
- Python 3.11+ with uv or pip

## Installation

1. Add training-hub to the `plugin` array in your `opencode.json` (global or project-level):

   ```json
   {
     "plugin": ["training-hub@git+https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git"]
   }
   ```

2. Restart OpenCode. The plugin auto-installs and registers all skills.

3. Install the Python library (needed for training):
   ```bash
   uv pip install training-hub
   ```
   For GPU support:
   ```bash
   uv pip install "training-hub[cuda]"
   ```

Verify by asking: "What LLM training skills do you have?"

## Usage

Use OpenCode's native `skill` tool:

```
use skill tool to list skills
use skill tool to load training-hub/training-guide
```

Or just describe what you want: "Fine-tune Llama 3 on my dataset using LoRA."

## Updating

Restart OpenCode to pull the latest version.

To pin a specific version:

```json
{
  "plugin": ["training-hub@git+https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git#v1.0.0"]
}
```

## Tool Mapping

When skills reference Claude Code tools:
- `Bash(...)` → your native shell execution tool
- `Read`, `Write`, `Edit` → your native file tools
- `/th-setup`, `/th-train`, `/th-estimate` commands → invoke the matching skill directly

## Troubleshooting

### Plugin not loading

1. Check logs: `opencode run --print-logs "hello" 2>&1 | grep -i training-hub`
2. Verify the plugin line in your `opencode.json`

### Skills not found

1. Use `skill` tool to list what's discovered
2. Check that the plugin is loading (see above)

## Uninstalling

Remove the `training-hub` entry from the `plugin` array in `opencode.json` and restart OpenCode.
