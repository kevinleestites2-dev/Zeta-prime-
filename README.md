# ZetaPrime

**AI Developer Agent — Master coder. Writes, debugs, optimizes. Maintains the Pantheon.**

Model: `qwen2.5-coder:7b` via Ollama `/api/generate`

## What It Does

ZetaPrime is a complete AI-powered developer agent that runs locally. It can write code from prompts, debug errors from tracebacks, optimize performance, explain code line by line, refactor messy code, generate unit tests, convert between languages, search GitHub and Stack Overflow, manage Git, install packages, run linters, generate project boilerplate, and monitor file changes with auto-testing.

## Architecture

| Module | Role |
|--------|------|
| **CodeIntelligence** | LLM-powered code operations: write, debug, optimize, explain, refactor, comment, test, convert, boilerplate |
| **ShellTool** | Safe bash execution with dangerous command blocking |
| **FileTool** | Read, write, search, list files with language detection |
| **GitTool** | Full Git operations: status, commit, push, pull, branch, log, blame, diff, clone, GitHub search |
| **PackageManager** | Install via pip, npm, or pkg (Termux) |
| **Linter** | Run pylint, eslint, ruff, flake8, tsc |
| **TestRunner** | Run tests, detect frameworks, watch for file changes |
| **Memory** | SQLite-backed session history, code context, learned debug patterns, test results |
| **TelegramGateway** | Remote control via Telegram bot |

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally with `qwen2.5-coder:7b` pulled
- `requests` library

## Quick Start

```bash
# 1. Install Ollama and pull the model
ollama pull qwen2.5-coder:7b

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (interactive CLI mode)
python zeta_prime.py

# 4. Or run with Telegram (set env vars first)
export TELEGRAM_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
python zeta_prime.py
```

## Commands

| Command | Description |
|---------|-------------|
| `/write <description>` | Write code from a prompt (use `--lang X` for non-Python) |
| `/read <file>` | Read and display a file |
| `/edit <file> <instructions>` | Edit a file with AI guidance |
| `/debug <file_or_traceback>` | Debug errors — runs the file or analyzes a pasted traceback |
| `/optimize <file>` | Optimize code for performance |
| `/explain <file>` | Explain code line by line |
| `/refactor <file>` | Refactor messy code into clean code |
| `/comment <file>` | Add comments and docstrings |
| `/test <file>` | Generate unit tests |
| `/run <command>` | Run any shell command (bash, python, node) |
| `/runtests [path]` | Run test suite (auto-detects framework) |
| `/convert <file> <from> <to>` | Convert code between languages |
| `/lint <file>` | Run linters (pylint, eslint, ruff) |
| `/search <query>` | Search GitHub for examples |
| `/stackoverflow <query>` | Search Stack Overflow |
| `/git <subcmd> [args]` | Git: status, commit, push, pull, branch, log, blame, diff, clone |
| `/install <pip\|npm\|pkg> <packages>` | Install packages |
| `/boilerplate <type> <name>` | Generate project boilerplate |
| `/watch <dir> [test_cmd]` | Monitor file changes, auto-run tests |
| `/files [dir]` | List files in a directory |
| `/find <dir> <pattern>` | Search for pattern in files |
| `/history [n]` | Show session history |
| `/status` | System status |
| `/help` | Show all commands |
| `/stop` | Stop ZetaPrime |

You can also type naturally — ZetaPrime will chat using the LLM.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | LLM model for code operations |
| `ZETA_DB` | `zeta_prime.db` | SQLite database path |
| `ZETA_WORKSPACE` | `./workspace` | Workspace directory for generated projects |
| `TELEGRAM_TOKEN` | _(empty)_ | Telegram bot token |
| `TELEGRAM_CHAT_ID` | _(empty)_ | Telegram chat ID |

## Modes

- **Interactive CLI**: Default when no Telegram credentials are set. Type commands and chat directly.
- **Telegram Bot**: Automatically activates when `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` are set.

## Security

- Dangerous shell commands are blocked (e.g., `rm -rf /`, `dd if=`, fork bombs)
- All shell execution has configurable timeouts
- No cloud dependencies — runs entirely on your local machine

## License

MIT
