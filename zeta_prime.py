#!/usr/bin/env python3
"""
ZetaPrime — AI Developer Agent
Master coder. Writes, debugs, optimizes. Maintains the Pantheon.
Model: qwen2.5-coder:7b via Ollama /api/generate
"""

import os
import re
import sys
import json
import time
import shutil
import sqlite3
import logging
import hashlib
import textwrap
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    raise ImportError("requests is required: pip install requests")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_FILE = "zeta_prime.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ZetaPrime")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
DB_PATH = os.getenv("ZETA_DB", "zeta_prime.db")
WORKSPACE = Path(os.getenv("ZETA_WORKSPACE", "./workspace"))
WORKSPACE.mkdir(exist_ok=True)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ─────────────────────────────────────────────
# MEMORY — SQLite-backed persistent memory
# ─────────────────────────────────────────────

class Memory:
    """Persistent memory for code context, session history, and learned patterns."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS code_context (
                    file_path TEXT PRIMARY KEY,
                    content TEXT,
                    language TEXT,
                    last_modified TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id TEXT PRIMARY KEY,
                    category TEXT,
                    pattern TEXT,
                    solution TEXT,
                    success_count INTEGER DEFAULT 0,
                    timestamp TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    test_command TEXT,
                    passed INTEGER,
                    output TEXT,
                    timestamp TEXT
                )
            """)

    def add_history(self, role: str, content: str):
        with self._conn() as c:
            c.execute(
                "INSERT INTO session_history (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, datetime.utcnow().isoformat())
            )

    def get_history(self, limit: int = 20) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, timestamp FROM session_history ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]

    def store_file(self, file_path: str, content: str, language: str):
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO code_context (file_path, content, language, last_modified)
                VALUES (?, ?, ?, ?)
            """, (file_path, content, language, datetime.utcnow().isoformat()))

    def get_file(self, file_path: str) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT content, language, last_modified FROM code_context WHERE file_path=?",
                (file_path,)
            ).fetchone()
        if row:
            return {"content": row[0], "language": row[1], "last_modified": row[2]}
        return None

    def store_pattern(self, category: str, pattern: str, solution: str):
        pid = hashlib.md5(f"{category}:{pattern}".encode()).hexdigest()[:12]
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO learned_patterns (id, category, pattern, solution, success_count, timestamp)
                VALUES (?, ?, ?, ?, COALESCE((SELECT success_count FROM learned_patterns WHERE id=?), 0) + 1, ?)
            """, (pid, category, pattern, solution, pid, datetime.utcnow().isoformat()))

    def find_pattern(self, category: str, keyword: str) -> Optional[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT pattern, solution, success_count FROM learned_patterns WHERE category=? ORDER BY success_count DESC",
                (category,)
            ).fetchall()
        for r in rows:
            if keyword.lower() in r[0].lower():
                return {"pattern": r[0], "solution": r[1], "success_count": r[2]}
        return None

    def store_test_result(self, file_path: str, test_command: str, passed: bool, output: str):
        with self._conn() as c:
            c.execute(
                "INSERT INTO test_results (file_path, test_command, passed, output, timestamp) VALUES (?, ?, ?, ?, ?)",
                (file_path, test_command, int(passed), output[:4096], datetime.utcnow().isoformat())
            )


# ─────────────────────────────────────────────
# LLM — Local Ollama interface
# ─────────────────────────────────────────────

class LLM:
    """Interface to local Ollama model (qwen2.5-coder:7b)."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            log.error(f"[LLM] Generation failed: {e}")
            return f"Error: {e}"

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────
# TOOLS — Shell, File I/O, Git, Package Management, Linting
# ─────────────────────────────────────────────

class ShellTool:
    """Safe shell command execution."""

    BLOCKED_PATTERNS = [
        r"rm\s+-rf\s+/",
        r"rm\s+--no-preserve-root",
        r"mkfs",
        r"dd\s+if=",
        r">\s*/dev/sd",
        r":(){ :|:& };:",
    ]

    @staticmethod
    def run(cmd: str, timeout: int = 120, cwd: str = None) -> Dict:
        for pattern in ShellTool.BLOCKED_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return {"status": "blocked", "output": f"Blocked dangerous command matching: {pattern}"}
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=cwd
            )
            return {
                "status": "ok",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "output": (result.stdout + result.stderr).strip(),
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "output": "Command timed out"}
        except Exception as e:
            return {"status": "error", "output": str(e)}


class FileTool:
    """File read/write/search operations."""

    @staticmethod
    def read(path: str) -> Dict:
        try:
            content = Path(path).read_text()
            lang = FileTool.detect_language(path)
            return {"status": "ok", "content": content, "language": lang, "lines": len(content.splitlines())}
        except Exception as e:
            return {"status": "error", "content": str(e)}

    @staticmethod
    def write(path: str, content: str) -> Dict:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content)
            return {"status": "ok", "path": path, "lines": len(content.splitlines())}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def search(directory: str, pattern: str, file_glob: str = "*") -> Dict:
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include", file_glob, pattern, directory],
                capture_output=True, text=True, timeout=30
            )
            matches = result.stdout.strip().splitlines()[:50]
            return {"status": "ok", "matches": matches, "count": len(matches)}
        except Exception as e:
            return {"status": "error", "matches": [], "message": str(e)}

    @staticmethod
    def list_files(directory: str, recursive: bool = True) -> List[str]:
        try:
            p = Path(directory)
            if recursive:
                return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
            return [str(f.relative_to(p)) for f in p.iterdir() if f.is_file()]
        except Exception:
            return []

    @staticmethod
    def detect_language(path: str) -> str:
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "jsx", ".tsx": "tsx", ".java": "java", ".c": "c",
            ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".rs": "rust",
            ".go": "go", ".rb": "ruby", ".php": "php", ".swift": "swift",
            ".kt": "kotlin", ".scala": "scala", ".sh": "bash",
            ".html": "html", ".css": "css", ".json": "json",
            ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
            ".md": "markdown", ".sql": "sql", ".r": "r",
        }
        ext = Path(path).suffix.lower()
        return ext_map.get(ext, "text")


class GitTool:
    """Git operations: commit, push, pull, branch, diff, log, blame."""

    @staticmethod
    def run_git(args: str, cwd: str = None) -> Dict:
        return ShellTool.run(f"git {args}", cwd=cwd)

    @staticmethod
    def status(cwd: str = None) -> Dict:
        return GitTool.run_git("status --short", cwd=cwd)

    @staticmethod
    def diff(cwd: str = None, staged: bool = False) -> Dict:
        flag = "--cached" if staged else ""
        return GitTool.run_git(f"diff {flag}", cwd=cwd)

    @staticmethod
    def log(cwd: str = None, limit: int = 10) -> Dict:
        return GitTool.run_git(f"log --oneline -n {limit}", cwd=cwd)

    @staticmethod
    def blame(file_path: str, cwd: str = None) -> Dict:
        return GitTool.run_git(f"blame {file_path}", cwd=cwd)

    @staticmethod
    def commit(message: str, cwd: str = None) -> Dict:
        GitTool.run_git("add -A", cwd=cwd)
        return GitTool.run_git(f'commit -m "{message}"', cwd=cwd)

    @staticmethod
    def push(branch: str = None, cwd: str = None) -> Dict:
        cmd = f"push origin {branch}" if branch else "push"
        return GitTool.run_git(cmd, cwd=cwd)

    @staticmethod
    def pull(cwd: str = None) -> Dict:
        return GitTool.run_git("pull", cwd=cwd)

    @staticmethod
    def branch(name: str = None, cwd: str = None) -> Dict:
        if name:
            return GitTool.run_git(f"checkout -b {name}", cwd=cwd)
        return GitTool.run_git("branch -a", cwd=cwd)

    @staticmethod
    def clone(url: str, dest: str = None) -> Dict:
        cmd = f"clone {url}"
        if dest:
            cmd += f" {dest}"
        return GitTool.run_git(cmd)

    @staticmethod
    def search_github(query: str) -> Dict:
        try:
            url = f"https://api.github.com/search/repositories?q={query}&sort=stars&per_page=5"
            resp = requests.get(url, timeout=15, headers={"User-Agent": "ZetaPrime/1.0"})
            items = resp.json().get("items", [])
            results = [
                {"name": i["full_name"], "description": i.get("description", ""),
                 "stars": i["stargazers_count"], "url": i["html_url"], "language": i.get("language")}
                for i in items
            ]
            return {"status": "ok", "results": results}
        except Exception as e:
            return {"status": "error", "results": [], "message": str(e)}


class PackageManager:
    """Install packages via pip, npm, or pkg (Termux)."""

    @staticmethod
    def pip_install(packages: List[str]) -> Dict:
        return ShellTool.run(f"pip install {' '.join(packages)}")

    @staticmethod
    def npm_install(packages: List[str] = None, cwd: str = None) -> Dict:
        if packages:
            return ShellTool.run(f"npm install {' '.join(packages)}", cwd=cwd)
        return ShellTool.run("npm install", cwd=cwd)

    @staticmethod
    def pkg_install(packages: List[str]) -> Dict:
        return ShellTool.run(f"pkg install -y {' '.join(packages)}")

    @staticmethod
    def detect_manager(cwd: str = ".") -> str:
        p = Path(cwd)
        if (p / "package.json").exists():
            return "npm"
        if (p / "requirements.txt").exists() or (p / "pyproject.toml").exists():
            return "pip"
        if (p / "Cargo.toml").exists():
            return "cargo"
        if (p / "go.mod").exists():
            return "go"
        return "unknown"


class Linter:
    """Run linters: pylint, eslint, ruff, flake8."""

    LINTERS = {
        "python": [
            ("ruff check", "ruff"),
            ("python -m pylint --output-format=text", "pylint"),
            ("python -m flake8", "flake8"),
        ],
        "javascript": [
            ("npx eslint", "eslint"),
        ],
        "typescript": [
            ("npx eslint", "eslint"),
            ("npx tsc --noEmit", "tsc"),
        ],
    }

    @staticmethod
    def lint(file_path: str, language: str = None) -> Dict:
        lang = language or FileTool.detect_language(file_path)
        linters = Linter.LINTERS.get(lang, [])
        if not linters:
            return {"status": "skip", "message": f"No linter configured for {lang}"}

        results = []
        for cmd_template, name in linters:
            result = ShellTool.run(f"{cmd_template} {file_path}", timeout=60)
            results.append({
                "linter": name,
                "returncode": result.get("returncode", -1),
                "output": result.get("output", "")[:2048],
            })
            if result.get("returncode") == 0:
                break

        return {"status": "ok", "results": results}


# ─────────────────────────────────────────────
# CODE INTELLIGENCE — Explain, Refactor, Comment, Convert, Test
# ─────────────────────────────────────────────

class CodeIntelligence:
    """LLM-powered code operations."""

    def __init__(self, llm: LLM, memory: Memory):
        self.llm = llm
        self.memory = memory

    def write_code(self, prompt: str, language: str = "python") -> str:
        system = (
            f"You are ZetaPrime, an expert {language} developer. "
            f"Write clean, production-quality {language} code based on the user's description. "
            "Include proper error handling, type hints (if applicable), and docstrings. "
            "Output ONLY the code, no explanations or markdown fences."
        )
        return self.llm.generate(prompt, system=system)

    def debug_error(self, code: str, error: str, language: str = "python") -> str:
        system = (
            f"You are ZetaPrime, an expert debugger for {language}. "
            "Analyze the error traceback, identify the root cause, and provide the fixed code. "
            "Explain the bug briefly, then output the corrected code."
        )
        prompt = f"Code:\n```{language}\n{code}\n```\n\nError:\n```\n{error}\n```\n\nFix this bug."
        response = self.llm.generate(prompt, system=system)
        self.memory.store_pattern("debug", error[:200], response[:500])
        return response

    def optimize(self, code: str, language: str = "python") -> str:
        system = (
            f"You are ZetaPrime, a performance optimization expert for {language}. "
            "Analyze the code for performance issues and provide an optimized version. "
            "Explain what you optimized and why, then output the improved code."
        )
        prompt = f"Optimize this code:\n```{language}\n{code}\n```"
        return self.llm.generate(prompt, system=system)

    def explain(self, code: str, language: str = "python") -> str:
        system = (
            f"You are ZetaPrime, a code educator. "
            f"Explain this {language} code line by line in clear, beginner-friendly language. "
            "Use numbered lines and explain what each section does."
        )
        prompt = f"Explain this code line by line:\n```{language}\n{code}\n```"
        return self.llm.generate(prompt, system=system)

    def refactor(self, code: str, language: str = "python") -> str:
        system = (
            f"You are ZetaPrime, a code quality expert for {language}. "
            "Refactor the messy code into clean, well-structured, maintainable code. "
            "Apply SOLID principles, extract functions where needed, improve naming, "
            "and add proper error handling. Output ONLY the refactored code."
        )
        prompt = f"Refactor this code:\n```{language}\n{code}\n```"
        return self.llm.generate(prompt, system=system)

    def add_comments(self, code: str, language: str = "python") -> str:
        system = (
            f"You are ZetaPrime, a documentation expert for {language}. "
            "Add clear, concise comments and docstrings to this code. "
            "Comment complex logic, function purposes, and parameter descriptions. "
            "Output ONLY the commented code."
        )
        prompt = f"Add comments and docstrings:\n```{language}\n{code}\n```"
        return self.llm.generate(prompt, system=system)

    def generate_tests(self, code: str, language: str = "python") -> str:
        framework_map = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "rust": "built-in #[test]",
            "go": "testing package",
            "java": "JUnit",
        }
        framework = framework_map.get(language, "appropriate testing framework")
        system = (
            f"You are ZetaPrime, a testing expert for {language}. "
            f"Generate comprehensive unit tests using {framework}. "
            "Cover edge cases, error conditions, and normal behavior. "
            "Output ONLY the test code, ready to run."
        )
        prompt = f"Generate unit tests for:\n```{language}\n{code}\n```"
        return self.llm.generate(prompt, system=system)

    def convert_language(self, code: str, source_lang: str, target_lang: str) -> str:
        system = (
            f"You are ZetaPrime, a polyglot programmer. "
            f"Convert this {source_lang} code to idiomatic {target_lang}. "
            f"Use {target_lang} conventions, libraries, and best practices. "
            "Output ONLY the converted code."
        )
        prompt = f"Convert from {source_lang} to {target_lang}:\n```{source_lang}\n{code}\n```"
        return self.llm.generate(prompt, system=system)

    def generate_boilerplate(self, project_type: str, name: str) -> Dict[str, str]:
        system = (
            "You are ZetaPrime, a project scaffolding expert. "
            f"Generate the boilerplate files for a {project_type} project named '{name}'. "
            "Output a JSON object where keys are file paths and values are file contents. "
            "Include README.md, main entry point, config files, and .gitignore. "
            "Output ONLY valid JSON, no markdown."
        )
        prompt = f"Generate boilerplate for: {project_type} project called '{name}'"
        response = self.llm.generate(prompt, system=system, temperature=0.2)
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"main.py": "# ZetaPrime boilerplate\nprint('Hello from " + name + "')\n"}

    def search_stackoverflow(self, query: str) -> Dict:
        try:
            url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                "order": "desc", "sort": "relevance",
                "q": query, "site": "stackoverflow",
                "pagesize": 5, "filter": "withbody",
            }
            resp = requests.get(url, params=params, timeout=15)
            items = resp.json().get("items", [])
            results = [
                {"title": i["title"], "link": i["link"],
                 "score": i["score"], "answered": i.get("is_answered", False)}
                for i in items
            ]
            return {"status": "ok", "results": results}
        except Exception as e:
            return {"status": "error", "results": [], "message": str(e)}


# ─────────────────────────────────────────────
# TEST RUNNER — Auto-run tests, monitor
# ─────────────────────────────────────────────

class TestRunner:
    """Run and monitor tests."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def run_tests(self, path: str, command: str = None, cwd: str = None) -> Dict:
        if not command:
            command = self._detect_test_command(path)
        result = ShellTool.run(command, timeout=120, cwd=cwd)
        passed = result.get("returncode", 1) == 0
        self.memory.store_test_result(path, command, passed, result.get("output", ""))
        return {
            "status": "ok",
            "passed": passed,
            "command": command,
            "output": result.get("output", "")[:4096],
        }

    def _detect_test_command(self, path: str) -> str:
        p = Path(path)
        if p.is_file():
            lang = FileTool.detect_language(path)
            if lang == "python":
                return f"python -m pytest {path} -v"
            elif lang in ("javascript", "typescript"):
                return f"npx jest {path}"
        if p.is_dir():
            if (p / "package.json").exists():
                return "npm test"
            if (p / "pytest.ini").exists() or (p / "pyproject.toml").exists():
                return "python -m pytest -v"
            if (p / "Cargo.toml").exists():
                return "cargo test"
        return f"python -m pytest {path} -v"

    def watch(self, directory: str, test_command: str = None, interval: int = 5):
        log.info(f"[TestRunner] Watching {directory} for changes (interval={interval}s)")
        last_mtimes: Dict[str, float] = {}
        try:
            while True:
                changed = False
                for f in Path(directory).rglob("*"):
                    if f.is_file() and f.suffix in (".py", ".js", ".ts", ".rs", ".go"):
                        mtime = f.stat().st_mtime
                        if str(f) not in last_mtimes or last_mtimes[str(f)] != mtime:
                            last_mtimes[str(f)] = mtime
                            changed = True
                if changed:
                    log.info("[TestRunner] Changes detected, running tests...")
                    result = self.run_tests(directory, test_command)
                    status = "PASS" if result["passed"] else "FAIL"
                    log.info(f"[TestRunner] {status}")
                time.sleep(interval)
        except KeyboardInterrupt:
            log.info("[TestRunner] Watch stopped")


# ─────────────────────────────────────────────
# TELEGRAM GATEWAY
# ─────────────────────────────────────────────

class TelegramGateway:
    """Telegram bot for remote interaction with ZetaPrime."""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self.offset = 0
        self._active = bool(token and chat_id)

    def send(self, text: str):
        if not self._active:
            return
        try:
            for chunk in [text[i:i+4096] for i in range(0, len(text), 4096)]:
                requests.post(f"{self.base}/sendMessage", json={
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "parse_mode": "Markdown",
                }, timeout=10)
        except Exception as e:
            log.warning(f"[Telegram] Send failed: {e}")

    def poll(self) -> List[str]:
        if not self._active:
            return []
        try:
            resp = requests.get(f"{self.base}/getUpdates", params={
                "offset": self.offset, "timeout": 5
            }, timeout=10)
            updates = resp.json().get("result", [])
            commands = []
            for u in updates:
                self.offset = u["update_id"] + 1
                text = u.get("message", {}).get("text", "")
                if text:
                    commands.append(text)
            return commands
        except Exception:
            return []


# ─────────────────────────────────────────────
# ZETAPRIME — Main Orchestrator
# ─────────────────────────────────────────────

class ZetaPrime:
    """
    AI Developer Agent — Master coder.
    Writes, debugs, optimizes. Maintains the Pantheon.
    """

    COMMANDS = {
        "/write": "Write code from a prompt",
        "/read": "Read a file",
        "/edit": "Edit a file with LLM guidance",
        "/debug": "Debug an error from a traceback",
        "/optimize": "Optimize code for performance",
        "/explain": "Explain code line by line",
        "/refactor": "Refactor messy code",
        "/comment": "Add comments and docstrings",
        "/test": "Generate unit tests",
        "/run": "Run code (bash, python, node)",
        "/runtests": "Run test suite",
        "/convert": "Convert code between languages",
        "/lint": "Run linters (pylint, eslint, ruff)",
        "/search": "Search GitHub for examples",
        "/stackoverflow": "Search Stack Overflow",
        "/git": "Git operations (status, commit, push, pull, branch, log, blame, diff, clone)",
        "/install": "Install packages (pip, npm, pkg)",
        "/boilerplate": "Generate project boilerplate",
        "/watch": "Monitor file changes, auto-run tests",
        "/files": "List files in a directory",
        "/find": "Search for a pattern in files",
        "/history": "Show session history",
        "/status": "System status",
        "/help": "Show available commands",
        "/stop": "Stop ZetaPrime",
    }

    def __init__(self):
        log.info("=== ZetaPrime Initializing ===")
        self.memory = Memory(DB_PATH)
        self.llm = LLM(OLLAMA_BASE, OLLAMA_MODEL)
        self.code_intel = CodeIntelligence(self.llm, self.memory)
        self.test_runner = TestRunner(self.memory)
        self.telegram = TelegramGateway(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        self._running = True

        if self.llm.is_available():
            log.info(f"=== ZetaPrime Ready | Model: {OLLAMA_MODEL} ===")
        else:
            log.warning(f"=== ZetaPrime Ready (LLM offline) | Expected: {OLLAMA_MODEL} ===")

        self.telegram.send("ZetaPrime Online — AI Developer ready.")

    def handle_command(self, raw_input: str) -> str:
        self.memory.add_history("user", raw_input)
        parts = raw_input.strip().split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        handler_map = {
            "/write": self._cmd_write,
            "/read": self._cmd_read,
            "/edit": self._cmd_edit,
            "/debug": self._cmd_debug,
            "/optimize": self._cmd_optimize,
            "/explain": self._cmd_explain,
            "/refactor": self._cmd_refactor,
            "/comment": self._cmd_comment,
            "/test": self._cmd_test,
            "/run": self._cmd_run,
            "/runtests": self._cmd_runtests,
            "/convert": self._cmd_convert,
            "/lint": self._cmd_lint,
            "/search": self._cmd_search,
            "/stackoverflow": self._cmd_stackoverflow,
            "/git": self._cmd_git,
            "/install": self._cmd_install,
            "/boilerplate": self._cmd_boilerplate,
            "/watch": self._cmd_watch,
            "/files": self._cmd_files,
            "/find": self._cmd_find,
            "/history": self._cmd_history,
            "/status": self._cmd_status,
            "/help": self._cmd_help,
            "/stop": self._cmd_stop,
        }

        handler = handler_map.get(cmd)
        if handler:
            try:
                response = handler(args)
            except Exception as e:
                response = f"Error: {e}\n{traceback.format_exc()}"
        elif cmd.startswith("/"):
            response = f"Unknown command: {cmd}\nType /help for available commands."
        else:
            response = self._cmd_chat(raw_input)

        self.memory.add_history("assistant", response[:2000])
        return response

    # ── Command implementations ──

    def _cmd_write(self, args: str) -> str:
        if not args:
            return "Usage: /write <description>\nExample: /write a FastAPI server with CRUD endpoints"
        parts = args.split(" --lang ", 1)
        prompt = parts[0]
        lang = parts[1].strip() if len(parts) > 1 else "python"
        code = self.code_intel.write_code(prompt, lang)
        return f"Generated {lang} code:\n\n{code}"

    def _cmd_read(self, args: str) -> str:
        if not args:
            return "Usage: /read <file_path>"
        result = FileTool.read(args.strip())
        if result["status"] == "ok":
            self.memory.store_file(args.strip(), result["content"], result["language"])
            return f"File: {args.strip()} ({result['language']}, {result['lines']} lines)\n\n{result['content']}"
        return f"Error reading file: {result['content']}"

    def _cmd_edit(self, args: str) -> str:
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /edit <file_path> <instructions>"
        file_path, instructions = parts[0], parts[1]
        file_data = FileTool.read(file_path)
        if file_data["status"] != "ok":
            return f"Cannot read {file_path}: {file_data['content']}"
        system = (
            f"You are ZetaPrime editing a {file_data['language']} file. "
            "Apply the requested changes to the code. Output ONLY the complete modified file, no explanations."
        )
        prompt = f"Current file ({file_path}):\n```\n{file_data['content']}\n```\n\nInstructions: {instructions}"
        new_code = self.llm.generate(prompt, system=system)
        clean = re.sub(r'^```\w*\n|```$', '', new_code.strip(), flags=re.MULTILINE)
        FileTool.write(file_path, clean)
        self.memory.store_file(file_path, clean, file_data["language"])
        return f"Updated {file_path}"

    def _cmd_debug(self, args: str) -> str:
        if not args:
            return "Usage: /debug <file_path>\n(reads file + last error, or paste traceback directly)"
        if Path(args.strip()).is_file():
            file_data = FileTool.read(args.strip())
            if file_data["status"] != "ok":
                return f"Cannot read {args}: {file_data['content']}"
            result = ShellTool.run(f"python {args.strip()}", timeout=30)
            if result["returncode"] == 0:
                return "Code ran without errors."
            return self.code_intel.debug_error(file_data["content"], result["output"], file_data["language"])
        return self.code_intel.debug_error("", args, "python")

    def _cmd_optimize(self, args: str) -> str:
        if not args:
            return "Usage: /optimize <file_path>"
        file_data = FileTool.read(args.strip())
        if file_data["status"] != "ok":
            return f"Cannot read: {file_data['content']}"
        return self.code_intel.optimize(file_data["content"], file_data["language"])

    def _cmd_explain(self, args: str) -> str:
        if not args:
            return "Usage: /explain <file_path>"
        file_data = FileTool.read(args.strip())
        if file_data["status"] != "ok":
            return f"Cannot read: {file_data['content']}"
        return self.code_intel.explain(file_data["content"], file_data["language"])

    def _cmd_refactor(self, args: str) -> str:
        if not args:
            return "Usage: /refactor <file_path>"
        file_data = FileTool.read(args.strip())
        if file_data["status"] != "ok":
            return f"Cannot read: {file_data['content']}"
        return self.code_intel.refactor(file_data["content"], file_data["language"])

    def _cmd_comment(self, args: str) -> str:
        if not args:
            return "Usage: /comment <file_path>"
        file_data = FileTool.read(args.strip())
        if file_data["status"] != "ok":
            return f"Cannot read: {file_data['content']}"
        result = self.code_intel.add_comments(file_data["content"], file_data["language"])
        clean = re.sub(r'^```\w*\n|```$', '', result.strip(), flags=re.MULTILINE)
        FileTool.write(args.strip(), clean)
        return f"Added comments to {args.strip()}"

    def _cmd_test(self, args: str) -> str:
        if not args:
            return "Usage: /test <file_path>"
        file_data = FileTool.read(args.strip())
        if file_data["status"] != "ok":
            return f"Cannot read: {file_data['content']}"
        tests = self.code_intel.generate_tests(file_data["content"], file_data["language"])
        test_path = str(Path(args.strip()).with_stem("test_" + Path(args.strip()).stem))
        clean = re.sub(r'^```\w*\n|```$', '', tests.strip(), flags=re.MULTILINE)
        FileTool.write(test_path, clean)
        return f"Tests written to {test_path}\n\n{clean}"

    def _cmd_run(self, args: str) -> str:
        if not args:
            return "Usage: /run <command>"
        result = ShellTool.run(args, timeout=120)
        return f"[Exit {result.get('returncode', '?')}]\n{result.get('output', '')}"

    def _cmd_runtests(self, args: str) -> str:
        path = args.strip() or "."
        result = self.test_runner.run_tests(path)
        status = "PASSED" if result["passed"] else "FAILED"
        return f"Tests {status}\nCommand: {result['command']}\n\n{result['output']}"

    def _cmd_convert(self, args: str) -> str:
        parts = args.split()
        if len(parts) < 3:
            return "Usage: /convert <file_path> <source_lang> <target_lang>"
        file_path, source_lang, target_lang = parts[0], parts[1], parts[2]
        file_data = FileTool.read(file_path)
        if file_data["status"] != "ok":
            return f"Cannot read: {file_data['content']}"
        return self.code_intel.convert_language(file_data["content"], source_lang, target_lang)

    def _cmd_lint(self, args: str) -> str:
        if not args:
            return "Usage: /lint <file_path>"
        result = Linter.lint(args.strip())
        if result["status"] == "skip":
            return result["message"]
        lines = []
        for r in result["results"]:
            status = "PASS" if r["returncode"] == 0 else "ISSUES"
            lines.append(f"[{r['linter']}] {status}\n{r['output']}")
        return "\n\n".join(lines)

    def _cmd_search(self, args: str) -> str:
        if not args:
            return "Usage: /search <query>"
        result = GitTool.search_github(args)
        if not result["results"]:
            return "No results found."
        lines = []
        for r in result["results"]:
            lines.append(f"* [{r['name']}]({r['url']}) ({r['language']}, {r['stars']} stars)\n  {r['description']}")
        return "\n".join(lines)

    def _cmd_stackoverflow(self, args: str) -> str:
        if not args:
            return "Usage: /stackoverflow <query>"
        result = self.code_intel.search_stackoverflow(args)
        if not result["results"]:
            return "No results found."
        lines = []
        for r in result["results"]:
            answered = "answered" if r["answered"] else "unanswered"
            lines.append(f"* [{r['title']}]({r['link']}) (score: {r['score']}, {answered})")
        return "\n".join(lines)

    def _cmd_git(self, args: str) -> str:
        parts = args.split(maxsplit=1)
        subcmd = parts[0] if parts else "status"
        sub_args = parts[1] if len(parts) > 1 else ""

        git_cmds = {
            "status": lambda: GitTool.status(),
            "diff": lambda: GitTool.diff(),
            "staged": lambda: GitTool.diff(staged=True),
            "log": lambda: GitTool.log(),
            "blame": lambda: GitTool.blame(sub_args) if sub_args else {"output": "Usage: /git blame <file>"},
            "commit": lambda: GitTool.commit(sub_args or "ZetaPrime auto-commit"),
            "push": lambda: GitTool.push(sub_args or None),
            "pull": lambda: GitTool.pull(),
            "branch": lambda: GitTool.branch(sub_args or None),
            "clone": lambda: GitTool.clone(sub_args),
        }

        handler = git_cmds.get(subcmd)
        if handler:
            result = handler()
            return result.get("output", json.dumps(result, indent=2))
        return f"Unknown git subcommand: {subcmd}\nAvailable: {', '.join(git_cmds.keys())}"

    def _cmd_install(self, args: str) -> str:
        parts = args.split()
        if not parts:
            return "Usage: /install <pip|npm|pkg> <package1> [package2 ...]"
        manager = parts[0]
        packages = parts[1:]
        if manager == "pip":
            result = PackageManager.pip_install(packages)
        elif manager == "npm":
            result = PackageManager.npm_install(packages)
        elif manager == "pkg":
            result = PackageManager.pkg_install(packages)
        else:
            return f"Unknown package manager: {manager}. Use pip, npm, or pkg."
        return result.get("output", "Done")

    def _cmd_boilerplate(self, args: str) -> str:
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /boilerplate <type> <name>\nExample: /boilerplate fastapi my-api"
        project_type, name = parts[0], parts[1]
        files = self.code_intel.generate_boilerplate(project_type, name)
        project_dir = WORKSPACE / name
        project_dir.mkdir(parents=True, exist_ok=True)
        created = []
        for fpath, content in files.items():
            full_path = project_dir / fpath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            created.append(str(fpath))
        return f"Created project '{name}' in {project_dir}\nFiles: {', '.join(created)}"

    def _cmd_watch(self, args: str) -> str:
        parts = args.split(maxsplit=1)
        directory = parts[0] if parts else "."
        test_cmd = parts[1] if len(parts) > 1 else None
        self.test_runner.watch(directory, test_cmd)
        return "Watch stopped."

    def _cmd_files(self, args: str) -> str:
        directory = args.strip() or "."
        files = FileTool.list_files(directory)
        if not files:
            return f"No files found in {directory}"
        return f"Files in {directory} ({len(files)}):\n" + "\n".join(f"  {f}" for f in files[:100])

    def _cmd_find(self, args: str) -> str:
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /find <directory> <pattern>"
        result = FileTool.search(parts[0], parts[1])
        if not result["matches"]:
            return "No matches found."
        return f"Found {result['count']} matches:\n" + "\n".join(result["matches"][:30])

    def _cmd_history(self, args: str) -> str:
        limit = int(args) if args.strip().isdigit() else 10
        history = self.memory.get_history(limit)
        lines = []
        for h in history:
            role = "You" if h["role"] == "user" else "Zeta"
            lines.append(f"[{h['timestamp'][:19]}] {role}: {h['content'][:120]}")
        return "\n".join(lines) or "No history."

    def _cmd_status(self, args: str) -> str:
        llm_status = "online" if self.llm.is_available() else "offline"
        return (
            f"ZetaPrime Status\n"
            f"  LLM: {llm_status} ({OLLAMA_MODEL})\n"
            f"  Workspace: {WORKSPACE}\n"
            f"  Database: {DB_PATH}\n"
            f"  Telegram: {'active' if self.telegram._active else 'inactive'}"
        )

    def _cmd_help(self, args: str) -> str:
        lines = ["ZetaPrime Commands:\n"]
        for cmd, desc in self.COMMANDS.items():
            lines.append(f"  {cmd:20s} {desc}")
        return "\n".join(lines)

    def _cmd_stop(self, args: str) -> str:
        self._running = False
        return "ZetaPrime shutting down..."

    def _cmd_chat(self, message: str) -> str:
        history = self.memory.get_history(6)
        context = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
        system = (
            "You are ZetaPrime, an AI Developer agent. Master coder. "
            "You help write, debug, optimize, and maintain code. "
            "Be concise, technical, and helpful. If the user seems to want a command, "
            "suggest the appropriate /command."
        )
        prompt = f"Conversation:\n{context}\n\nuser: {message}\nassistant:"
        return self.llm.generate(prompt, system=system)

    # ── Main loops ──

    def run_interactive(self):
        """Interactive CLI mode."""
        print("\n=== ZetaPrime — AI Developer Agent ===")
        print("Type /help for commands, or chat naturally.\n")
        while self._running:
            try:
                user_input = input("You > ").strip()
                if not user_input:
                    continue
                response = self.handle_command(user_input)
                print(f"\nZeta > {response}\n")
            except KeyboardInterrupt:
                print("\nZetaPrime shutting down.")
                break
            except EOFError:
                break

    def run_telegram(self):
        """Telegram bot mode."""
        log.info("[Main] Starting Telegram loop")
        self.telegram.send("ZetaPrime Online — Type /help for commands.")
        while self._running:
            try:
                for message in self.telegram.poll():
                    log.info(f"[Telegram] Received: {message[:80]}")
                    response = self.handle_command(message)
                    self.telegram.send(response)
                time.sleep(2)
            except KeyboardInterrupt:
                self._running = False
            except Exception as e:
                log.error(f"[Main] Error: {e}")
                time.sleep(5)
        self.telegram.send("ZetaPrime Offline.")
        log.info("[Main] ZetaPrime shutdown complete")

    def run(self):
        """Auto-detect mode: Telegram if configured, else interactive CLI."""
        if self.telegram._active:
            self.run_telegram()
        else:
            self.run_interactive()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    bot = ZetaPrime()
    bot.run()
