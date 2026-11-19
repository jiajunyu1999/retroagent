"""Terminal-once, isolated local Codex CLI transport. No API-key fallback."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

from luna_tool_client import adapt
from retrosynthesis_client import atomic_json, digest
from rsi_fs_sandbox import restrict, system_reads

MODEL = 'gpt-6.1-sol'
EFFORT = 'medium'
SAFE_ROOT = Path('/tmp/retroagent-clean-exec')


class Client:
    def __init__(self, folder, *, model=MODEL, effort=EFFORT, proxy=None,
                 timeout=300, **unused):
        if (model, effort) != (MODEL, EFFORT):
            raise ValueError('This experiment fixes gpt-6.1-sol / medium')
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.model, self.effort, self.timeout = model, effort, timeout

    def attempt(self, body):
        # Compatibility with frozen solver binding; generate never calls HTTP.
        raise RuntimeError('Use generate: local exec only')

    def invoke(self, prompt, folder):
        SAFE_ROOT.mkdir(exist_ok=True)
        SAFE_ROOT.chmod(0o755)
        sandbox = Path(tempfile.mkdtemp(prefix='call-', dir=SAFE_ROOT))
        try:
            sandbox.chmod(0o755)
            home, work, output, temp = [sandbox / p for p in ('home', 'work', 'output', 'tmp')]
            for path in (home, work, output, temp):
                path.mkdir(); os.chown(path, 65534, 65534); path.chmod(0o700)
            work.chmod(0o555)
            shutil.copyfile(Path(os.environ.get('RETROAGENT_CODEX_AUTH', Path.home()/'.codex'/'auth.json')), home / 'auth.json')
            os.chown(home / 'auth.json', 65534, 65534)
            (home / 'auth.json').chmod(0o600)
            env = {k: v for k, v in os.environ.items() if k in (
                'LANG', 'LC_ALL', 'TZ', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
                'SSL_CERT_FILE', 'SSL_CERT_DIR')}
            env.update(CODEX_HOME=str(home), TMPDIR=str(temp), PATH='/usr/bin:/bin',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            target = output / 'answer.txt'
            args = [os.environ.get('RETROAGENT_CODEX_BIN', '/usr/bin/codex'), 'exec', '--model', self.model,
                    '--ignore-user-config', '--ignore-rules', '--ephemeral',
                    '--disable','apps','--disable','plugins','--disable','remote_plugin',
                    '--disable','plugin_discovery_prefetch','--disable','workspace_dependencies',
                    '--disable','shell_tool','--disable','browser_use','--disable','multi_agent',
                    '--sandbox', 'danger-full-access', '-c', 'approval_policy="never"',
                    '-c', 'model_reasoning_effort="medium"', '-c', 'project_doc_max_bytes=0',
                    '-c', 'features.memories=false', '-c', 'features.skip_host_skill_discovery=true',
                    '-c', 'features.skill_search=false', '-c', 'web_search="disabled"',
                    '--skip-git-repo-check', '--json', '-C', str(work), '-o', str(target), '-']
            fixed = ('Return only the requested JSON. Use only the supplied observations and '
                     'learned chemistry. Do not read files, execute commands, search the web, '
                     'or call any tools. Never invent tool observations.\n')
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'prompt.txt').write_text(prompt)
            began = time.monotonic()
            with (folder / 'events.jsonl').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
                process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=stdout,
                    stderr=stderr, text=True, cwd=work, env=env, start_new_session=True,
                    preexec_fn=lambda: restrict(system_reads() + [work], [home, output, temp, Path('/dev/null')]))
                atomic_json(folder / 'process.json', dict(pid=process.pid, sandbox=str(sandbox),
                    start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21]))
                try:
                    process.communicate(fixed + prompt, timeout=self.timeout)
                    code = process.returncode
                except subprocess.TimeoutExpired:
                    code = 124
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
            events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines() if line.strip()]
            forbidden = [e for e in events if e.get('item', {}).get('type') in (
                'command_execution', 'file_change', 'mcp_tool_call', 'web_search', 'dynamic_tool_call')]
            terminal = [e for e in events if e.get('type') == 'turn.completed']
            thread = next((e.get('thread_id') for e in events if e.get('type') == 'thread.started'), None)
            item = dict(transport='local_codex_exec', exit_code=code,
                raw_text=target.read_text() if target.exists() else '',
                usage=terminal[-1].get('usage') if terminal else None,
                thread_id=thread, elapsed_seconds=time.monotonic()-began,
                terminal_event='local_exec.finished', forbidden_tool_events=len(forbidden),
                requested_model=self.model, reasoning_effort=self.effort,
                model_verification='CLI requested model; server model identity not exposed by JSONL',
                fresh_session=True, landlock=True, uid=65534, api_key_file_used=False)
            if code or not terminal or forbidden or not item['raw_text']:
                item['error'] = 'Local exec failed, incomplete, empty, or used a forbidden tool'
            atomic_json(folder / 'invocation.json', item)
            return item
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    def generate(self, message, tid):
        key = digest(dict(transport='retroagent_clean_exec', target_id=tid,
                          model=self.model, effort=self.effort, prompt=message))
        path = self.folder / (key + '.json')
        with (self.folder / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                return json.loads(path.read_text())
            record = dict(request_key=key, target_id=tid, model=self.model, effort=self.effort,
                prompt=message, status='failed', tools=[], attempts=[], transport='local_codex_exec')
            # Write an immutable pending marker before invocation. A crash must not silently reroll.
            atomic_json(path, dict(record, invocation_started=True))
            try:
                item = self.invoke(message, self.folder / key)
                record['attempts'].append(item)
                if not item.get('error'):
                    record.update(route=adapt(item['raw_text'], tid, 'route'), status='completed',
                        returned_model=self.model, model_verification=item['model_verification'],
                        usage=item['usage'], thread_id=item['thread_id'])
            except Exception as exc:
                record['error'] = type(exc).__name__ + ': ' + str(exc)
            atomic_json(path, record)
            return record
