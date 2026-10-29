"""Helpers for the M0 acceptance run.

Servers are started in their own session, and cleanup signals only the processes started
here (the launcher plus the descendants seen while it ran), never anything else.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

import psutil

TINY_OVERRIDE = '{"num_hidden_layers": 4, "n_routed_experts": 128, "num_hash_layers": 0}'
READY_RE = re.compile(
    r"\[ASAP\] scheduler ready: role=(?P<role>\w+) dp_rank=(?P<dp>\S+) tp_rank=(?P<tp>\S+) "
    r"tp_size=(?P<tp_size>\S+) world_size=(?P<world>\S+) gpu_id=(?P<gpu>\S+) pid=(?P<pid>\d+) "
    r"hooks=(?P<ok>\d+)/(?P<total>\d+)")
ARGS_OK_RE = re.compile(r"\[ASAP\] server args ok: role=(\w+) hooks=(\d+)/(\d+)")


def server_env(gpus: str, real_weights: bool = False) -> dict:
    """Environment of a server. The M0 runs use dummy weights and pretend the experts are FP8;
    real checkpoints keep the layout detected from the files."""
    env = dict(os.environ)
    env.pop("ASAP_ROLE", None)
    env.pop("NCCL_DEBUG_FILE", None)
    env.update({"CUDA_VISIBLE_DEVICES": gpus, "SGLANG_PLUGINS": "sgl_asap_plugin", "NCCL_DEBUG": "WARN",
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"})
    if real_weights:
        env.pop("SGLANG_DSV4_FP4_EXPERTS", None)
    else:
        env["SGLANG_DSV4_FP4_EXPERTS"] = "0"
    return env


def _alive(p: psutil.Process) -> bool:
    try:
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class Server:
    def __init__(self, name, argv, env, log_dir, port):
        self.argv, self.env, self.port = argv, env, port
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self.proc = None
        self._family = []

    def start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def wait_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(3)
        return False

    def log_text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def ready_lines(self):
        return [m.groupdict() for m in READY_RE.finditer(self.log_text())]

    def scheduler_procs(self):
        """Live ``sglang::scheduler*`` descendants of the launcher (sglang sets the process title)."""
        root = psutil.Process(self.proc.pid)
        procs = [p for p in root.children(recursive=True) if "sglang::scheduler" in " ".join(p.cmdline())]
        self._family = [root] + root.children(recursive=True)
        return procs

    def stop(self):
        if self.proc is None:
            return
        try:
            root = psutil.Process(self.proc.pid)
            self._family = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        for sig, grace in ((signal.SIGTERM, 25), (signal.SIGKILL, 10)):
            for p in self._family:  # a stopped process only reacts to SIGTERM after SIGCONT
                for s in (signal.SIGCONT, sig):
                    try:
                        p.send_signal(s)
                    except psutil.NoSuchProcess:
                        break
            end = time.time() + grace
            while time.time() < end and any(_alive(p) for p in self._family):
                time.sleep(0.5)
            if not any(_alive(p) for p in self._family):
                break
        self._log.close()


class LogServer(Server):
    """A background service that is ready when a line matching ``ready_re`` shows up in its log."""

    def __init__(self, name, argv, env, log_dir, ready_re):
        super().__init__(name, argv, env, log_dir, port=None)
        self.ready_re = re.compile(ready_re)

    def wait_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            if self.ready_re.search(self.log_text()):
                return True
            time.sleep(1)
        return False

    def stop(self):
        """Ask the service to finish (SIGTERM to its main process only: it stops its own workers and
        prints their statistics), then fall back to the hard stop of ``Server``."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        super().stop()


def generate(port, rank=None, timeout=60.0, text="The capital of France is"):
    """POST /generate for one token. Returns (succeeded, seconds)."""
    body = {"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
    if rank is not None:
        body["routed_dp_rank"] = rank
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = resp.status == 200 and "text" in json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        ok = False
    return ok, time.time() - start


def start_server(make, timeout, attempts=3):
    """Start ``make()`` and wait until healthy -> (server, attempts_used, ready).

    Retries only for a known sglang/host race: the DP controller picks each replica's
    rendezvous port with bind(0) and releases it before the replica binds it, and this host's
    ephemeral range (10000-61000) lets any outgoing connection take it in between, which shows
    up as EADDRINUSE in the log. Every other failure is final.
    """
    for used in range(1, attempts + 1):
        srv = make()
        srv.start()
        if srv.wait_ready(timeout):
            return srv, used, True
        racy = "EADDRINUSE" in srv.log_text()
        srv.stop()
        if not racy or used == attempts:
            return srv, used, False
