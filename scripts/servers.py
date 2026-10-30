"""Start, probe and stop the servers of an ASAP deployment (used by acceptance.py and handy interactively).

Every server runs in its own session and logs to a file. stop() signals only the processes started here (the
launcher plus the descendants seen while it ran), never anything else on the machine.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psutil


def _process_tools():
    """Keep acceptance analysis importable without the GPU environment's dependencies."""
    try:
        return importlib.import_module("psutil")
    except ImportError as exc:
        raise RuntimeError("Starting acceptance servers requires psutil in the SGLang environment") from exc

MODEL = "/root/paddlejob/workspace/lhd_models/DeepSeek-V4-Flash-0731"
TINY_OVERRIDE = '{"num_hidden_layers": 4, "n_routed_experts": 128, "num_hash_layers": 0}'
ATTN_READY_RE = re.compile(r"\[ASAP\] attn rank ready: replica=(?P<replica>\d+) tp_rank=(?P<tp>\d+) "
                           r"world_size=(?P<world>\S+) gpu=(?P<gpu>\d+) pid=(?P<pid>\d+) hooks=(?P<ok>\d+)/(?P<total>\d+)"
                           r" afd=(?P<afd>\w+)")
ARGS_OK_RE = re.compile(r"\[ASAP\] server args ok: role=(\w+) afd=(\w+) hooks=(\d+)/(\d+)")
FFN_RANK_RE = re.compile(r"\[ASAP\] ffn rank (\d+) (ready|stats): (\{.*\})")


def server_env(gpus: str, **extra) -> dict:
    """Environment of a server. The venv's bin directory goes first on PATH, as after ``source activate``:
    FlashInfer's JIT runs whichever ninja PATH finds, and ninja 1.11 (/usr/local/bin) and 1.13 (the venv) do not
    share their dependency logs, so switching between them recompiles the MXFP4 MoE module (~15 minutes)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASAP_")}
    env.pop("NCCL_DEBUG_FILE", None)
    env.update({"CUDA_VISIBLE_DEVICES": gpus, "SGLANG_PLUGINS": "sgl_asap_plugin", "NCCL_DEBUG": "WARN",
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0", "PYTHONDONTWRITEBYTECODE": "1",
                "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")})
    env.update(extra)
    return env


def _alive(p: psutil.Process) -> bool:
    psutil = _process_tools()
    try:
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class Server:
    """An HTTP sglang server; ready when /health answers 200."""

    def __init__(self, name, argv, env, log_dir, port):
        self.name, self.argv, self.env, self.port = name, argv, env, port
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self.proc, self._family, self._log = None, [], None

    def start(self):
        _process_tools()  # Validate cleanup support before creating any process.
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        return self

    def is_ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def wait_ready(self, timeout) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline and self.proc.poll() is None:
            if self.is_ready():
                return True
            time.sleep(3)
        return False

    def log_text(self) -> str:
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def attn_ranks(self):
        return [m.groupdict() for m in ATTN_READY_RE.finditer(self.log_text())]

    def descendants(self):
        psutil = _process_tools()
        root = psutil.Process(self.proc.pid)
        self._family = [root] + root.children(recursive=True)
        return self._family[1:]

    def stop(self):
        if self.proc is None:
            return
        psutil = _process_tools()
        try:
            self.descendants()
        except psutil.NoSuchProcess:
            pass
        for sig, grace in ((signal.SIGTERM, 25), (signal.SIGKILL, 10)):
            for p in self._family:
                for s in (signal.SIGCONT, sig):  # a stopped process reacts to SIGTERM only after SIGCONT
                    try:
                        p.send_signal(s)
                    except psutil.NoSuchProcess:
                        break
            end = time.time() + grace
            while time.time() < end and any(_alive(p) for p in self._family):
                time.sleep(0.5)
            if not any(_alive(p) for p in self._family):
                break
        if self._log:
            self._log.close()


class FfnService(Server):
    """``python -m sgl_asap_plugin.ffn``: ready when the launcher prints "[ASAP] ffn ready"."""

    def __init__(self, name, argv, env, log_dir):
        super().__init__(name, argv, env, log_dir, port=None)

    def is_ready(self) -> bool:
        return "[ASAP] ffn ready" in self.log_text()

    def reports(self, kind):
        """{rank: dict} of the "ready" or "stats" reports of the ranks."""
        return {int(m.group(1)): json.loads(m.group(3)) for m in FFN_RANK_RE.finditer(self.log_text())
                if m.group(2) == kind}

    def stop(self):
        """SIGTERM to the launcher only: it stops its ranks, which print their statistics; then the hard stop."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                pass
        super().stop()


def start_server(make, timeout, attempts=3):
    """Start ``make()`` and wait until ready -> (server, attempts used, ready).

    Retries only for a known race: sglang's DP controller picks each replica's rendezvous port with bind(0) and
    releases it before the replica binds it, and this host's ephemeral range (10000-61000) lets any outgoing
    connection take the port in between (EADDRINUSE in the log). Every other failure is final.
    """
    for used in range(1, attempts + 1):
        srv = make().start()
        if srv.wait_ready(timeout):
            return srv, used, True
        racy = "EADDRINUSE" in srv.log_text()
        srv.stop()
        if not racy or used == attempts:
            return srv, used, False


def launch_server_argv(model, port, *flags):
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--host", "127.0.0.1",
            "--port", str(port), *map(str, flags)]


def attn_argv(model, port, dp, ffn_ranks, rdv_port, n_max, *flags):
    """An AFD attention role: D plain replicas of TP 1, experts on the F service at rdv_port."""
    return launch_server_argv(model, port, "--dp-size", dp, "--tp-size", 1, "--asap-role", "attn",
                              "--asap-ffn-ranks", ffn_ranks, "--asap-rdv-addr", f"127.0.0.1:{rdv_port}",
                              "--asap-n-max", n_max, "--chunked-prefill-size", n_max, "--max-prefill-tokens", n_max,
                              *flags)


def ffn_argv(model, ranks, num_replicas, rdv_port, n_max, *flags):
    return [sys.executable, "-m", "sgl_asap_plugin.ffn", "--model-path", model, "--tp-size", str(ranks),
            "--ep-size", str(ranks), "--asap-num-replicas", str(num_replicas), "--asap-replica-tp", "1",
            "--asap-rdv-addr", f"127.0.0.1:{rdv_port}", "--asap-n-max", str(n_max), *map(str, flags)]


def generate(port, prompt=None, *, input_ids=None, max_new_tokens=1, rank=None, logprobs=False, logprob_start_len=0,
             timeout=600.0):
    """POST /generate (greedy) -> (ok, seconds, response dict or error text). With ``logprobs``: top-5 of the
    output tokens and the logprob of every prompt token from ``logprob_start_len`` on."""
    body = {"sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0}}
    if input_ids is not None:
        body["input_ids"] = input_ids
    else:
        body["text"] = prompt if prompt is not None else "The capital of France is"
    if rank is not None:
        body["routed_dp_rank"] = rank
    if logprobs:
        body.update(return_logprob=True, logprob_start_len=logprob_start_len, top_logprobs_num=5)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read())
            return resp.status == 200 and "text" in out, time.time() - start, out
    except (urllib.error.URLError, OSError, ValueError) as exc:
        detail = exc.read().decode(errors="replace")[:500] if hasattr(exc, "read") else str(exc)
        return False, time.time() - start, detail
