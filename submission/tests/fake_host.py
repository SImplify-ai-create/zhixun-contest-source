# -*- coding: utf-8 -*-
"""本地假宿主：用 JSONL 替身驱动 `app/harness.py` 的三个入口。

不联网、不挂 /model、不调用真实模型；只验证协议形状与 stdout/stderr 分工。
"""

import os
import subprocess
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(APP_DIR, "app", "harness.py")


def _env(extra=None):
    env = os.environ.copy()
    for key in ("CONTEST_SUBMISSION", "CONTEST_ARTIFACT_PATH", "CONTEST_ARTIFACT_MANIFEST",
                "CONTEST_PREDICTION_CONTEXT", "CONTEST_DATASET_KEY", "CONTEST_TASK",
                "CONTEST_MODEL_DIR"):
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def run(command, stdin_text="", env=None, cwd=None, timeout=120):
    """跑一个入口，返回 (rc, stdout_lines, stderr_text)。"""
    rc, raw, err = run_bytes(command, stdin_text, env=env, cwd=cwd, timeout=timeout)
    out = raw.decode("utf-8", "replace")
    return rc, [ln for ln in out.splitlines() if ln.strip()], err


def run_bytes(command, stdin_text="", env=None, cwd=None, timeout=120):
    """跑一个入口，返回 (rc, stdout_bytes, stderr_text)。

    确定性审计用：逐字节比对 stdout，不经过解码/分行，任何空白或顺序差异都会暴露。
    """
    proc = subprocess.run(
        [sys.executable, "-B", HARNESS, command],
        input=(stdin_text or "").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(env),
        cwd=cwd,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr.decode("utf-8", "replace")


def stream_predict(lines, env=None, cwd=None, timeout=30):
    """逐行喂入，每喂一行就要求立刻拿到一行响应（验证 flush）。"""
    proc = subprocess.Popen(
        [sys.executable, "-B", HARNESS, "predict"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(env),
        cwd=cwd,
    )
    replies = []
    try:
        for line in lines:
            proc.stdin.write((line + "\n").encode("utf-8"))
            proc.stdin.flush()
            replies.append(_readline_with_timeout(proc, timeout))
        proc.stdin.close()
        rc = proc.wait(timeout=timeout)
        err = proc.stderr.read().decode("utf-8", "replace")
    finally:
        if proc.poll() is None:
            proc.kill()
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()
    return rc, replies, err


def _readline_with_timeout(proc, timeout):
    """在管道上带超时地读一行。Windows 下没有 select，用线程兜底。"""
    import queue
    import threading

    box = queue.Queue()

    def _reader():
        try:
            box.put(proc.stdout.readline())
        except Exception as exc:  # pragma: no cover
            box.put(exc)

    worker = threading.Thread(target=_reader, daemon=True)
    worker.start()
    try:
        item = box.get(timeout=timeout)
    except queue.Empty:
        raise AssertionError("predict 在 %.0fs 内没有为当前行写出结果（未 flush 或阻塞）" % timeout)
    if isinstance(item, Exception):  # pragma: no cover
        raise item
    return item.decode("utf-8", "replace").strip()
