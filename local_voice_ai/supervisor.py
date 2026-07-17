"""Async process supervisor.

Spawns a list of child processes, line-prefixes their stdout/stderr through the
parent logger, polls per-child readiness URLs, and propagates SIGTERM/SIGINT to
all children for a clean shutdown. A child that dies before becoming ready is a
fatal error; a child that dies after becoming ready is restarted with linear
backoff up to a cap.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from collections import deque
from typing import Awaitable, Callable, Optional

import httpx

logger = logging.getLogger("supervisor")


@dataclass
class ChildSpec:
    name: str
    argv: list[str]
    role: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    ready_url: Optional[str] = None
    ready_timeout: float = 180.0
    max_restarts: int = 5


@dataclass
class _Child:
    spec: ChildSpec
    process: Optional[asyncio.subprocess.Process] = None
    restart_count: int = 0
    ready: bool = False
    pump_task: Optional[asyncio.Task] = None
    watch_task: Optional[asyncio.Task] = None
    recent_output: deque[str] = field(default_factory=lambda: deque(maxlen=40))


class Supervisor:
    def __init__(self, specs: list[ChildSpec]) -> None:
        self._children: list[_Child] = [_Child(spec=spec) for spec in specs]
        self._stop_event = asyncio.Event()
        self._http: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    async def start_all(self) -> None:
        """Spawn every child and wait for each to pass its readiness probe."""
        if not self._children:
            return

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(2.0))

        # Spawn all children in parallel; each one's readiness wait is independent.
        async with self._lock:
            await asyncio.gather(*(self._start(child) for child in self._children))
            await asyncio.gather(*(self._await_ready(child) for child in self._children))

    async def replace_roles(self, roles: set[str], specs: list[ChildSpec]) -> None:
        """Replace running children for the given roles and wait until ready."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(2.0))

        async with self._lock:
            old_children = [child for child in self._children if (child.spec.role or child.spec.name) in roles]
            self._children = [
                child for child in self._children if (child.spec.role or child.spec.name) not in roles
            ]

            await asyncio.gather(*(self._stop_child(child) for child in old_children))

            new_children = [
                _Child(spec=spec)
                for spec in specs
                if (spec.role or spec.name) in roles
            ]
            self._children.extend(new_children)
            await asyncio.gather(*(self._start(child) for child in new_children))
            await asyncio.gather(*(self._await_ready(child) for child in new_children))

    async def run_until_signal(self) -> int:
        """Wait until a signal arrives or any child exceeds restart budget."""
        loop = asyncio.get_running_loop()

        def _handle(signum: int) -> None:
            logger.info("received signal %s; shutting down", signal.Signals(signum).name)
            self._stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _handle, sig)
            except NotImplementedError:  # pragma: no cover — Windows
                pass

        await self._stop_event.wait()
        await self.shutdown()
        return 0

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Terminate all children. SIGTERM, wait, SIGKILL if needed."""
        self._stop_event.set()

        for child in self._children:
            if child.process and child.process.returncode is None:
                logger.info("[%s] terminating", child.spec.name)
                try:
                    child.process.terminate()
                except ProcessLookupError:
                    pass

        deadline = asyncio.get_running_loop().time() + timeout
        for child in self._children:
            if not child.process:
                continue
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            try:
                await asyncio.wait_for(child.process.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning("[%s] killing (did not exit in %.1fs)", child.spec.name, timeout)
                try:
                    child.process.kill()
                except ProcessLookupError:
                    pass
                await child.process.wait()

        for child in self._children:
            for task in (child.pump_task, child.watch_task):
                if task and not task.done():
                    task.cancel()

        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ---------------- internals ----------------

    async def _start(self, child: _Child) -> None:
        env = {**os.environ, **child.spec.env}
        logger.info("[%s] starting: %s", child.spec.name, " ".join(child.spec.argv))
        child.process = await asyncio.create_subprocess_exec(
            *child.spec.argv,
            cwd=child.spec.cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        child.pump_task = asyncio.create_task(
            self._pump_output(child),
            name=f"pump:{child.spec.name}",
        )
        child.watch_task = asyncio.create_task(
            self._watch_exit(child),
            name=f"watch:{child.spec.name}",
        )

    async def _stop_child(self, child: _Child, timeout: float = 10.0) -> None:
        child.ready = False
        if child.watch_task and not child.watch_task.done():
            child.watch_task.cancel()
        if child.process and child.process.returncode is None:
            logger.info("[%s] terminating for reload", child.spec.name)
            try:
                child.process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(child.process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("[%s] killing for reload", child.spec.name)
                try:
                    child.process.kill()
                except ProcessLookupError:
                    pass
                await child.process.wait()

        for task in (child.pump_task, child.watch_task):
            if task and not task.done():
                task.cancel()

    async def _pump_output(self, child: _Child) -> None:
        assert child.process is not None
        prefix = f"[{child.spec.name}]"

        async def pump(stream: Optional[asyncio.StreamReader], level: int) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    return
                decoded = line.decode(errors="replace").rstrip()
                child.recent_output.append(decoded)
                logger.log(level, "%s %s", prefix, decoded)

        await asyncio.gather(
            pump(child.process.stdout, logging.INFO),
            pump(child.process.stderr, logging.WARNING),
        )

    async def _await_ready(self, child: _Child) -> None:
        if child.spec.ready_url is None:
            child.ready = True
            return

        assert self._http is not None
        deadline = asyncio.get_running_loop().time() + child.spec.ready_timeout
        delay = 0.5

        while True:
            if self.stopping:
                raise RuntimeError(f"{child.spec.name}: aborted during readiness wait")
            if child.process and child.process.returncode is not None:
                if child.pump_task:
                    try:
                        await asyncio.wait_for(asyncio.shield(child.pump_task), timeout=1.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                recent = "\n".join(child.recent_output)
                suffix = f"\nRecent output:\n{recent}" if recent else ""
                raise RuntimeError(
                    f"{child.spec.name}: exited (rc={child.process.returncode}) before ready"
                    f"{suffix}"
                )
            try:
                resp = await self._http.get(child.spec.ready_url)
                if resp.status_code < 400:
                    child.ready = True
                    logger.info("[%s] ready", child.spec.name)
                    return
            except (httpx.RequestError, httpx.HTTPError):
                pass

            now = asyncio.get_running_loop().time()
            if now >= deadline:
                raise TimeoutError(
                    f"{child.spec.name}: readiness probe {child.spec.ready_url} did not "
                    f"succeed within {child.spec.ready_timeout:.0f}s"
                )
            await asyncio.sleep(min(delay, deadline - now))
            delay = min(delay * 1.5, 5.0)

    async def _watch_exit(self, child: _Child) -> None:
        assert child.process is not None
        rc = await child.process.wait()
        if self.stopping:
            return
        if not child.ready:
            return  # _await_ready will report
        logger.warning("[%s] exited (rc=%s); restart_count=%d", child.spec.name, rc, child.restart_count)

        if child.restart_count >= child.spec.max_restarts:
            logger.error(
                "[%s] exceeded max_restarts=%d; signalling supervisor shutdown",
                child.spec.name,
                child.spec.max_restarts,
            )
            self._stop_event.set()
            return

        child.restart_count += 1
        backoff = min(2.0 * child.restart_count, 10.0)
        await asyncio.sleep(backoff)
        if self.stopping:
            return

        child.ready = False
        await self._start(child)
        try:
            await self._await_ready(child)
        except Exception as exc:
            logger.error("[%s] failed to recover: %s", child.spec.name, exc)
            self._stop_event.set()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
