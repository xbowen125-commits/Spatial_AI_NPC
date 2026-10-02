"""NPC TTS 顺序播放队列；所有阻塞操作都在独立线程中。"""

import queue
import threading
import time
from dataclasses import dataclass

from .audio_config import TTS_ENABLED
from .tts_engine import LocalTTSEngine


@dataclass(frozen=True)
class SpeechOutputLifecycle:
    phase: str
    response_event: object
    error: str | None = None
    latency_ms: int = 0


class SpeechOutputController:
    def __init__(self):
        self.playing = False
        self.status = "DISABLED"
        self._queue = queue.Queue()
        self._events = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = None
        self._engine = None

    @property
    def busy(self):
        return self.playing or self._queue.unfinished_tasks > 0

    def start(self):
        if not TTS_ENABLED:
            print("TTS 已在配置中关闭。")
            return False
        self.status = "READY"
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="LocalTTS",
            daemon=True,
        )
        self._worker.start()
        return True

    def enqueue(self, response_event):
        if self.status == "DISABLED" or self._stop_event.is_set():
            return False
        if response_event is None or not str(response_event.text).strip():
            print("NPC 回复为空，未加入 TTS 队列。")
            return False
        # 提交时间用于评估 TTS 队列等待 + 实际播放的总耗时。
        self._queue.put((response_event, time.monotonic()))
        return True

    def poll_event(self):
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                self._queue.task_done()
                break

            response_event, started_at = job
            started = False
            try:
                if self._engine is None:
                    self.status = "INITIALIZING"
                    self._engine = LocalTTSEngine()
                self.status = "PLAYING"
                self.playing = True
                started = True
                self._events.put(SpeechOutputLifecycle("started", response_event))
                self._engine.speak(response_event.text, response_event.language)
            except Exception as error:
                self._events.put(
                    SpeechOutputLifecycle(
                        "error_after_start" if started else "error_before_start",
                        response_event,
                        str(error),
                        int((time.monotonic() - started_at) * 1000),
                    )
                )
            finally:
                self.playing = False
                self.status = "READY"
                if started:
                    self._events.put(
                        SpeechOutputLifecycle(
                            "finished",
                            response_event,
                            latency_ms=int(
                                (time.monotonic() - started_at) * 1000
                            ),
                        )
                    )
                self._queue.task_done()

    def shutdown(self):
        self._stop_event.set()
        if self._engine is not None:
            self._engine.stop()
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=3.0)
        self.playing = False
        self.status = "DISABLED"
