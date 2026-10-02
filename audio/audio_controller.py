"""非阻塞麦克风录制、VAD 与后台 STT 控制器。"""

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .audio_config import (
    BLOCK_SIZE,
    CHANNELS,
    MAX_RECORDING_SECONDS,
    MIC_DEVICE,
    MIN_RECORDING_SECONDS,
    PUSH_TO_TALK,
    SAMPLE_RATE,
)
from .speech_recognition import LocalSpeechRecognizer
from .voice_activity import VoiceActivityDetector


@dataclass(frozen=True)
class RecognitionOutcome:
    result: object | None = None
    error: str | None = None
    latency_ms: int = 0


class AudioController:
    """音频失败时保持禁用状态，不影响视觉主循环。"""

    def __init__(self):
        self.mic_status = "DISABLED"
        self.is_recording = False
        self.is_speaking = False
        self.current_rms = 0.0
        self.last_speech = ""
        self.input_suppressed = False
        self._sd = None
        self._stream = None
        self._chunks = []
        self._recording_started = 0.0
        self._speech_detected_in_recording = False
        self._auto_stop_requested = False
        self._lock = threading.Lock()
        self._vad = VoiceActivityDetector()
        self._jobs = queue.Queue()
        self._results = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = None

    def start(self):
        """打开默认或指定麦克风；失败时只禁用音频。"""
        if not PUSH_TO_TALK:
            print("当前版本仅支持 Push-To-Talk；音频模块保持禁用。")
            return False
        try:
            import sounddevice as sd

            sd.check_input_settings(
                device=MIC_DEVICE,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
            )
            self._sd = sd
            self.mic_status = "READY"
            self._worker = threading.Thread(
                target=self._stt_worker,
                name="LocalSTT",
                daemon=True,
            )
            self._worker.start()
            print("麦克风已就绪；仅在按 SPACE 后打开输入流。")
            return True
        except Exception as error:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self.mic_status = "UNAVAILABLE"
            print(f"音频模块不可用，视觉系统继续运行：{error}")
            return False

    def toggle_recording(self):
        if self.input_suppressed:
            print("NPC 正在说话，Half-Duplex 暂停玩家录音。")
            return False
        if self.mic_status != "READY":
            print("麦克风未就绪，无法开始录音。")
            return False
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
        return self.is_recording

    def start_recording(self):
        if self.input_suppressed:
            return
        with self._lock:
            self._chunks = []
            self._vad.reset()
            self.is_speaking = False
            self.is_recording = True
            self._auto_stop_requested = False
            self._recording_started = time.monotonic()
            self._speech_detected_in_recording = False

        try:
            self._stream = self._sd.InputStream(
                device=MIC_DEVICE,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as error:
            with self._lock:
                self.is_recording = False
                self.is_speaking = False
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self.mic_status = "UNAVAILABLE"
            print(f"无法开始麦克风录音，视觉系统继续运行：{error}")
            return
        print("录音开始，请说话；再次按 SPACE 结束。")

    def stop_recording(self):
        with self._lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self.is_speaking = False
            chunks = self._chunks
            self._chunks = []
            speech_detected = self._speech_detected_in_recording
            self._vad.reset()

        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as error:
                print(f"关闭麦克风时发生警告：{error}")

        if not chunks:
            print("录音为空，已忽略。")
            return

        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        duration = audio.size / float(SAMPLE_RATE)
        if duration < MIN_RECORDING_SECONDS:
            print(f"录音过短（{duration:.2f} 秒），已忽略。")
            return
        if not speech_detected:
            print("录音中没有检测到稳定语音，已忽略。")
            return

        self.mic_status = "TRANSCRIBING"
        # 提交时间用于评估 STT 队列等待 + 实际识别的总耗时。
        self._jobs.put((audio, time.monotonic()))
        print(f"录音结束（{duration:.1f} 秒），正在后台识别……")

    def update(self):
        """由视觉主循环调用，只处理轻量状态，不执行 STT。"""
        if self._auto_stop_requested:
            self._auto_stop_requested = False
            self.stop_recording()
            return True
        return False

    def poll_result(self):
        try:
            outcome = self._results.get_nowait()
        except queue.Empty:
            return None

        self.mic_status = "READY" if self._sd is not None else "UNAVAILABLE"
        if outcome.result is not None:
            self.last_speech = outcome.result.text
        return outcome

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # 回调线程不抛异常，短暂设备警告不会终止视觉系统。
            pass

        samples = np.asarray(indata[:, 0], dtype=np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        with self._lock:
            self.current_rms = rms
            if self.input_suppressed or not self.is_recording:
                self.is_speaking = False
                return
            self._chunks.append(samples.copy())
            self.is_speaking = self._vad.update(rms)
            self._speech_detected_in_recording |= self.is_speaking
            if time.monotonic() - self._recording_started >= MAX_RECORDING_SECONDS:
                self._auto_stop_requested = True

    def set_input_suppressed(self, suppressed):
        """Half-Duplex：NPC 发声时丢弃输入，且不允许提交 STT。"""
        suppressed = bool(suppressed)
        if suppressed and self.is_recording:
            self._discard_recording()
        self.input_suppressed = suppressed

    def _discard_recording(self):
        """关闭当前输入流并丢弃内存录音，不进入 STT 队列。"""
        with self._lock:
            self.is_recording = False
            self.is_speaking = False
            self._chunks = []
            self._speech_detected_in_recording = False
            self._vad.reset()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _stt_worker(self):
        recognizer = LocalSpeechRecognizer()
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            audio, started = job
            try:
                result = recognizer.transcribe(audio)
                latency_ms = int((time.monotonic() - started) * 1000)
                self._results.put(
                    RecognitionOutcome(result=result, latency_ms=latency_ms)
                )
            except Exception as error:
                latency_ms = int((time.monotonic() - started) * 1000)
                self._results.put(
                    RecognitionOutcome(error=str(error), latency_ms=latency_ms)
                )

    def close(self):
        self._stop_event.set()
        self._jobs.put(None)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._sd = None
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self.mic_status = "DISABLED"
        self.is_speaking = False
