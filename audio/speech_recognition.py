"""faster-whisper 本地语音识别封装。"""

from dataclasses import dataclass

import numpy as np

from .audio_config import STT_COMPUTE_TYPE, STT_DEVICE, STT_MODEL


@dataclass(frozen=True)
class SpeechResult:
    text: str
    language: str
    confidence: float | None = None


class LocalSpeechRecognizer:
    """延迟加载模型；调用方应在后台线程中执行 transcribe。"""

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "未安装 faster-whisper，请先运行：python -m pip install faster-whisper"
            ) from error

        self._model = WhisperModel(
            STT_MODEL,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
        )

    def transcribe(self, audio_samples):
        """识别内存中的 float32 单声道音频，不创建永久录音文件。"""
        samples = np.asarray(audio_samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise ValueError("录音为空。")

        self._load_model()
        segments, info = self._model.transcribe(
            samples,
            beam_size=5,
            vad_filter=False,
        )
        text = "".join(segment.text for segment in segments).strip()
        if not text:
            raise ValueError("没有识别到清晰语音。")

        # faster-whisper 没有可靠统一的句级置信度，因此明确返回 None。
        language = getattr(info, "language", None) or "unknown"
        return SpeechResult(text=text, language=language, confidence=None)
