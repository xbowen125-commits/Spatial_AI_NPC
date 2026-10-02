"""Voice Input / Output 的集中配置。"""

# None 表示使用 Windows 当前默认输入设备，也可以填写 sounddevice 设备编号。
MIC_DEVICE = None
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

# RMS 迟滞：音量超过开始阈值一段时间才进入 speaking，低于停止阈值后延迟退出。
VAD_START_THRESHOLD = 0.025
VAD_STOP_THRESHOLD = 0.015
VAD_HANGOVER = 0.60
VAD_MIN_SPEECH_DURATION = 0.15

# Push-To-Talk 录音参数。
PUSH_TO_TALK = True
MIN_RECORDING_SECONDS = 0.40
MAX_RECORDING_SECONDS = 30.0

# faster-whisper 本地模型参数。首次使用时需先准备/下载模型文件。
STT_MODEL = "base"
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"

# 隐私默认值：不保存录音。当前实现直接在内存中完成识别。
DEBUG_SAVE_AUDIO = False
LAST_SPEECH_DISPLAY_LENGTH = 32

# Windows SAPI 本地 TTS。Voice ID 为 None 时按回复语言自动选择系统 Voice。
TTS_ENABLED = True
TTS_RATE = 175
TTS_VOLUME = 1.0
TTS_VOICE_ID = None

# 默认半双工：NPC 发声期间禁止录音和 STT 提交，避免听见自己的扬声器。
HALF_DUPLEX_MODE = True
RESPOND_TO_UNKNOWN_DIRECTED_SPEECH = True

# Unity 字幕在 NPC 说完后继续保留的秒数。
SUBTITLE_DURATION = 1.0

# 隐私默认值：TTS 直接播放，不保存 wav。
DEBUG_SAVE_TTS_AUDIO = False
