# Spatial_AI_NPC

面向 Unity 与未来 VR/XR 场景的多模态具身 AI NPC 系统。当前版本为 **AI NPC v1.1 — Unity Preview**。

项目组合了 Vision / Spatial Perception、Voice STT/TTS、Multimodal Agent、Proactive Behavior、Relationship Memory、Emotion System，以及 Unity / VRM Avatar 集成。默认优先本地运行，不保存摄像头画面或音频。

当前已经实机验证：

    Camera
      ↓
    Python Spatial Perception
      ↓ UDP :5052
    Unity
      ↓
    VRM Airi Avatar
      ↓
    Breathing Idle + Head/Eye LookAt

> 仓库只包含代码。开发时使用的 VRM Avatar、Mixamo FBX、UniVRM 二进制包和模型权重不随仓库发布，详见 [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md)。

## 当前功能

- OpenCV + MediaPipe 摄像头、人体姿态、人脸、头部方向和粗略视线感知
- 玩家位置、相对距离、举手、注意力、眼神接触和 directed speech 状态
- 本地 faster-whisper STT、Windows SAPI TTS、Push-To-Talk、VAD 和半双工控制
- OpenAI-compatible Agent、结构化决策、白名单动作和规则 Fallback
- 主动事件、行为优先级、冷却与 NPC Behavior State Machine
- 轻量关系记忆、查看/导出/清除控制和隐私 Allowlist
- 短期 Emotion State、事件调制、混合、衰减和 Evaluation 日志
- Python → Unity UDP、VRM Humanoid Idle、Head/Eye LookAt、Wave/Nod/字幕接口

## 路线进度

- [x] Visual Perception
- [x] Eye Contact
- [x] Voice Input
- [x] STT
- [x] TTS
- [x] Speech Event
- [x] Rule Response
- [x] LLM Provider
- [x] NPC Personality
- [x] World Context
- [x] Short-term Conversation
- [x] Structured Agent Decision
- [x] NPC Action
- [x] NPC Emotion
- [x] Rule Fallback
- [x] Evaluation Logger
- [x] Metrics
- [x] Offline Replay
- [x] World Event Detector
- [x] Proactive Behavior Rules
- [x] Behavior State Machine
- [x] Behavior Cooldown
- [x] Relationship Memory
- [x] Player Profile Persistence
- [x] Memory Show / Export / Clear
- [x] Short-term Emotion State
- [x] Emotion Decay / Blending
- [x] Long-term Memory
- [x] Proactive NPC
- [ ] Advanced Emotion
- [ ] Lip Sync
- [ ] VR Integration

## 架构

              Camera
                 ↓
          Visual Perception
                 ↓
            Player State
                 ↓
           World Context ──────┐
                               │
    Microphone                 │
        ↓                      │
       STT                     │
        ↓                      │
    Speech Event ──────────────┤
                               ↓
                         NPC Agent
                       ↙    ↓     ↘
                   Reply  Action  Emotion
                     ↓      ↓       ↓
                    TTS   Unity   NPC State
                     ↓
                   Voice

视觉参数集中在 vision/perception_config.py，音频参数集中在 audio/audio_config.py。

## 目录结构

    Spatial_AI_NPC/
    ├── vision/       # 摄像头、Pose、Head Pose、Gaze、Player State、Unity桥接
    ├── audio/        # 麦克风、VAD、STT、TTS与播放队列
    ├── npc/          # Agent、Personality、World Context与结构化决策
    ├── behavior/     # World Event检测、规则、优先级、冷却与状态机
    ├── emotion/      # 短期Emotion State、Policy、Decay与Blend
    ├── memory/       # 关系档案、Memory Policy与CLI控制
    ├── evaluation/   # JSONL Logger、Replay与Metrics
    ├── events/       # Speech、World、NPC State等事件Schema
    ├── unity/        # 可复制到Unity Assets的C#脚本
    └── README.md

`logs/`、`memory/player_profile.json`、`memory/exports/`、缓存和模型文件均为本地运行数据，不进入版本控制。

## 环境要求

- Windows 10/11
- Python 3.11（推荐使用 Conda 环境 `vision_ai`）
- 摄像头；语音功能另需麦克风和系统扬声器
- Unity `6000.3.25f1` 或兼容版本
- Humanoid Avatar；VRM 1.0 集成已在 UniVRM `v0.131.2` 验证
- 可选：本地 Ollama 或其他 OpenAI-compatible Chat Completions 服务

## 安装

    conda create -n vision_ai python=3.11
    conda activate vision_ai
    cd Spatial_AI_NPC
    python -m pip install -r requirements.txt

`faster-whisper` 的模型在首次启用 STT 时单独下载或从本地缓存加载，不应提交到仓库。仅测试视觉时，可以先安装 OpenCV、MediaPipe 和 NumPy。

## Player State

    {
      "person_detected": true,
      "face_detected": true,
      "horizontal_position": "center",
      "distance": "medium",
      "left_hand": "down",
      "right_hand": "down",
      "head_direction": "center",
      "head_yaw": 3.2,
      "head_pitch": -1.8,
      "attention_score": 0.86,
      "looking_at_npc": "true",
      "gaze_horizontal": "center",
      "gaze_vertical": "center",
      "gaze_score": 0.91,
      "eye_contact": "true",
      "is_speaking": true,
      "directed_speech": "true"
    }

looking_at_npc、eye_contact 和 directed_speech 使用 true / false / unknown 三态字符串。识别文本不会放进每秒发送的 Player State。

## Head Pose 与 Looking At NPC

系统从 Face Mesh 选取鼻尖、下巴、双眼外角和双嘴角，通过 OpenCV solvePnP 得到粗略头部 yaw 与 pitch。Head Pose 使用系数 0.25 的 EMA。

looking_at_npc 保留 v0.4 的“头部注意力”语义：

- attention_score >= 0.75：进入 true
- attention_score <= 0.55：退出为 false
- 中间区域保持原状态
- 连续3帧确认后切换
- 无有效人脸或 Head Pose 时为 unknown

## 虹膜与视线方向

Face Mesh 使用 refine_landmarks=True，取得 468–477 号虹膜关键点。每只眼睛都在自己的局部坐标中计算：

    horizontal_ratio = (iris_x - eye_left_x) / eye_width
    vertical_ratio = (iris_y - eye_top_y) / eye_height

这样比直接使用整张画面的坐标更能抵抗头部平移。左右眼结果按可靠度融合；只检测到一只眼时允许降级使用，双眼严重冲突时输出 unknown。

默认中性点为 (0.5, 0.5)。视线方向阈值：

- 水平偏移 <= -0.10：left
- 水平偏移 >= +0.10：right
- 垂直偏移 <= -0.08：up
- 垂直偏移 >= +0.08：down
- 阈值之间：center

虹膜比例使用系数 0.35 的 EMA，gaze_score 使用系数 0.30 的 EMA；离散视线状态连续3帧一致后才切换。

不同摄像头镜像方式可能造成 left/right 相反。若实测方向相反，将 vision/perception_config.py 中的 GAZE_INVERT_HORIZONTAL 改为 True 后重新运行。

## 视线校准

运行时正视摄像头并按 c，保持自然正视直到收集45个有效帧。系统会用均值更新中性视线点：

- DEFAULT：使用默认中性点
- CALIBRATING n/45：正在采集
- CALIBRATED：当前进程已校准

校准数据只保存在内存中，重启程序后恢复默认值，不会保存人脸图片或身份数据。

## Gaze Score 与 Eye Contact

gaze_score 是可解释的启发式分数，不是专业眼动仪测量：

    0.35 × horizontal_center_score
    + 0.30 × vertical_center_score
    + 0.20 × both_eye_consistency
    + 0.15 × landmark_reliability

最终再乘方向门控，避免眼睛明显偏向一侧时仅靠其他项取得高分。

eye_contact 同时依赖头部注意力和眼睛视线：

    combined = 0.45 × attention_score + 0.55 × gaze_score

- looking_at_npc=true 且 combined >= 0.80：进入 true
- looking_at_npc=true 且 combined <= 0.60：退出为 false
- 0.60 到 0.80：保持原状态（迟滞）
- 连续3帧确认后切换
- 视线信息无效：unknown
- 头部已明确没有朝向 NPC：false

## Microphone、Push-To-Talk 与 VAD

音频输入使用 sounddevice，默认读取 Windows 默认麦克风：

- MIC_DEVICE = None：系统默认输入设备，也可改成 microphone_test.py 显示的设备编号
- SAMPLE_RATE = 16000
- CHANNELS = 1
- BLOCK_SIZE = 1024
- 按 SPACE 开始录音，再按一次 SPACE 结束并提交识别
- 默认不会永久监听；麦克风输入流只在一次 Push-To-Talk 录音期间打开
- 最短录音0.40秒，最长30秒
- 音频只保存在内存中，DEBUG_SAVE_AUDIO 默认为 False

VAD 使用 RMS 音量和迟滞，不使用神经网络：

- RMS >= 0.025 持续0.15秒：进入 Speaking
- Speaking 后以0.015为停止阈值
- 低音量持续0.60秒后退出，避免底噪附近快速闪烁

VAD 只负责 is_speaking；录音起止仍由 Push-To-Talk 控制。

## Local Speech-to-Text

STT 使用 faster-whisper 的多语言 base 模型，默认配置：

    STT_MODEL = "base"
    STT_DEVICE = "cpu"
    STT_COMPUTE_TYPE = "int8"

它支持中文、英文和常见中英混说，推理完全在本机执行，不调用云端 STT API。第一次使用模型名称时，faster-whisper 可能需要联网下载模型；之后可从本地缓存加载，也可以把 STT_MODEL 改成本地模型目录。

当前 vision_ai 环境已有 sounddevice。语音输入和输出需要手动安装：

    python -m pip install faster-whisper pyttsx3

模型在独立后台线程中加载和推理。录音、模型下载或识别失败只会产生错误提示，不会终止 Camera、Pose、Face、Gaze 或 UDP 心跳。

## Rule-based Response

npc/response_engine.py 不使用 LLM，只处理少量可解释规则，并根据 Speech Event 的 language 选择中文或英文：

- 你好 / hello / hi → 你好。/ Hello.
- 你能看到我吗 / can you see me → 根据 person_detected 回答
- 你在看我吗 / are you looking at me → 根据 eye_contact 回答
- 再见 / goodbye / bye → 再见。/ Goodbye.
- 其他内容 → 当前无法回答的固定提示

directed_speech=false 时不回应；unknown 是否回应由 RESPOND_TO_UNKNOWN_DIRECTED_SPEECH 控制，默认 True。

## Local TTS 与 Queue

TTS 使用 pyttsx3 调用 Windows SAPI5，文本不会上传云端，也不会保存 wav：

    TTS_ENABLED = True
    TTS_RATE = 175
    TTS_VOLUME = 1.0
    TTS_VOICE_ID = None
    HALF_DUPLEX_MODE = True
    RESPOND_TO_UNKNOWN_DIRECTED_SPEECH = True
    SUBTITLE_DURATION = 1.0
    DEBUG_SAVE_TTS_AUDIO = False

TTS_VOICE_ID=None 时按 language 尝试选择中文或英文系统 Voice。Windows 没有中文 Voice 时输出 No Chinese system voice found，并安全使用默认 Voice；可在系统语言设置中安装中文语音包，或将 TTS_VOICE_ID 设置为 tts_test.py 列出的 Voice ID。

SpeechOutputController 使用一个后台工作线程和 FIFO 队列。一句话播放期间，后续回复依次等待，不会重叠播放。TTS 初始化或播放失败时，文字回复仍会发送到 Unity。

单独测试 TTS：

    python audio/tts_test.py

测试脚本会列出 Engine、Voice、Rate、Volume，并依次尝试中文和英文句子。

## Half-Duplex 与自语音抑制

默认流程为：玩家录音 → STT → NPC 回复 → TTS 结束 → 重新允许玩家录音。

NPC 发声期间：

- npc_is_speaking=true
- 新的 Push-To-Talk 请求被拒绝
- 如果输入流意外仍在运行，录音块会被丢弃
- 不会提交 VAD / STT 任务
- Camera、MediaPipe 和视觉 UDP 心跳继续运行

这是一种简单的 Self-Speech Suppression，不是 Acoustic Echo Cancellation。未来全双工需要真正的 AEC。

## Directed Speech

directed_speech 是“玩家可能在对 NPC 讲话”的简单近似，并不理解语言意图：

- 未说话：false
- 说话 + eye_contact=true：true
- 说话 + eye_contact=false 或 looking_at_npc=false：false
- 正在说话，但玩家或虹膜信息不足：unknown

## Speech Event

识别完成后单独发送一次事件，不把长文本塞进 Player State：

    {
      "message_type": "interaction_event",
      "event_type": "speech",
      "timestamp": 1234567890000,
      "text": "你好，你能看到我吗",
      "language": "zh",
      "confidence": null,
      "context": {
        "person_detected": true,
        "face_detected": true,
        "horizontal_position": "center",
        "eye_contact": "true",
        "looking_at_npc": "true",
        "distance": "medium",
        "left_hand": "down",
        "right_hand": "down",
        "directed_speech": "true"
      }
    }

Player State 数据包使用 message_type=player_state。Unity 对没有 message_type 的旧数据包仍按 Player State 解析。

Agent 或规则 Fallback 产生的 Response Event：

    {
      "message_type": "interaction_event",
      "event_type": "npc_response",
      "timestamp": 1234567890001,
      "text": "能，我看到你了。",
      "language": "zh",
      "source_event": "speech",
      "subtitle_duration": 1.0
    }

Agent 思考、TTS 开始和结束时发送独立 NPC State：

    {
      "message_type": "npc_state",
      "npc_is_thinking": false,
      "npc_is_speaking": true,
      "npc_emotion": "happy"
    }

## Multimodal NPC Agent

当前 Multimodal 使用 Perception-first 架构：摄像头图像只进入本地 CV，LLM 只接收经过整理的语义状态，不直接接收图像。

模块职责：

- npc/agent.py：异步队列、Provider 调用、验证和规则降级
- npc/llm_provider.py：统一 LLMProvider 接口与 OpenAI-compatible HTTP 实现
- npc/personality.py：Airi 人格和 System Prompt
- npc/world_context.py：把 Player State 转为权威语义事实
- npc/conversation_context.py：内存中的最近6轮对话
- npc/action_schema.py：JSON解析、动作/情绪白名单和感知冲突检查
- npc/response_engine.py：保留原有规则 Fallback

### Agent 配置

配置集中在 npc/agent_config.py。默认 LLM_ENABLED=False，因此不配置模型也能继续使用规则回复。

PowerShell 配置 OpenAI-compatible 服务示例：

    $env:SPATIAL_NPC_LLM_ENABLED = "true"
    $env:SPATIAL_NPC_LLM_PROVIDER = "openai_compatible"
    $env:SPATIAL_NPC_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
    $env:SPATIAL_NPC_LLM_MODEL = "your-model-name"

远程服务的密钥只通过环境变量提供：

    $env:SPATIAL_NPC_API_KEY = "<set-your-key-locally>"

代码不会硬编码或打印 API Key。localhost 服务允许不设置 Key；若远程地址缺少配置的 Key，自动 Rule Fallback。

其他参数：LLM_TIMEOUT=20秒、MAX_CONVERSATION_TURNS=6、AGENT_QUEUE_SIZE=4、DEBUG_AGENT=True。Python 退出后短期Conversation立即消失，不使用数据库、Embedding或向量存储。

### Personality 与输入

默认NPC为 Airi：温和、好奇、自然、简洁，通常回答1到3句话，不机械复述玩家，不声称不存在的能力。

每次调用包含：

1. 独立 System Prompt / Personality
2. AUTHORITATIVE PERCEPTION CONTEXT
3. 最近6轮 Conversation
4. 当前玩家 Speech，始终作为 user message

玩家文本绝不会拼入 System Prompt。提示词明确要求忽略玩家要求绕过人格、感知事实、JSON Schema或动作白名单的指令。

### World Context

原始状态会转换成类似：

    Player is visible.
    Player distance is near.
    Player is making eye contact.
    Player has their left hand raised.
    Player's right hand is down.

unknown 会被表达为 uncertain，不会假装已知。

### Structured Decision

LLM只能返回：

    {
      "reply": "你好呀。",
      "action": "wave",
      "emotion": "happy",
      "source": "llm"
    }

动作白名单：none、wave、nod、look_at_player。

情绪白名单：neutral、happy、curious、surprised。

Python 能解析纯JSON、Markdown json代码块和JSON前后的额外文字。非法 action 自动变为 none，非法 emotion 自动变为 neutral；缺失/空 reply、非法JSON、timeout、网络错误或Provider异常都会触发规则 Fallback。

对于 person_detected=false、eye_contact不确定或手部状态未知等情况，Python还会检查常见矛盾声明；检测到“没有玩家却声称看见玩家”等冲突时直接 Fallback。该检查是保守安全层，不等同于完整自然语言事实验证。

### Agent 生命周期与事件

Agent在独立线程执行，不阻塞 Camera、MediaPipe、STT、TTS或UDP。只有 speech Interaction Event 会提交Agent；Player State心跳和视觉状态变化不会触发调用。

Agent开始/结束会更新：

    {
      "message_type": "npc_state",
      "npc_is_thinking": false,
      "npc_is_speaking": false,
      "npc_emotion": "neutral"
    }

验证后的动作单独发送：

    {
      "message_type": "interaction_event",
      "event_type": "npc_action",
      "timestamp": 1234567890002,
      "action": "wave",
      "emotion": "happy"
    }

reply继续使用 npc_response → TTS → Subtitle，不把动作塞进TTS。

### Agent 独立测试

交互测试：

    python npc/agent_test.py

自动测试JSON、非法action、感知冲突、timeout和LLM disabled fallback：

    python npc/agent_test.py --self-test

## Evaluation Pipeline

本阶段只评估当前能力，不增加新的 NPC 行为：

              Camera
                 ↓
             Perception
                 ↓
               Agent
                 ↓
              Logger
                 ↓
              Metrics

完整交互默认追加到 `logs/interactions.jsonl`。每行是一个独立 JSON 对象，包含：

- speech_text、speech_language 与当时的语义 speech_context
- world_context
- agent_source、reply、action、emotion 与 fallback_reason
- stt_latency、agent_latency、tts_latency、total_latency，单位均为毫秒
- action_error_count 与 fact_violation_count

日志不会保存麦克风音频、人脸、摄像头图片、API Key 或 Authorization Header。日志仍包含玩家识别文本，因此分享或提交日志前应先检查文本隐私。

统计默认日志：

    python evaluation/metrics.py

也可以指定其他 JSONL：

    python evaluation/metrics.py logs/replay_interactions.jsonl

输出包括交互总数、LLM成功率、Fallback率、各阶段平均延迟、非法动作数和感知事实冲突数。

运行 A～J 固定案例，不需要摄像头、麦克风、TTS 或 Unity：

    python evaluation/replay.py

默认结果写入 `logs/replay_interactions.jsonl`。若配置了 LLM，A～G 会使用当前 Provider；未配置时安全使用规则 Fallback。H 强制关闭 LLM，I 使用模拟非法 action，J 使用模拟非法 JSON。

重放历史交互中的 Speech Event 与语义上下文：

    python evaluation/replay.py logs/interactions.jsonl --output logs/history_replay.jsonl

Replay 只重新运行 Agent，不播放 TTS、不发送 Unity 消息，也不读取摄像头和麦克风。历史原始日志不会被修改。

## Reactive NPC → Proactive NPC

v0.8 的 Reactive NPC 只在玩家产生 Speech Event 后回答。v0.9 在旁边增加独立的纯规则主动行为层：

           Player State
                ↓
          Event Detector
                ↓
           World Event
                ↓
        Behavior Manager
                ↓
        Interaction Event
             ↙     ↘
           TTS     Unity
            ↓        ↓
          Voice    Action

Event Detector 每帧只更新计数器；只有稳定事件成立时 Behavior Manager 才决策，不会每帧调用规则，更不会调用 LLM。

World Event 使用统一格式，并只保存触发所需的 Player State 摘要：

    {
      "message_type": "world_event",
      "event_type": "hand_wave",
      "timestamp": 123456789,
      "priority": "medium",
      "context": {
        "left_hand": "raised",
        "right_hand": "down",
        "eye_contact": "true"
      }
    }

当前支持：player_enter、player_leave、eye_contact_started、eye_contact_long、hand_wave、player_approach、player_far。

稳定确认参数：

- player_enter：person_detected 连续3帧为 true
- player_leave：已确认玩家后，person_detected 持续5秒为 false
- eye_contact_started：连续3帧为 true
- eye_contact_long：稳定对视持续10秒，每次对视过程只触发一次
- hand_wave：单手从 down 到 raised，连续3帧确认
- player_approach：稳定距离从 far 到 near，near 连续4帧确认
- player_far：稳定距离从 near/medium 到 far，far 连续4帧确认

规则输出统一为 `npc_behavior` Decision，然后由主编排转换为原有 `npc_response` 与 `npc_action`：

    {
      "message_type": "interaction_event",
      "event_type": "npc_behavior",
      "reply": "欢迎回来。",
      "action": "wave",
      "emotion": "happy",
      "source": "behavior_rule"
    }

规则：player_enter → 欢迎回来 + wave；player_leave → 再见；eye_contact_long → 询问有什么想问；hand_wave → 你好 + wave；player_approach → look_at_player。eye_contact_started 与 player_far 当前只形成事件，暂不产生动作。

事件优先级：player_enter/player_leave 为 high；hand_wave/player_approach/player_far 为 medium；eye_contact_started/eye_contact_long 为 low。同一帧只执行最高优先级。player_enter 执行时同时启动 hand_wave 冷却，避免“欢迎 + 挥手问候”重复两次。

事件冷却：player_enter 60秒、eye contact 30秒、hand_wave 5秒。每个事件本身也是边沿触发，稳定状态保持不变时不会重复产生事件。

### NPC Behavior State Machine

`npc_behavior_state` 包含 idle、observing、greeting、listening、thinking、speaking、cooldown，并作为现有 NPC State 的附加字段发送：

    {
      "message_type": "npc_state",
      "npc_is_thinking": false,
      "npc_is_speaking": true,
      "npc_emotion": "happy",
      "npc_behavior_state": "greeting"
    }

玩家 Speech Event 优先级最高：它会清除尚未执行的主动事件，并进入 listening → thinking → speaking。NPC 正在 greeting、thinking 或 speaking 时，主动事件最多只保留一个最高优先级等待项；不会打断当前回答。TTS 完成后经过0.5秒 cooldown，再回到 observing 或 idle，并处理仍有效的等待事件。

主动行为仍通过既有 `npc_response` 与 `npc_action` 进入 Unity；Wave 和 LookAt 继续由现有 NpcActionController 执行。v1.1 仅额外提供可选的 NpcEmotionController，用于把内部情绪映射到 Animator 参数。

主动行为日志默认写入 `logs/behaviors.jsonl`，记录 event_type、priority、Decision、action、reply、cooldown_status、剩余冷却时间和 npc_behavior_state。它不保存音频或图像。

离线测试：

    python behavior/behavior_test.py

测试覆盖玩家进入、持续存在不重复、离开5秒、长时间注视、挥手、靠近、冷却、同帧优先级、忙碌等待和 Speech Event 取消。

Behavior Manager 只依赖 `rules.decide(world_event)` 接口。未来可以将 BehaviorRules 替换为 LLM Behavior Planner，而不改变 World Event、Behavior Decision、TTS 或 Unity Action 接口；v0.9 当前不会从主动行为调用 LLM。

## NPC Relationship Memory

v1.0 增加轻量、结构化、可审查的长期关系状态。它不是完整聊天记忆，不使用数据库、Embedding 或向量存储。

默认档案位置：

    memory/player_profile.json

档案结构：

    {
      "player_id": "local_player",
      "first_seen": 123456789,
      "interaction_count": 3,
      "times_seen": 2,
      "times_spoken": 3,
      "times_waved": 1,
      "relationship_level": "acquaintance",
      "last_interaction": 123456999,
      "preferences": {}
    }

只保存结构化计数、时间戳、关系等级，以及玩家明确确认后允许保存的简单偏好。不会保存音频、图片、人脸、生物信息、聊天记录或 Speech Text。

所有写入先形成 Memory Candidate，再由 Memory Policy 判断：

    {
      "type": "interaction",
      "event": "first_meeting",
      "timestamp": 123456789,
      "data": {}
    }

默认自动保存：first_meeting、interaction_count、times_seen、player_waved。`interaction_count` 可以同时标记本次是否为 spoken，从而更新 times_spoken。未知 Candidate 默认拒绝。

`player_preference` 必须由调用方明确传入 `confirmed=True`；未确认时不会写入。偏好仅允许200字符以内的短文本或简单数值，不允许自动从聊天中推断。

关系升级参数集中在 `memory/memory_policy.py`：

- stranger → acquaintance：times_seen ≥ 2，或 times_spoken ≥ 3，或 interaction_count ≥ 3
- acquaintance → friend：times_seen ≥ 5，并且 times_spoken ≥ 5，并且 interaction_count ≥ 10

关系只会根据计数升级，不分析语音内容。Player Profile 使用 UTF-8 JSON 原子替换写入；程序重启后会重新加载。JSON 损坏时，原文件会复制为 `player_profile.corrupt-时间戳.json`，系统创建干净档案继续运行。

Behavior Integration：首次确认 `player_enter` 时回复“你好，第一次见面。”；已有 first_seen 时继续使用“欢迎回来。”。RelationshipBehaviorRules 通过原有 `rules.decide(world_event)` 接口注入，没有改变 BehaviorManager。

Agent Integration：Speech Context 增加 relationship_level，World Context 会包含：

    Player relationship is friend.

关系信息只是附加语义，不能覆盖 person_detected、eye_contact、hand state 等权威实时感知事实。

离线测试：

    python memory/memory_test.py

覆盖首次进入、重复进入、挥手累计、关系升级、重启加载、损坏JSON恢复、偏好确认、隐私Schema与Agent World Context。

## Memory Controls

请在项目根目录运行以下命令。为避免正在运行的 NPC 进程稍后把内存中的旧档案重新写回，执行 `clear` 前应先关闭 `vision/unity_bridge.py`。

查看当前档案：

    python memory/memory_cli.py show

输出只包含 Player Profile Allowlist：player_id、first_seen、interaction_count、times_seen、times_spoken、times_waved、relationship_level、last_interaction、preferences。档案尚不存在时显示一份未见过玩家的默认 Profile，不会创建文件。

导出当前档案：

    python memory/memory_cli.py export

默认导出到：

    memory/exports/player_profile_<UTC timestamp>.json

导出文件只包含同一组 Allowlist 字段，不会读取或复制 Speech Text、Conversation Context、Audio、Image、Face Data、API Key 或 Evaluation Log。

清除当前档案：

    python memory/memory_cli.py clear

CLI 会要求二次确认：

    This will permanently clear the local NPC relationship profile.
    Type CLEAR to continue:

只有准确输入大写 `CLEAR` 才会删除。用于自动测试的无交互形式：

    python memory/memory_cli.py clear --yes

清除会删除当前 `player_profile.json`、临时写入文件和损坏档案备份，但保留用户主动创建的 exports，也不会修改 Evaluation 日志。下一次确认 player_enter 时会重新执行“第一次见面”流程。

## NPC Internal State

三类状态严格分离：Memory 表示过去发生过什么，长期持久；Relationship 表示 NPC 与玩家的关系，长期缓慢变化；Emotion 表示 NPC 此刻的短期内部状态，只存在于当前进程并自动衰减。

              World Event
                   |
                   v
             Emotion Engine
                   |
                   v
             Emotion State
              /         \
             v           v
        NPC Agent     Behavior
             \           /
              \         /
               v       v
               NPC Decision
                    |
             +------+------+
             |             |
             v             v
            TTS          Unity

Emotion State：

    {
      "emotion": "happy",
      "valence": 0.45,
      "arousal": 0.35,
      "intensity": 0.42,
      "cause": "player_enter",
      "updated_at": 123456789
    }

允许的标签只有 neutral、happy、curious、surprised。valence Clamp 在 -1～1；arousal 与 intensity Clamp 在 0～1。当前不包含 angry、sad、fear、love 或 jealousy。

外部事件策略集中在 `emotion/emotion_policy.py`：player_enter → happy(0.45)；hand_wave → happy(0.60)；eye_contact_started → curious(0.35)；eye_contact_long → curious(0.50)；player_approach → curious(0.30)。括号内为基础 intensity。

Relationship 只轻微乘算 intensity：stranger 0.85、acquaintance 1.0、friend 1.15，最终仍 Clamp 到1。关系不会直接指定情绪，也不会让 friend 永久保持 happy。

已有活跃情绪时，新事件使用 old 0.35 + new 0.65 混合 valence、arousal 与 intensity；标签优先采用新事件。没有新事件时每秒更新一次指数衰减，30秒后约保留5%；intensity < 0.10 时精确恢复 neutral、cause=none 和全零数值。

Speech Event 不分析文字或玩家心理，只把 arousal 增加0.05。Emotion 只响应 World Event 与 Speech Event，不会生成新的 World Event，因此不存在 Emotion → Behavior → Emotion 反馈循环。

Agent World Context 只接收当前标签和强度，例如：

    NPC current emotion is happy.
    NPC emotion intensity is 0.42.

System Prompt 只允许 Emotion 轻微影响语气和措辞，不得覆盖 Perception、Relationship、事实或制造事件。Behavior 只读当前 Emotion：例如 curious + eye_contact_long 可附加 look_at_player；读取过程不会更新 Emotion。

NPC State 新增向后兼容字段：

    {
      "npc_is_thinking": false,
      "npc_is_speaking": false,
      "npc_behavior_state": "idle",
      "npc_emotion": "happy",
      "npc_emotion_intensity": 0.42
    }

内部每秒衰减，但只有标签变化、强度相对上次发送变化至少0.05、其他 NPC State 改变或5秒心跳时才发送，避免逐帧 UDP 消息。

Emotion 日志默认写入 `logs/emotions.jsonl`，只包含 timestamp、trigger_event、前后标签、前后强度和 relationship_level，不保存 Speech Text。

测试：

    python emotion/emotion_test.py

Emotion 不写入 `memory/player_profile.json`；重启程序后始终从 neutral 开始。

## Python 调试画面

显示原有视觉状态，以及 Mic、Speaking、Directed Speech 和截断后的 Last Speech。完整识别文本只在终端和一次性事件中输出。

## 运行

主要入口：

    python vision/unity_bridge.py

也可以先进入项目目录，再使用当前 Conda 环境的 Python，避免在脚本或文档中保存个人绝对路径。

- 按 c：开始当前进程的视线校准
- 按 SPACE：开始/结束一次 Push-To-Talk 录音
- 按 q：退出

单独测试麦克风设备和音量：

    python audio/microphone_test.py

该脚本会枚举设备、显示默认输入设备和实时 RMS；按 q 或 Ctrl+C 退出，不保存录音。

## Unity 配置

将 unity 目录中的六个脚本复制或替换到 Unity Assets，并把 Voice、Action 与可选 Emotion Controller 挂到 NPC：

- NpcPerceptionReceiver.cs
- NpcReactionController.cs
- NpcLookAtController.cs
- NpcVoiceInteractionController.cs
- NpcActionController.cs
- NpcEmotionController.cs

配置要点：

1. UDP 端口保持5052，超时保持2秒。
2. Reaction 与 LookAt 控制器连接同一个 Receiver。
3. Animator 保留 Wave Trigger；可选添加 AttentionDetected 和 EyeContactDetected Trigger。
4. Humanoid 自动查找 Head；`autoFindEyeBones` 开启时可自动取得 LeftEye/RightEye，没有眼睛骨骼时安全跳过。
5. 当前实机配置中 Head 侧向目标为22°；眼睛使用 Head 目标的25%，侧向辅助约不超过6°。
6. 头部跟随速度为6；玩家大幅偏离并持续约0.6秒后，身体才逐渐参与转向。
7. Voice Controller 连接同一个 Receiver；可选在 Animator 中添加 Listening Bool。
8. 可选在 Animator 中添加 Speaking Bool，用于 Talking Idle 动画。
9. 可选将 Unity UI Text 或 TextMeshProUGUI 拖到 Npc Subtitle Text；没有 UI 时安全跳过。
10. NpcActionController 连接同一个 Receiver；可选添加 Thinking Bool、Nod 和 LookAtPlayer Trigger。
11. NpcEmotionController 连接同一个 Receiver；Animator 可选添加 Int 参数 EmotionState 和 Float 参数 EmotionIntensity。参数不存在时安全跳过。

Unity Receiver 区分 player_state、interaction_event 与 npc_state。NpcActionController 在 Unity 端再次执行动作白名单，只允许 none、wave、nod、look_at_player。非法动作只警告并忽略。Thinking、Speaking 和 Emotion 只在状态变化时输出；没有动画参数或字幕组件时安全跳过。

### Humanoid Idle 与 Head LookAt

NpcLookAtController 在 Animator 和普通 LateUpdate 完成后运行。启用 Humanoid Idle 时，它每帧读取 Animator 当前生成的 Head 世界旋转，再沿角色根节点的 Up/Right 轴叠加已经平滑和限幅的 LookAt yaw/pitch；不会使用静态启动姿态覆盖 Idle，也不会把上一帧 LookAt 继续累积。没有 Animator Controller 时自动从启动局部姿态重建世界姿态，旧场景仍可使用。

VRMInstance 使用 Late Update 时无需手工调整 Project Settings 中的 Script Execution Order：当前 UniVRM 的 Vrm10Instance/FastSpringBone 顺序为 11000/11010，NpcLookAtController 使用 12000，确保最终头骨偏移在它们之后应用。若项目中另有自定义脚本被显式设置到更晚的执行顺序并再次写 Head，才需要把 NpcLookAtController 排在该脚本之后。`debugLookAt` 默认关闭；开启后只进行低频输出，不会每帧刷屏。

Airi 使用角色根节点的 Up/Right 世界轴叠加 Head yaw/pitch，因此不依赖 VRoid Head 骨骼的局部轴；Root Y=180 已自然包含在角色坐标中。默认侧向 Head 目标为22度。Humanoid Eye Bone 可自动取得，但只使用 Head 目标的25%，并限制在较小角度；关闭 `autoFindEyeBones` 后，Inspector 中 Eye Bone=None 就表示完全不控制眼睛。`debugLookAt` 开启时最多每秒输出一次平滑 yaw/pitch 与实际 worldYaw。

Head Pose 的3D人脸模型使用OpenCV相机坐标（X右、Y下、Z远离相机），正视时 pitch/yaw 应接近0度。旧版Y/Z轴相反会让正脸被 solvePnP 表示成约正负180度；该坐标问题已在进入EMA与 head_direction 阈值前修正，原有方向阈值保持不变。

## 建议测试

1. 运行 agent_test.py --self-test，确认解析、白名单、感知守卫和Fallback通过。
2. 保持 LLM_ENABLED=False 输入“你好”，确认使用 RuleBasedResponseEngine。
3. 配置兼容模型后输入“你好”，检查 source=llm、Thinking和一次Agent调用。
4. person_detected=true/false 时分别询问“你能看到我吗”，检查回答不违背事实。
5. left_hand=raised 时询问举哪只手；状态unknown时确认NPC承认不确定。
6. eye_contact=unknown 时询问是否对视，确认不会瞎猜。
7. 使用模拟非法 action=attack 和 emotion=angry，确认变为 none/neutral。
8. 模拟timeout、非法JSON和空reply，确认 Console 显示 Agent fallback 且仍有规则回复。
9. 看着 NPC 说“你好”，确认听到回复，Unity Thinking、Action、Speaking和字幕按顺序变化。
10. 检查一个Speech Event只产生一次Decision，UDP心跳不调用Agent。

原有音频闭环继续测试：

- 看着 NPC 说“你好”，确认听到“你好。”，Unity Speaking 和字幕正常变化。
- 站在摄像头前与离开画面后分别问“你能看到我吗”。
- 看向别处说话，确认 Directed Speech Gate 仍生效。
- NPC 播放语音时按 SPACE，确认 Half-Duplex 拒绝录音。
- 设置 TTS_ENABLED=False，确认文字决策和Unity Action仍可工作。

## 已知限制

- 这是普通 RGB 摄像头上的粗略虹膜方向估计，不等同于专业眼动仪。
- 眼睛像素过小、眨眼、眼镜反光、强逆光、遮挡和极端侧脸会降低可靠度。
- 局部虹膜比例能减少头部平移影响，但较大的头部旋转仍会产生透视误差。
- 默认校准点未必适合所有人；启动后建议先按 c 校准。
- 摄像头镜像设置不同，左右方向可能需要手动反转。
- 当前只处理一张主要人脸和一个主要人体。
- Unity 眼睛 LookAt 使用离散视线状态，不是真实3D gaze ray。
- RMS VAD 不能可靠区分玩家声音、扬声器声音和其他环境噪声。
- 普通麦克风在风噪、键盘声、回声和远距离下可能误触发或漏检。
- Whisper 自动语言识别通常只给主要语言，中英混说不会逐词标注语言。
- CPU 首次加载 base 模型可能较慢，但不会阻塞视觉线程。
- directed_speech 只是“视觉注意力 + 说话活动”的规则近似，无法证明说话对象就是 NPC。
- pyttsx3 的中文音质与可用 Voice 取决于 Windows 已安装的语言语音包。
- 当前采用 Half-Duplex，NPC 说话时玩家不能打断；尚未实现 AEC 或全双工。
- 规则引擎只理解少量固定短语，不做语义推理。
- Speaking 仅为 Animator Bool，不包含 Lip Sync、音素或 viseme。
- 默认不保存音频，也没有长期语音历史。
- LLM 输出具有随机性；当前只检查 JSON、白名单和少量常见感知矛盾，不是完整事实验证器。
- OpenAI-compatible Provider 当前使用 Chat Completions 风格的 `/chat/completions` 接口；不同兼容服务对模型名、认证和 JSON 遵循程度可能不同。
- 远程 Provider 会接收玩家语音文本、最近6轮对话和语义感知状态；隐私与数据保留策略取决于所选服务。摄像头原始图像不会发送给 LLM。
- Agent 的逐句Conversation仍只保存在内存中；长期部分仅保存关系计数和明确确认的偏好，不保存聊天内容、RAG数据或身份画像。
- 主动行为仍是启发式规则；遮挡、状态误判或短时间反复进出画面可能产生不自然事件。
- 每次只保留一个等待中的主动事件；更低优先级事件会被忽略，不会形成完整事件历史队列。
- player_far 与 eye_contact_started 当前只记录事件，不产生语音或动作。
- Emotion 是简单事件规则和数值衰减，不是心理模型，也不推断玩家的真实情绪。
- Unity Emotion 默认只映射可选 Animator 整数和强度参数；真实表情仍取决于角色动画、BlendShape与美术资源。
- UDP 没有确认和重传，但保留2秒超时恢复。
- Codex 运行环境不能代替实体摄像头、麦克风和扬声器测试，最终测试需在本机 vision_ai 环境完成。

## 下一阶段建议

先通过 `logs/emotions.jsonl` 验证触发频率、30秒衰减和关系倍率是否自然，再根据实际角色资源添加轻量 BlendShape 表情。不要增加玩家情绪识别或复杂心理模型；Lip Sync 仍应保持独立。

## Privacy / Memory Design

- 摄像头帧、脸部图像、麦克风音频和 TTS 音频默认不落盘。
- 长期 Memory 只保存 Allowlist 内的结构化关系计数和用户明确确认的偏好，不保存完整聊天记录或生物信息。
- `memory/player_profile.json`、Memory Export、运行日志和 Evaluation JSONL 都被 `.gitignore` 排除。
- Emotion 是不持久化的短期运行时状态，程序重启后恢复 neutral。
- 启用远程 OpenAI-compatible Provider 时，文本、最近短期对话和语义 World Context 会发送到所配置的服务；其隐私政策由服务提供方决定。
- API Key 只从本地环境变量读取。不要把 `.env`、终端输出或真实 Key 提交到 Git。

## License / Third-party Assets

本项目自有源码采用 [MIT License](LICENSE) 发布。

开发中使用的 Airi VRM Avatar、Mixamo Breathing Idle、UniVRM 包、语音模型及其他第三方资源均不包含在仓库中，也不应被视为随源码授权。资源获取方式和发布注意事项见 [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md)。
