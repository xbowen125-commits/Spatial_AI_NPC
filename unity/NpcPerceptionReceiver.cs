using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class PerceptionState
{
    public string message_type;
    public bool person_detected;
    public bool face_detected;
    public string horizontal_position;
    public string distance;
    public string left_hand;
    public string right_hand;
    public string head_direction;
    public float head_yaw;
    public float head_pitch;
    public float attention_score;
    public string looking_at_npc;
    public string gaze_horizontal;
    public string gaze_vertical;
    public float gaze_score;
    public string eye_contact;
    public bool is_speaking;
    public string directed_speech;

    // Python 会暂时保留这两个旧字段，兼容已有场景。
    public string gesture;
    public string position;
}

[Serializable]
public class MessageEnvelope
{
    public string message_type;
}

[Serializable]
public class SpeechEventContext
{
    public bool person_detected;
    public bool face_detected;
    public string horizontal_position;
    public string eye_contact;
    public string looking_at_npc;
    public string distance;
    public string left_hand;
    public string right_hand;
    public string directed_speech;
}

[Serializable]
public class InteractionEventMessage
{
    public string message_type;
    public string event_type;
    public long timestamp;
    public string text;
    public string language;
    public string source_event;
    public float subtitle_duration;
    public string action;
    public string emotion;
    public SpeechEventContext context;
}

[Serializable]
public class NpcStateMessage
{
    public string message_type;
    public bool npc_is_thinking;
    public bool npc_is_speaking;
    public string npc_emotion;
    public string npc_behavior_state;
    public float npc_emotion_intensity;
}

public class NpcPerceptionReceiver : MonoBehaviour
{
    // 必须与 Python 中的 UNITY_PORT 保持一致。
    public int port = 5052;

    // 超过此时间没有收到 UDP，就把玩家状态重置为离线。
    public float udpTimeoutSeconds = 2f;

    // Spatial Perception v0.2 的标准玩家状态。
    public bool CurrentPersonDetected { get; private set; }
    public bool CurrentFaceDetected { get; private set; }
    public string CurrentHorizontalPosition { get; private set; } = "unknown";
    public string CurrentDistance { get; private set; } = "unknown";
    public string CurrentLeftHand { get; private set; } = "unknown";
    public string CurrentRightHand { get; private set; } = "unknown";
    public string CurrentHeadDirection { get; private set; } = "unknown";
    public float CurrentHeadYaw { get; private set; }
    public float CurrentHeadPitch { get; private set; }
    public float CurrentAttentionScore { get; private set; }
    public string CurrentLookingAtNpc { get; private set; } = "unknown";
    public string CurrentGazeHorizontal { get; private set; } = "unknown";
    public string CurrentGazeVertical { get; private set; } = "unknown";
    public float CurrentGazeScore { get; private set; }
    public string CurrentEyeContact { get; private set; } = "unknown";
    public bool CurrentIsSpeaking { get; private set; }
    public string CurrentDirectedSpeech { get; private set; } = "false";
    public bool CurrentNpcIsSpeaking { get; private set; }
    public bool CurrentNpcIsThinking { get; private set; }
    public string CurrentNpcEmotion { get; private set; } = "neutral";
    public string CurrentNpcBehaviorState { get; private set; } = "idle";
    public float CurrentNpcEmotionIntensity { get; private set; }
    public bool HasState { get; private set; }

    // 一次性 Interaction Event 交给独立控制器处理。
    public event Action<InteractionEventMessage> InteractionEventReceived;

    // 保留旧属性，避免已有组件立即失效。
    public string CurrentGesture { get; private set; } = "No person detected";
    public string CurrentPosition { get; private set; } = "Unknown";

    private UdpClient udpClient;
    private Thread receiveThread;
    private readonly object messageLock = new object();
    private readonly Queue<string> messageQueue = new Queue<string>();
    private string lastLoggedState;
    private float lastMessageTime;
    private bool timeoutApplied;

    private void Start()
    {
        udpClient = new UdpClient(port);
        receiveThread = new Thread(ReceiveMessages);
        receiveThread.IsBackground = true;
        receiveThread.Start();

        Debug.Log($"等待 Python Player State，UDP 端口：{port}");
    }

    private void ReceiveMessages()
    {
        IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);

        try
        {
            while (true)
            {
                byte[] data = udpClient.Receive(ref sender);
                string message = Encoding.UTF8.GetString(data);

                // 网络线程只保存消息，Unity 主线程在 Update 中解析。
                lock (messageLock)
                {
                    messageQueue.Enqueue(message);
                }
            }
        }
        catch (SocketException)
        {
            // 关闭程序时正常结束接收线程。
        }
        catch (ObjectDisposedException)
        {
            // UDP 被关闭时正常结束。
        }
    }

    private void Update()
    {
        string message = null;

        lock (messageLock)
        {
            if (messageQueue.Count > 0)
            {
                message = messageQueue.Dequeue();
            }
        }

        if (message == null)
        {
            CheckUdpTimeout();
            return;
        }

        MessageEnvelope envelope = JsonUtility.FromJson<MessageEnvelope>(message);
        if (envelope != null && envelope.message_type == "interaction_event")
        {
            InteractionEventMessage interactionEvent =
                JsonUtility.FromJson<InteractionEventMessage>(message);
            InteractionEventReceived?.Invoke(interactionEvent);
            CheckUdpTimeout();
            return;
        }
        if (envelope != null && envelope.message_type == "npc_state")
        {
            NpcStateMessage npcState = JsonUtility.FromJson<NpcStateMessage>(message);
            CurrentNpcIsThinking = npcState.npc_is_thinking;
            CurrentNpcIsSpeaking = npcState.npc_is_speaking;
            CurrentNpcEmotion = npcState.npc_emotion ?? "neutral";
            CurrentNpcBehaviorState = npcState.npc_behavior_state ?? "idle";
            CurrentNpcEmotionIntensity = Mathf.Clamp01(
                npcState.npc_emotion_intensity
            );
            CheckUdpTimeout();
            return;
        }

        // message_type 缺失时仍按旧版 Player State 解析，保持向后兼容。
        PerceptionState state = JsonUtility.FromJson<PerceptionState>(message);

        CurrentPersonDetected = state.person_detected;
        CurrentFaceDetected = state.face_detected;
        CurrentHorizontalPosition = state.horizontal_position ?? "unknown";
        CurrentDistance = state.distance ?? "unknown";
        CurrentLeftHand = state.left_hand ?? "unknown";
        CurrentRightHand = state.right_hand ?? "unknown";
        CurrentHeadDirection = state.head_direction ?? "unknown";
        CurrentHeadYaw = state.head_yaw;
        CurrentHeadPitch = state.head_pitch;
        CurrentAttentionScore = state.attention_score;
        CurrentLookingAtNpc = state.looking_at_npc ?? "unknown";
        CurrentGazeHorizontal = state.gaze_horizontal ?? "unknown";
        CurrentGazeVertical = state.gaze_vertical ?? "unknown";
        CurrentGazeScore = state.gaze_score;
        CurrentEyeContact = state.eye_contact ?? "unknown";
        CurrentIsSpeaking = state.is_speaking;
        CurrentDirectedSpeech = state.directed_speech ?? "false";
        HasState = true;
        lastMessageTime = Time.realtimeSinceStartup;
        timeoutApplied = false;

        // 从新状态派生旧属性，现有 Wave 和其他旧组件仍可继续使用。
        CurrentGesture = BuildLegacyGesture();
        CurrentPosition = ToTitleCase(CurrentHorizontalPosition);

        string stateSummary =
            $"{CurrentPersonDetected}|{CurrentFaceDetected}|" +
            $"{CurrentHorizontalPosition}|{CurrentDistance}|" +
            $"{CurrentLeftHand}|{CurrentRightHand}|{CurrentHeadDirection}|" +
            $"{CurrentHeadYaw:F1}|{CurrentHeadPitch:F1}|" +
            $"{CurrentAttentionScore:F2}|{CurrentLookingAtNpc}|" +
            $"{CurrentGazeHorizontal}|{CurrentGazeVertical}|" +
            $"{CurrentGazeScore:F2}|{CurrentEyeContact}|" +
            $"{CurrentIsSpeaking}|{CurrentDirectedSpeech}";

        // UDP 有低频心跳，但 Console 只在状态真正改变时输出。
        if (stateSummary != lastLoggedState)
        {
            Debug.Log(
                $"玩家：{CurrentPersonDetected}，人脸：{CurrentFaceDetected}，" +
                $"位置：{CurrentHorizontalPosition}，" +
                $"距离：{CurrentDistance}，左手：{CurrentLeftHand}，" +
                $"右手：{CurrentRightHand}，头部：{CurrentHeadDirection}，" +
                $"Yaw：{CurrentHeadYaw:F1}，Pitch：{CurrentHeadPitch:F1}，" +
                $"Attention：{CurrentAttentionScore:F2}，" +
                $"Looking：{CurrentLookingAtNpc}，" +
                $"Gaze：{CurrentGazeHorizontal}/{CurrentGazeVertical}，" +
                $"Gaze Score：{CurrentGazeScore:F2}，" +
                $"Eye Contact：{CurrentEyeContact}，" +
                $"Speaking：{CurrentIsSpeaking}，" +
                $"Directed Speech：{CurrentDirectedSpeech}"
            );
            lastLoggedState = stateSummary;
        }
    }

    private void CheckUdpTimeout()
    {
        if (
            !HasState ||
            timeoutApplied ||
            Time.realtimeSinceStartup - lastMessageTime <= udpTimeoutSeconds
        )
        {
            return;
        }

        CurrentPersonDetected = false;
        CurrentFaceDetected = false;
        CurrentHorizontalPosition = "unknown";
        CurrentDistance = "unknown";
        CurrentLeftHand = "unknown";
        CurrentRightHand = "unknown";
        CurrentHeadDirection = "unknown";
        CurrentHeadYaw = 0f;
        CurrentHeadPitch = 0f;
        CurrentAttentionScore = 0f;
        CurrentLookingAtNpc = "unknown";
        CurrentGazeHorizontal = "unknown";
        CurrentGazeVertical = "unknown";
        CurrentGazeScore = 0f;
        CurrentEyeContact = "unknown";
        CurrentIsSpeaking = false;
        CurrentDirectedSpeech = "false";
        CurrentNpcIsSpeaking = false;
        CurrentNpcIsThinking = false;
        CurrentNpcEmotion = "neutral";
        CurrentNpcBehaviorState = "idle";
        CurrentNpcEmotionIntensity = 0f;
        CurrentGesture = "No person detected";
        CurrentPosition = "Unknown";
        timeoutApplied = true;
        lastLoggedState = null;

        Debug.LogWarning("超过 UDP 超时时间，NPC 已恢复为 idle 状态。");
    }

    private string BuildLegacyGesture()
    {
        if (!CurrentPersonDetected)
        {
            return "No person detected";
        }
        if (CurrentLeftHand == "raised" && CurrentRightHand == "raised")
        {
            return "Both hands raised";
        }
        if (CurrentLeftHand == "raised")
        {
            return "Left hand raised";
        }
        if (CurrentRightHand == "raised")
        {
            return "Right hand raised";
        }
        return "No hand raised";
    }

    private string ToTitleCase(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return "Unknown";
        }
        return char.ToUpperInvariant(value[0]) + value.Substring(1);
    }

    private void OnDestroy()
    {
        udpClient?.Close();
        receiveThread?.Join(200);
    }
}
