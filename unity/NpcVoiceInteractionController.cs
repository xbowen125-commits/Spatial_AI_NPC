using System.Collections;
using System.Reflection;
using UnityEngine;

public class NpcVoiceInteractionController : MonoBehaviour
{
    // 独立处理语音事件与 Listening 状态，避免继续扩大 ReactionController。
    public NpcPerceptionReceiver perceptionReceiver;
    public Animator npcAnimator;
    public string listeningBool = "Listening";
    public string speakingBool = "Speaking";

    // 可选拖入 Text、TextMeshProUGUI 等具有 text 属性的字幕组件。
    public Component npcSubtitleText;
    public float subtitleDuration = 1f;

    private bool lastListening;
    private bool lastSpeaking;
    private Coroutine hideSubtitleCoroutine;

    private void Start()
    {
        if (npcAnimator == null)
        {
            npcAnimator = GetComponent<Animator>();
        }
    }

    private void OnEnable()
    {
        if (perceptionReceiver != null)
        {
            perceptionReceiver.InteractionEventReceived += OnInteractionEvent;
        }
    }

    private void OnDisable()
    {
        if (perceptionReceiver != null)
        {
            perceptionReceiver.InteractionEventReceived -= OnInteractionEvent;
        }
        SetOptionalAnimatorBool(listeningBool, false);
        SetOptionalAnimatorBool(speakingBool, false);
    }

    private void Update()
    {
        bool listening =
            perceptionReceiver != null &&
            perceptionReceiver.HasState &&
            perceptionReceiver.CurrentIsSpeaking;

        if (listening != lastListening)
        {
            Debug.Log(listening ? "NPC：正在听玩家说话。" : "NPC：玩家停止说话。");
            SetOptionalAnimatorBool(listeningBool, listening);
            lastListening = listening;
        }

        bool speaking =
            perceptionReceiver != null &&
            perceptionReceiver.CurrentNpcIsSpeaking;

        if (speaking != lastSpeaking)
        {
            Debug.Log(speaking ? "NPC：开始说话。" : "NPC：说话结束。");
            SetOptionalAnimatorBool(speakingBool, speaking);
            if (!speaking)
            {
                ScheduleSubtitleHide();
            }
            lastSpeaking = speaking;
        }
    }

    private void OnInteractionEvent(InteractionEventMessage interactionEvent)
    {
        if (interactionEvent == null)
        {
            return;
        }

        if (interactionEvent.event_type == "npc_response")
        {
            Debug.Log($"NPC: {interactionEvent.text}");
            if (interactionEvent.subtitle_duration >= 0f)
            {
                subtitleDuration = interactionEvent.subtitle_duration;
            }
            ShowSubtitle(interactionEvent.text);
            return;
        }

        if (interactionEvent.event_type != "speech")
        {
            return;
        }

        Debug.Log($"Player said: {interactionEvent.text} [{interactionEvent.language}]");

        if (interactionEvent.context != null)
        {
            Debug.Log(
                $"Speech Context：Eye Contact=" +
                $"{interactionEvent.context.eye_contact}，Looking=" +
                $"{interactionEvent.context.looking_at_npc}，Distance=" +
                $"{interactionEvent.context.distance}，Directed=" +
                $"{interactionEvent.context.directed_speech}"
            );
        }
    }

    private void ShowSubtitle(string text)
    {
        if (npcSubtitleText == null)
        {
            return;
        }
        if (hideSubtitleCoroutine != null)
        {
            StopCoroutine(hideSubtitleCoroutine);
            hideSubtitleCoroutine = null;
        }
        SetSubtitleText(text);
    }

    private void ScheduleSubtitleHide()
    {
        if (npcSubtitleText == null)
        {
            return;
        }
        if (hideSubtitleCoroutine != null)
        {
            StopCoroutine(hideSubtitleCoroutine);
        }
        hideSubtitleCoroutine = StartCoroutine(HideSubtitleAfterDelay());
    }

    private IEnumerator HideSubtitleAfterDelay()
    {
        yield return new WaitForSeconds(Mathf.Max(0f, subtitleDuration));
        SetSubtitleText("");
        hideSubtitleCoroutine = null;
    }

    private void SetSubtitleText(string value)
    {
        // 使用反射兼容 Unity UI Text 与 TextMeshPro，无需强制引入 TMP 依赖。
        try
        {
            PropertyInfo textProperty = npcSubtitleText.GetType().GetProperty("text");
            if (textProperty != null && textProperty.CanWrite)
            {
                textProperty.SetValue(npcSubtitleText, value, null);
            }
        }
        catch
        {
            // 字幕组件不兼容时安全跳过，不影响语音、动画和感知。
        }
    }

    private void SetOptionalAnimatorBool(string parameterName, bool value)
    {
        if (npcAnimator == null || string.IsNullOrEmpty(parameterName))
        {
            return;
        }

        foreach (AnimatorControllerParameter parameter in npcAnimator.parameters)
        {
            if (
                parameter.name == parameterName &&
                parameter.type == AnimatorControllerParameterType.Bool
            )
            {
                npcAnimator.SetBool(parameterName, value);
                return;
            }
        }
    }
}
