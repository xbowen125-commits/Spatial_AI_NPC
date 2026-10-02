using UnityEngine;

public class NpcActionController : MonoBehaviour
{
    public NpcPerceptionReceiver perceptionReceiver;
    public Animator npcAnimator;

    public string waveTrigger = "Wave";
    public string nodTrigger = "Nod";
    public string lookAtPlayerTrigger = "LookAtPlayer";
    public string thinkingBool = "Thinking";

    private bool lastThinking;
    private string lastEmotion = "neutral";

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
        SetOptionalAnimatorBool(thinkingBool, false);
    }

    private void Update()
    {
        if (perceptionReceiver == null)
        {
            return;
        }

        bool thinking = perceptionReceiver.CurrentNpcIsThinking;
        if (thinking != lastThinking)
        {
            Debug.Log(thinking ? "NPC：开始思考。" : "NPC：思考结束。");
            SetOptionalAnimatorBool(thinkingBool, thinking);
            lastThinking = thinking;
        }

        string emotion = ValidateEmotion(perceptionReceiver.CurrentNpcEmotion);
        if (emotion != lastEmotion)
        {
            Debug.Log($"NPC Emotion: {emotion}");
            lastEmotion = emotion;
        }
    }

    private void OnInteractionEvent(InteractionEventMessage interactionEvent)
    {
        if (
            interactionEvent == null ||
            interactionEvent.event_type != "npc_action"
        )
        {
            return;
        }

        // Unity 再做一次白名单验证，不信任网络消息中的任意动作名称。
        string action = interactionEvent.action ?? "none";
        switch (action)
        {
            case "wave":
                Debug.Log("NPC Action: wave");
                PlayOptionalTrigger(waveTrigger);
                break;
            case "nod":
                Debug.Log("NPC Action: nod");
                PlayOptionalTrigger(nodTrigger);
                break;
            case "look_at_player":
                Debug.Log("NPC Action: look_at_player");
                PlayOptionalTrigger(lookAtPlayerTrigger);
                break;
            case "none":
                Debug.Log("NPC Action: none");
                break;
            default:
                Debug.LogWarning($"忽略非白名单 NPC Action: {action}");
                break;
        }
    }

    private string ValidateEmotion(string emotion)
    {
        switch (emotion)
        {
            case "happy":
            case "curious":
            case "surprised":
            case "neutral":
                return emotion;
            default:
                return "neutral";
        }
    }

    private void PlayOptionalTrigger(string triggerName)
    {
        if (npcAnimator == null || string.IsNullOrEmpty(triggerName))
        {
            return;
        }
        foreach (AnimatorControllerParameter parameter in npcAnimator.parameters)
        {
            if (
                parameter.name == triggerName &&
                parameter.type == AnimatorControllerParameterType.Trigger
            )
            {
                npcAnimator.SetTrigger(triggerName);
                return;
            }
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
