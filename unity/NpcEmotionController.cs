using UnityEngine;

public class NpcEmotionController : MonoBehaviour
{
    public NpcPerceptionReceiver perceptionReceiver;
    public Animator npcAnimator;

    // 可选Animator参数。不存在时安全跳过。
    public string emotionStateParameter = "EmotionState";
    public string emotionIntensityParameter = "EmotionIntensity";

    private string lastEmotion = "neutral";
    private float lastIntensity = -1f;

    private void Start()
    {
        if (npcAnimator == null)
        {
            npcAnimator = GetComponent<Animator>();
        }
    }

    private void Update()
    {
        if (perceptionReceiver == null)
        {
            return;
        }

        string emotion = ValidateEmotion(perceptionReceiver.CurrentNpcEmotion);
        float intensity = Mathf.Clamp01(
            perceptionReceiver.CurrentNpcEmotionIntensity
        );
        bool labelChanged = emotion != lastEmotion;
        bool intensityChanged = Mathf.Abs(intensity - lastIntensity) >= 0.05f;
        if (!labelChanged && !intensityChanged)
        {
            return;
        }

        Debug.Log($"NPC Emotion: {emotion} ({intensity:F2})");
        SetOptionalAnimatorInt(emotionStateParameter, EmotionToInt(emotion));
        SetOptionalAnimatorFloat(emotionIntensityParameter, intensity);
        lastEmotion = emotion;
        lastIntensity = intensity;
    }

    private int EmotionToInt(string emotion)
    {
        switch (emotion)
        {
            case "happy": return 1;
            case "curious": return 2;
            case "surprised": return 3;
            default: return 0;
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

    private void SetOptionalAnimatorInt(string parameterName, int value)
    {
        if (npcAnimator == null || string.IsNullOrEmpty(parameterName))
        {
            return;
        }
        foreach (AnimatorControllerParameter parameter in npcAnimator.parameters)
        {
            if (
                parameter.name == parameterName &&
                parameter.type == AnimatorControllerParameterType.Int
            )
            {
                npcAnimator.SetInteger(parameterName, value);
                return;
            }
        }
    }

    private void SetOptionalAnimatorFloat(string parameterName, float value)
    {
        if (npcAnimator == null || string.IsNullOrEmpty(parameterName))
        {
            return;
        }
        foreach (AnimatorControllerParameter parameter in npcAnimator.parameters)
        {
            if (
                parameter.name == parameterName &&
                parameter.type == AnimatorControllerParameterType.Float
            )
            {
                npcAnimator.SetFloat(parameterName, value);
                return;
            }
        }
    }
}
