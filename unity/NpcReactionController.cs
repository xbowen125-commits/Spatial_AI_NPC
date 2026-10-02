using UnityEngine;

public class NpcReactionController : MonoBehaviour
{
    // 把场景中的 NpcPerceptionReceiver 组件拖到这里。
    public NpcPerceptionReceiver perceptionReceiver;

    // 可选：NPC 的 Animator。留空时会自动查找当前物体上的 Animator。
    public Animator npcAnimator;
    public string waveTrigger = "Wave";
    public string attentionDetectedTrigger = "AttentionDetected";
    public string eyeContactDetectedTrigger = "EyeContactDetected";

    // 玩家位于画面左右两侧时，NPC 相对初始方向转动的角度。
    public float sideTurnAngle = 30f;
    public float turnSpeed = 90f;

    // 头部先看向玩家；持续偏离中心一段时间后，身体才参与转向。
    public float bodyTurnDelay = 0.6f;

    private Quaternion initialRotation;
    private string lastHandState;
    private string lastDistance = "unknown";
    private float offCenterDuration;
    private string lastLookingAtNpc = "unknown";
    private string lastEyeContact = "unknown";

    private void Start()
    {
        initialRotation = transform.rotation;

        if (npcAnimator == null)
        {
            npcAnimator = GetComponent<Animator>();
        }
    }

    private void Update()
    {
        if (perceptionReceiver == null || !perceptionReceiver.HasState)
        {
            return;
        }

        // 没有检测到玩家时，不执行转向、挥手等互动行为。
        if (!perceptionReceiver.CurrentPersonDetected)
        {
            offCenterDuration = 0f;
            lastHandState = "unknown:unknown";
            lastDistance = "unknown";
            lastLookingAtNpc = "unknown";
            lastEyeContact = "unknown";
            RotateBodyTowards(0f);
            return;
        }

        UpdateHorizontalRotation();
        UpdateHandReaction();
        ReportDistanceChange();
        UpdateAttentionReaction();
        UpdateEyeContactReaction();
    }

    private void UpdateHorizontalRotation()
    {
        float targetAngle = 0f;
        bool playerOffCenter =
            perceptionReceiver.CurrentHorizontalPosition == "left" ||
            perceptionReceiver.CurrentHorizontalPosition == "right";

        if (playerOffCenter)
        {
            offCenterDuration += Time.deltaTime;
        }
        else
        {
            offCenterDuration = 0f;
        }

        if (
            offCenterDuration >= bodyTurnDelay &&
            perceptionReceiver.CurrentHorizontalPosition == "left"
        )
        {
            targetAngle = -sideTurnAngle;
        }
        else if (
            offCenterDuration >= bodyTurnDelay &&
            perceptionReceiver.CurrentHorizontalPosition == "right"
        )
        {
            targetAngle = sideTurnAngle;
        }

        RotateBodyTowards(targetAngle);
    }

    private void RotateBodyTowards(float targetAngle)
    {
        // RotateTowards 让整体水平转向保持平滑，而不是瞬间旋转。
        Quaternion targetRotation =
            initialRotation * Quaternion.Euler(0f, targetAngle, 0f);
        transform.rotation = Quaternion.RotateTowards(
            transform.rotation,
            targetRotation,
            turnSpeed * Time.deltaTime
        );
    }

    private void UpdateHandReaction()
    {
        string handState =
            perceptionReceiver.CurrentLeftHand + ":" +
            perceptionReceiver.CurrentRightHand;

        if (handState == lastHandState)
        {
            return;
        }

        bool handRaised =
            perceptionReceiver.CurrentLeftHand == "raised" ||
            perceptionReceiver.CurrentRightHand == "raised";

        if (handRaised)
        {
            Debug.Log(
                $"NPC：检测到举手，左手 {perceptionReceiver.CurrentLeftHand}，" +
                $"右手 {perceptionReceiver.CurrentRightHand}。"
            );
            // v0.9 由 Behavior Manager 发出 npc_action=wave，
            // 再交给 NpcActionController 统一触发，避免同一次举手播放两次。
        }

        lastHandState = handState;
    }

    private void ReportDistanceChange()
    {
        if (perceptionReceiver.CurrentDistance == lastDistance)
        {
            return;
        }

        Debug.Log($"NPC：玩家相对距离变为 {perceptionReceiver.CurrentDistance}。");
        lastDistance = perceptionReceiver.CurrentDistance;
    }

    private void UpdateAttentionReaction()
    {
        string currentLooking = perceptionReceiver.CurrentLookingAtNpc;

        // 只在非 true → true 的上升沿触发一次，不会随 UDP 心跳重复触发。
        if (currentLooking == "true" && lastLookingAtNpc != "true")
        {
            Debug.Log("NPC：检测到玩家开始注意我。");
            PlayOptionalTrigger(attentionDetectedTrigger, false);
        }

        lastLookingAtNpc = currentLooking;
    }

    private void PlayOptionalTrigger(string triggerName, bool warnIfMissing)
    {
        if (npcAnimator == null)
        {
            return;
        }

        // 只有 Animator 中存在 Wave Trigger 时才调用，避免参数缺失报错。
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

        if (warnIfMissing)
        {
            Debug.LogWarning($"Animator 中没有名为 {triggerName} 的 Trigger 参数。");
        }
    }

    private void UpdateEyeContactReaction()
    {
        string currentEyeContact = perceptionReceiver.CurrentEyeContact;

        // false / unknown → true 时只触发一次，不随 UDP 心跳重复。
        if (currentEyeContact == "true" && lastEyeContact != "true")
        {
            Debug.Log("NPC：检测到玩家与我建立了眼神接触。");
            PlayOptionalTrigger(eyeContactDetectedTrigger, false);
        }

        lastEyeContact = currentEyeContact;
    }
}
