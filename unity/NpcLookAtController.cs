using UnityEngine;

[DefaultExecutionOrder(12000)]
public class NpcLookAtController : MonoBehaviour
{
    // Player State 接收器。
    public NpcPerceptionReceiver perceptionReceiver;

    // Humanoid Animator 与头部骨骼。留空时脚本会自动查找。
    public Animator npcAnimator;
    public Transform headBone;

    // 可选眼睛骨骼；缺失时安全跳过，不影响头部 LookAt。
    public Transform leftEyeBone;
    public Transform rightEyeBone;

    // 玩家位于左右区域时，头部尝试转动的角度。
    public float sideLookAngle = 22f;

    // 安全旋转范围，防止头部出现不自然的大角度扭转。
    public float maxHeadYaw = 50f;
    public float maxHeadPitch = 25f;

    // 数值越大，头部跟随越快。
    public float headFollowSpeed = 6f;

    // 眼睛先于头部响应，并限制在自然的小角度范围内。
    public float sideEyeAngle = 6f;
    public float maxEyeYaw = 8f;
    public float maxEyePitch = 6f;
    public float eyeFollowSpeed = 10f;

    // Inspector中的Eye Bone为空时，可以通过Humanoid Avatar自动取得。
    // 关闭后None就表示完全不控制眼睛，便于和VRM LookAt做AB测试。
    public bool autoFindEyeBones = true;

    // 眼睛只承担小幅辅助，主要方向由头部完成。
    [Range(0f, 0.4f)] public float eyeAssistRatio = 0.25f;

    // 不同角色骨骼方向可能相反，可以在 Inspector 中切换。
    public bool invertYaw;

    // 可选调试：只在玩家位置状态变化时输出，不会每帧刷屏。
    [SerializeField] private bool debugLookAt;
    [SerializeField] private float debugLogInterval = 1f;

    private Quaternion initialHeadLocalRotation;
    private Quaternion initialLeftEyeLocalRotation;
    private Quaternion initialRightEyeLocalRotation;
    private float currentHeadYaw;
    private float currentHeadPitch;
    private float currentEyeYaw;
    private float currentEyePitch;
    private string lastDebugLookAtState;
    private float nextDebugLogTime;

    private void Start()
    {
        if (npcAnimator == null)
        {
            npcAnimator = GetComponent<Animator>();
        }

        if (
            headBone == null &&
            npcAnimator != null &&
            npcAnimator.isHuman
        )
        {
            headBone = npcAnimator.GetBoneTransform(HumanBodyBones.Head);
        }

        if (autoFindEyeBones && npcAnimator != null && npcAnimator.isHuman)
        {
            if (leftEyeBone == null)
            {
                leftEyeBone = npcAnimator.GetBoneTransform(HumanBodyBones.LeftEye);
            }
            if (rightEyeBone == null)
            {
                rightEyeBone = npcAnimator.GetBoneTransform(HumanBodyBones.RightEye);
            }
        }

        if (headBone == null)
        {
            Debug.LogWarning("NpcLookAtController：没有找到 NPC Head 骨骼。");
            enabled = false;
            return;
        }

        initialHeadLocalRotation = headBone.localRotation;

        if (leftEyeBone != null)
        {
            initialLeftEyeLocalRotation = leftEyeBone.localRotation;
        }
        if (rightEyeBone != null)
        {
            initialRightEyeLocalRotation = rightEyeBone.localRotation;
        }
    }

    private void LateUpdate()
    {
        if (headBone == null)
        {
            return;
        }

        float targetYaw = 0f;
        float targetPitch = 0f;

        if (
            perceptionReceiver != null &&
            perceptionReceiver.HasState &&
            perceptionReceiver.CurrentPersonDetected
        )
        {
            if (perceptionReceiver.CurrentHorizontalPosition == "left")
            {
                targetYaw = -sideLookAngle;
            }
            else if (perceptionReceiver.CurrentHorizontalPosition == "right")
            {
                targetYaw = sideLookAngle;
            }
        }

        if (invertYaw)
        {
            targetYaw = -targetYaw;
        }

        targetYaw = Mathf.Clamp(targetYaw, -maxHeadYaw, maxHeadYaw);
        targetPitch = Mathf.Clamp(targetPitch, -maxHeadPitch, maxHeadPitch);

        // 平滑的是LookAt偏移角，而不是动画每帧生成的完整头骨姿态。
        float blend = 1f - Mathf.Exp(-headFollowSpeed * Time.deltaTime);
        currentHeadYaw = Mathf.Lerp(currentHeadYaw, targetYaw, blend);
        currentHeadPitch = Mathf.Lerp(currentHeadPitch, targetPitch, blend);

        bool animatorDrivingPose = IsAnimatorDrivingPose();

        // Animator存在时读取本帧Idle/VRM处理后的世界姿态；没有Controller时
        // 从启动局部姿态重建世界姿态，避免把上一帧LookAt继续累积。
        Quaternion animatedBaseWorldRotation = animatorDrivingPose
            ? headBone.rotation
            : HeadInitialWorldRotation();

        // 使用角色根节点的Up/Right作为语义明确的yaw/pitch轴，不依赖
        // VRoid Head骨骼自身的局部轴方向。Airi Root Y=180会自然包含在这些轴中。
        Transform avatarRoot = npcAnimator != null
            ? npcAnimator.transform
            : transform;
        Quaternion yawOffset = Quaternion.AngleAxis(
            currentHeadYaw,
            avatarRoot.up
        );
        Quaternion pitchOffset = Quaternion.AngleAxis(
            currentHeadPitch,
            avatarRoot.right
        );
        headBone.rotation =
            yawOffset * pitchOffset * animatedBaseWorldRotation;

        UpdateEyeLookAt(
            animatorDrivingPose,
            targetYaw,
            targetPitch
        );
        LogLookAtStateIfNeeded(
            currentHeadYaw,
            currentHeadPitch,
            avatarRoot
        );
    }

    private Quaternion HeadInitialWorldRotation()
    {
        if (headBone.parent == null)
        {
            return initialHeadLocalRotation;
        }
        return headBone.parent.rotation * initialHeadLocalRotation;
    }

    private bool IsAnimatorDrivingPose()
    {
        return
            npcAnimator != null &&
            npcAnimator.isActiveAndEnabled &&
            npcAnimator.runtimeAnimatorController != null;
    }

    private void UpdateEyeLookAt(
        bool animatorDrivingPose,
        float headTargetYaw,
        float headTargetPitch
    )
    {
        float yawLimit = Mathf.Min(Mathf.Abs(sideEyeAngle), maxEyeYaw);
        float targetYaw = Mathf.Clamp(
            headTargetYaw * eyeAssistRatio,
            -yawLimit,
            yawLimit
        );
        float targetPitch = Mathf.Clamp(
            headTargetPitch * eyeAssistRatio,
            -maxEyePitch,
            maxEyePitch
        );
        float eyeBlend = 1f - Mathf.Exp(-eyeFollowSpeed * Time.deltaTime);
        currentEyeYaw = Mathf.Lerp(currentEyeYaw, targetYaw, eyeBlend);
        currentEyePitch = Mathf.Lerp(currentEyePitch, targetPitch, eyeBlend);
        Quaternion eyeOffset = Quaternion.Euler(
            currentEyePitch,
            currentEyeYaw,
            0f
        );

        if (leftEyeBone != null)
        {
            Quaternion leftEyeBase = animatorDrivingPose
                ? leftEyeBone.localRotation
                : initialLeftEyeLocalRotation;
            leftEyeBone.localRotation = leftEyeBase * eyeOffset;
        }

        if (rightEyeBone != null)
        {
            Quaternion rightEyeBase = animatorDrivingPose
                ? rightEyeBone.localRotation
                : initialRightEyeLocalRotation;
            rightEyeBone.localRotation = rightEyeBase * eyeOffset;
        }
    }

    private void LogLookAtStateIfNeeded(
        float yaw,
        float pitch,
        Transform avatarRoot
    )
    {
        if (!debugLookAt)
        {
            return;
        }

        string position = "offline";
        if (
            perceptionReceiver != null &&
            perceptionReceiver.HasState &&
            perceptionReceiver.CurrentPersonDetected
        )
        {
            position = perceptionReceiver.CurrentHorizontalPosition;
        }

        bool stateChanged = position != lastDebugLookAtState;
        if (!stateChanged && Time.unscaledTime < nextDebugLogTime)
        {
            return;
        }

        Vector3 flatHeadForward = Vector3.ProjectOnPlane(
            headBone.forward,
            avatarRoot.up
        );
        float worldYaw = flatHeadForward.sqrMagnitude > 0.0001f
            ? Vector3.SignedAngle(
                avatarRoot.forward,
                flatHeadForward,
                avatarRoot.up
            )
            : 0f;
        Debug.Log(
            $"NpcLookAt：position={position}，yaw={yaw:F1}，" +
            $"pitch={pitch:F1}，worldYaw={worldYaw:F1}。"
        );
        lastDebugLookAtState = position;
        nextDebugLogTime = Time.unscaledTime + Mathf.Max(
            0.25f,
            debugLogInterval
        );
    }
}
