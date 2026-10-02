"""将原始感知字段转换成简洁、权威的语义上下文。"""


def build_world_context(speech_event):
    context = speech_event.context
    lines = []

    lines.append(
        "Player is visible."
        if context.person_detected
        else "Player is not visible."
    )
    lines.append(
        "Player's face is detected."
        if context.face_detected
        else "Player's face is not detected."
    )

    if context.distance == "unknown":
        lines.append("Player distance is uncertain.")
    else:
        lines.append(f"Player distance is {context.distance}.")

    if context.horizontal_position == "unknown":
        lines.append("Player horizontal position is uncertain.")
    else:
        lines.append(f"Player is positioned {context.horizontal_position}.")

    lines.append(_tri_state("Player is making eye contact.", "Player is not making eye contact.", "Eye contact is uncertain.", context.eye_contact))
    lines.append(_tri_state("Player's head is oriented toward the NPC.", "Player's head is oriented away from the NPC.", "Player head attention is uncertain.", context.looking_at_npc))
    lines.append(_hand_state("left", context.left_hand))
    lines.append(_hand_state("right", context.right_hand))
    lines.append(f"Directed speech estimate is {context.directed_speech}.")
    relationship = str(
        getattr(context, "relationship_level", "stranger")
    ).lower()
    if relationship not in {"stranger", "acquaintance", "friend"}:
        relationship = "stranger"
    lines.append(f"Player relationship is {relationship}.")
    npc_emotion = str(getattr(context, "npc_emotion", "neutral")).lower()
    if npc_emotion not in {"neutral", "happy", "curious", "surprised"}:
        npc_emotion = "neutral"
    intensity = max(
        0.0,
        min(1.0, float(getattr(context, "npc_emotion_intensity", 0.0))),
    )
    lines.append(f"NPC current emotion is {npc_emotion}.")
    lines.append(f"NPC emotion intensity is {intensity:.2f}.")

    return "\n".join(lines)


def _tri_state(true_text, false_text, unknown_text, value):
    if value == "true":
        return true_text
    if value == "false":
        return false_text
    return unknown_text


def _hand_state(side, value):
    if value == "raised":
        return f"Player has their {side} hand raised."
    if value == "down":
        return f"Player's {side} hand is down."
    return f"Player's {side} hand state is uncertain."
