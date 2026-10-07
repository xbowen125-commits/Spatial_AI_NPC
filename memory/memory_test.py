"""无需 Camera、Microphone、Unity、Agent 或 LLM 的关系记忆测试。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from events.interaction_event import InteractionEvent
from events.world_event import WorldEvent
from memory.memory_cli import (
    build_parser,
    clear_profile,
    export_profile,
    show_profile,
)
from memory.memory_policy import (
    MemoryCandidate,
    MemoryPolicy,
    RelationshipBehaviorRules,
)
from memory.memory_store import MemoryStore
from memory.player_profile import MEMORY_ALLOWLIST_FIELDS
from npc.world_context import build_world_context


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "player_profile.json"
        self.store = MemoryStore(self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_first_and_repeated_enter(self):
        rules = RelationshipBehaviorRules()
        first = WorldEvent.create("player_enter", {})
        self.store.observe_world_events([first])
        first_decision = rules.decide(first)
        self.assertTrue(first.context["is_first_meeting"])
        self.assertEqual(first_decision.reply, "你好，第一次见面。")
        self.assertEqual(self.store.profile.times_seen, 1)

        repeated = WorldEvent.create("player_enter", {})
        self.store.observe_world_events([repeated])
        repeated_decision = rules.decide(repeated)
        self.assertFalse(repeated.context["is_first_meeting"])
        self.assertEqual(repeated_decision.reply, "欢迎回来。")
        self.assertEqual(self.store.profile.times_seen, 2)

    def test_wave_counter(self):
        for _ in range(3):
            self.store.observe_world_events(
                [WorldEvent.create("hand_wave", {})]
            )
        self.assertEqual(self.store.profile.times_waved, 3)

    def test_candidate_events_update_expected_counters(self):
        policy = MemoryPolicy()
        for event in ("seen", "spoken", "wave"):
            self.assertTrue(
                policy.evaluate(
                    MemoryCandidate("interaction", event)
                ).should_save
            )

        self.store.remember(MemoryCandidate("interaction", "seen"))
        self.store.remember(MemoryCandidate("interaction", "spoken"))
        self.store.remember(MemoryCandidate("interaction", "wave"))
        self.assertEqual(self.store.profile.times_seen, 1)
        self.assertEqual(self.store.profile.times_spoken, 1)
        self.assertEqual(self.store.profile.times_waved, 1)
        self.assertEqual(self.store.profile.interaction_count, 1)

    def test_same_world_event_is_counted_only_once(self):
        enter = WorldEvent.create("player_enter", {}, timestamp=1000)
        self.store.observe_world_events([enter])
        self.store.observe_world_events([enter])
        self.assertEqual(self.store.profile.times_seen, 1)

        wave = WorldEvent.create("hand_wave", {}, timestamp=2000)
        self.store.observe_world_events([wave, wave])
        self.assertEqual(self.store.profile.times_waved, 1)

    def test_relationship_upgrade(self):
        for _ in range(2):
            self.store.remember(MemoryCandidate("counter", "times_seen"))
        self.assertEqual(self.store.profile.relationship_level, "acquaintance")

        for _ in range(3):
            self.store.remember(MemoryCandidate("counter", "times_seen"))
        for _ in range(10):
            self.store.record_speech_interaction()
        self.assertEqual(self.store.profile.relationship_level, "friend")

    def test_restart_loads_profile(self):
        self.store.observe_world_events([WorldEvent.create("player_enter", {})])
        self.store.record_speech_interaction()
        loaded = MemoryStore(self.path)
        self.assertEqual(loaded.profile.to_dict(), self.store.profile.to_dict())

    def test_corrupt_json_recovers_and_keeps_backup(self):
        self.path.write_text("{not valid json", encoding="utf-8")
        recovered = MemoryStore(self.path)
        self.assertEqual(recovered.profile.relationship_level, "stranger")
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["times_seen"], 0)
        backups = list(self.path.parent.glob("player_profile.corrupt-*.json"))
        self.assertEqual(len(backups), 1)

    def test_preference_requires_confirmation(self):
        denied = self.store.save_preference("drink", "tea", confirmed=False)
        self.assertFalse(denied.should_save)
        self.assertNotIn("drink", self.store.profile.preferences)

        saved = self.store.save_preference("drink", "tea", confirmed=True)
        self.assertTrue(saved.should_save)
        self.assertEqual(self.store.profile.preferences["drink"], "tea")

    def test_schema_contains_no_raw_media_or_chat(self):
        keys = set(self.store.profile.to_dict())
        forbidden = {
            "audio",
            "image",
            "face",
            "biometric",
            "chat_history",
            "messages",
        }
        self.assertTrue(keys.isdisjoint(forbidden))

    def test_relationship_is_added_to_agent_world_context(self):
        player_state = SimpleNamespace(
            person_detected=True,
            face_detected=True,
            horizontal_position="center",
            eye_contact="true",
            looking_at_npc="true",
            distance="medium",
            left_hand="down",
            right_hand="down",
            directed_speech="true",
        )
        speech = SimpleNamespace(text="你好", language="zh", confidence=None)
        event = InteractionEvent.speech(speech, player_state, "friend")
        self.assertIn(
            "Player relationship is friend.",
            build_world_context(event),
        )
        self.assertNotIn("relationship_level", event.to_dict()["context"])

    def test_cli_show(self):
        self.store.observe_world_events([WorldEvent.create("player_enter", {})])
        output = []
        shown = show_profile(self.path, output=output.append)
        self.assertEqual(set(shown), set(MEMORY_ALLOWLIST_FIELDS))
        self.assertEqual(json.loads(output[0]), shown)

    def test_cli_export_uses_allowlist(self):
        self.store.record_speech_interaction()
        export_directory = self.path.parent / "exports"
        output = []
        exported = export_profile(
            self.path,
            directory=export_directory,
            output=output.append,
        )
        self.assertTrue(exported.is_file())
        exported_data = json.loads(exported.read_text(encoding="utf-8"))
        self.assertEqual(set(exported_data), set(MEMORY_ALLOWLIST_FIELDS))
        serialized = json.dumps(exported_data).lower()
        for forbidden in (
            "speech_text",
            "conversation",
            "audio",
            "image",
            "face_data",
            "api_key",
            "evaluation_log",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_cli_clear_and_first_meeting_after_restart(self):
        self.store.observe_world_events([WorldEvent.create("player_enter", {})])
        self.assertTrue(self.path.exists())
        output = []
        self.assertTrue(
            clear_profile(
                self.path,
                assume_yes=True,
                output=output.append,
            )
        )
        self.assertFalse(self.path.exists())

        restarted = MemoryStore(self.path)
        event = WorldEvent.create("player_enter", {})
        restarted.observe_world_events([event])
        self.assertTrue(event.context["is_first_meeting"])
        self.assertEqual(restarted.profile.times_seen, 1)

    def test_clear_confirmation_and_other_files_untouched(self):
        self.store.observe_world_events([WorldEvent.create("player_enter", {})])
        other_log = self.path.parent.parent / "logs" / "interactions.jsonl"
        other_log.parent.mkdir(parents=True, exist_ok=True)
        other_log.write_text('{"keep": true}\n', encoding="utf-8")
        export = self.store.export_profile(self.path.parent / "exports")

        output = []
        prompts = []
        cancelled = clear_profile(
            self.path,
            input_function=lambda prompt: prompts.append(prompt) or "not-clear",
            output=output.append,
        )
        self.assertFalse(cancelled)
        self.assertTrue(self.path.exists())
        self.assertEqual(prompts, ["Type CLEAR to continue: "])
        self.assertEqual(
            output[0],
            "This will permanently clear the local NPC relationship profile.",
        )

        clear_profile(self.path, assume_yes=True, output=output.append)
        self.assertTrue(other_log.exists())
        self.assertTrue(export.exists())
        parsed = build_parser().parse_args(["clear", "--yes"])
        self.assertTrue(parsed.yes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
