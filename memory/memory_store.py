"""使用单个 JSON 文件持久化玩家关系数据。"""

import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .memory_policy import (
    MemoryCandidate,
    MemoryPolicy,
    MemoryPolicyDecision,
    calculate_relationship_level,
)
from .player_profile import PlayerProfile


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "player_profile.json"


class MemoryStore:
    def __init__(self, path=DEFAULT_PROFILE_PATH, policy=None):
        self.path = Path(path)
        self.policy = policy or MemoryPolicy()
        self._lock = threading.Lock()
        self.profile = self._load_or_recover()

    def remember(self, candidate, confirmed=False):
        decision = self.policy.evaluate(candidate, confirmed=confirmed)
        if not decision.should_save:
            return decision

        with self._lock:
            original = PlayerProfile.from_dict(self.profile.to_dict())
            try:
                self._apply(candidate)
                self.profile.relationship_level = calculate_relationship_level(
                    self.profile
                )
                self._save()
            except (OSError, TypeError, ValueError) as error:
                self.profile = original
                print(f"Memory save failed; current interaction continues: {error}")
                return MemoryPolicyDecision(False, "persistence_failed")
        return decision

    def observe_world_events(self, events):
        """把 World Event 转为候选，并补充当时的关系摘要。"""
        for event in events:
            if event.event_type == "player_enter":
                is_first = self.profile.first_seen == 0
                if is_first:
                    self.remember(
                        MemoryCandidate("interaction", "first_meeting")
                    )
                self.remember(MemoryCandidate("counter", "times_seen"))
                event.context["is_first_meeting"] = is_first
            elif event.event_type == "hand_wave":
                self.remember(MemoryCandidate("counter", "player_waved"))

            event.context["relationship_level"] = (
                self.profile.relationship_level
            )
        return events

    def record_speech_interaction(self):
        return self.remember(
            MemoryCandidate(
                "interaction",
                "interaction_count",
                data={"spoken": True},
            )
        )

    def save_preference(self, key, value, confirmed=False):
        candidate = MemoryCandidate(
            "preference",
            "player_preference",
            data={"key": key, "value": value},
        )
        return self.remember(candidate, confirmed=confirmed)

    def export_profile(self, directory=None, timestamp=None):
        """只导出 PlayerProfile Allowlist，不读取其他项目数据。"""
        directory = (
            self.path.parent / "exports"
            if directory is None
            else Path(directory)
        )
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = timestamp or datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        destination = directory / f"player_profile_{timestamp}.json"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                self.profile.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary, destination)
        return destination

    def clear_profile(self):
        """清除当前档案；保留用户主动创建的 exports。"""
        with self._lock:
            targets = [
                self.path,
                self.path.with_suffix(self.path.suffix + ".tmp"),
            ]
            targets.extend(
                self.path.parent.glob(
                    f"{self.path.stem}.corrupt-*{self.path.suffix}"
                )
            )
            for target in targets:
                try:
                    target.unlink(missing_ok=True)
                except OSError as error:
                    raise OSError(f"无法清除Memory文件 {target}: {error}") from error
            self.profile = PlayerProfile()

    def _apply(self, candidate):
        timestamp = max(0, int(candidate.timestamp))
        if candidate.event == "first_meeting":
            if self.profile.first_seen == 0:
                self.profile.first_seen = timestamp
        elif candidate.event == "times_seen":
            self.profile.times_seen += 1
        elif candidate.event == "interaction_count":
            self.profile.interaction_count += 1
            if candidate.data.get("spoken") is True:
                self.profile.times_spoken += 1
        elif candidate.event == "player_waved":
            self.profile.times_waved += 1
        elif candidate.event == "player_preference":
            key = str(candidate.data.get("key", "")).strip()
            value = candidate.data.get("value")
            if not key or not _safe_preference_value(value):
                raise ValueError("Preference 必须是明确的短文本或简单数值。")
            self.profile.preferences[key[:64]] = (
                value[:200] if isinstance(value, str) else value
            )
        self.profile.last_interaction = timestamp

    def _load_or_recover(self):
        if not self.path.exists():
            return PlayerProfile()
        try:
            with self.path.open("r", encoding="utf-8") as file:
                return PlayerProfile.from_dict(json.load(file))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            self._backup_corrupt_file()
            print(f"Memory profile damaged; recovered with a new profile: {error}")
            profile = PlayerProfile()
            self.profile = profile
            try:
                self._save()
            except OSError as save_error:
                print(f"Memory recovery could not be persisted: {save_error}")
            return profile

    def _backup_corrupt_file(self):
        if not self.path.exists():
            return
        suffix = int(time.time() * 1000)
        backup = self.path.with_name(
            f"{self.path.stem}.corrupt-{suffix}{self.path.suffix}"
        )
        try:
            shutil.copy2(self.path, backup)
        except OSError:
            pass

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                self.profile.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary, self.path)


def _safe_preference_value(value):
    if isinstance(value, str):
        return 0 < len(value.strip()) <= 200
    return isinstance(value, (bool, int, float)) and not isinstance(value, complex)
