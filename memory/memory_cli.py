"""查看、导出和清除本地 NPC Relationship Memory。"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.memory_store import DEFAULT_PROFILE_PATH, MemoryStore


def show_profile(path=DEFAULT_PROFILE_PATH, output=print):
    store = MemoryStore(path)
    text = json.dumps(store.profile.to_dict(), ensure_ascii=False, indent=2)
    output(text)
    return store.profile.to_dict()


def export_profile(path=DEFAULT_PROFILE_PATH, directory=None, output=print):
    store = MemoryStore(path)
    destination = store.export_profile(directory=directory)
    output(f"Memory profile exported to: {destination}")
    return destination


def clear_profile(
    path=DEFAULT_PROFILE_PATH,
    assume_yes=False,
    input_function=input,
    output=print,
):
    if not assume_yes:
        output("This will permanently clear the local NPC relationship profile.")
        confirmation = input_function("Type CLEAR to continue: ")
        if confirmation != "CLEAR":
            output("Memory clear cancelled.")
            return False

    store = MemoryStore(path)
    store.clear_profile()
    output("Local NPC relationship profile cleared.")
    return True


def build_parser():
    parser = argparse.ArgumentParser(description="NPC Relationship Memory Control")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show", help="显示当前Player Profile")
    subparsers.add_parser("export", help="导出Allowlist档案副本")
    clear_parser = subparsers.add_parser("clear", help="清除当前Player Profile")
    clear_parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过CLEAR确认，仅供自动测试或明确脚本调用",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "show":
        show_profile()
    elif args.command == "export":
        export_profile()
    elif args.command == "clear":
        clear_profile(assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
