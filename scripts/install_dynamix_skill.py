#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dynamix_trace2skill.skill_install import install_skill_for_trace2skill


def main() -> None:
    parser = argparse.ArgumentParser(description="Install DynaMix SKILL.md into Trace2Skill-compatible skills root")
    parser.add_argument("--skill-md", required=True)
    parser.add_argument("--output-skills-root", required=True)
    parser.add_argument("--canonical-skill-name", default="dynamix-qwen3.5-9b-thinking")
    parser.add_argument("--no-trace2skill-compat", action="store_true")
    args = parser.parse_args()
    manifest = install_skill_for_trace2skill(
        args.skill_md,
        args.output_skills_root,
        canonical_skill_name=args.canonical_skill_name,
        create_trace2skill_compatibility_dirs=not args.no_trace2skill_compat,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
