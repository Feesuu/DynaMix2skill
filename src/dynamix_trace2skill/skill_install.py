from __future__ import annotations

import json
import shutil
from pathlib import Path

TRACE2SKILL_REQUIRED_SKILL_DIRS = ("xlsx", "xlsx-122B", "xlsx-35B")


def _copy_skill_folder(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for child in list(dest_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in source_dir.iterdir():
        dest = dest_dir / child.name
        if child.is_dir():
            shutil.copytree(child, dest)
        else:
            shutil.copy2(child, dest)


def install_skill_for_trace2skill(
    skill_md: str | Path,
    output_skills_root: str | Path,
    *,
    canonical_skill_name: str = "dynamix-qwen3.5-9b-thinking",
    create_trace2skill_compatibility_dirs: bool = True,
) -> dict:
    """Install a generated DynaMix skill folder into a Trace2Skill skills root.

    The exporter may create SKILL.md plus support files under references/ and
    scripts/.  Trace2Skill's preloaded agent tells the model that support files
    are located in the skill directory, so installing only SKILL.md would break
    references.  This function copies the entire generated skill folder into a
    canonical DynaMix directory and, when requested, into Trace2Skill's allowed
    spreadsheet skill names.
    """
    skill_md = Path(skill_md)
    if skill_md.is_dir():
        source_dir = skill_md
        skill_md = source_dir / "SKILL.md"
    else:
        source_dir = skill_md.parent
    if not skill_md.exists():
        raise FileNotFoundError(skill_md)
    root = Path(output_skills_root)
    root.mkdir(parents=True, exist_ok=True)

    canonical_dir = root / "dynamix_skills" / canonical_skill_name
    _copy_skill_folder(source_dir, canonical_dir)
    canonical_path = canonical_dir / "SKILL.md"

    installed = {canonical_skill_name: str(canonical_path)}
    compatibility = {}
    if create_trace2skill_compatibility_dirs:
        for name in TRACE2SKILL_REQUIRED_SKILL_DIRS:
            dest_dir = root / name
            _copy_skill_folder(source_dir, dest_dir)
            compatibility[name] = str(dest_dir / "SKILL.md")

    manifest = {
        "skills_root": str(root),
        "canonical_skill_name": canonical_skill_name,
        "canonical_skill_path": str(canonical_path),
        "canonical_skill_dir": str(canonical_dir),
        "trace2skill_compatibility_dirs": compatibility,
        "source_skill_md": str(skill_md),
        "source_skill_dir": str(source_dir),
        "copied_support_files": True,
    }
    (root / "dynamix_install_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
