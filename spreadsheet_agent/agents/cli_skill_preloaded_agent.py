"""
CLI Skill Preloaded Agent - Bash CLI agent with skill content pre-loaded in context.

Unlike CLISkillAgent which discovers skills and instructs the agent to read them
on demand via bash, this agent reads all skill content at initialization and embeds
it directly into the system prompt. The agent does not need to decide whether or
when to read a skill file — the full guidance is already available in context.
"""

import json
import os
from datetime import datetime
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from react_agent import Tool

try:
    from dynamix_trace2skill.skillbank import SkillBankSelector, selected_skills_to_system_content
except Exception:  # pragma: no cover - DynaMix extension is optional for vanilla Trace2Skill use.
    SkillBankSelector = None  # type: ignore[assignment]
    selected_skills_to_system_content = None  # type: ignore[assignment]

from .base import BaseSpreadsheetAgent
from ..tools import create_bash_tool
from ..system_prompts import render_full_system_prompt


SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")


@dataclass
class SkillMetadata:
    name: str
    description: str
    file_path: str


def extract_skill_metadata(skill_file: str) -> SkillMetadata | None:
    try:
        with open(skill_file, "r", encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return None

    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        return None

    frontmatter = frontmatter_match.group(1)
    name_match = re.search(r'^name:\s*["\']?([^"\'\n]+)["\']?\s*$', frontmatter, re.MULTILINE)
    desc_match = re.search(
        r'^description:\s*["\']?([^"\'\n]+)["\']?\s*$',
        frontmatter,
        re.MULTILINE,
    )
    if not name_match:
        return None

    return SkillMetadata(
        name=name_match.group(1).strip(),
        description=desc_match.group(1).strip() if desc_match else "",
        file_path=skill_file,
    )


def discover_skills(skills_dir: str) -> list[SkillMetadata]:
    skills = []
    if not os.path.exists(skills_dir):
        return skills

    for entry in sorted(os.listdir(skills_dir)):
        skill_dir = os.path.join(skills_dir, entry)
        skill_file = os.path.join(skill_dir, "SKILL.md")
        if not (os.path.isdir(skill_dir) and os.path.exists(skill_file)):
            continue
        metadata = extract_skill_metadata(skill_file)
        if metadata is not None:
            skills.append(metadata)
    return skills


def read_skill_content(skill: SkillMetadata) -> str:
    """Read the full content of a skill file, stripping YAML frontmatter."""
    try:
        with open(skill.file_path, "r") as f:
            content = f.read()

        # Strip YAML frontmatter (already parsed into metadata)
        if content.startswith("---"):
            end = content.index("---", 3)
            content = content[end + 3:].lstrip("\n")

        return content
    except Exception:
        return ""


def render_preloaded_skills_section(skills: list[tuple[SkillMetadata, str]], skills_dir: str) -> str:
    """
    Render the skills section with full skill content embedded.

    Args:
        skills: List of (metadata, content) tuples for each loaded skill.
        skills_dir: Absolute path to the skills directory.
    """
    if not skills:
        return ""

    lines = [
        "## Skills",
        "",
        "The following skills have been loaded for this session. Their full guidance is included below.",
        "",
    ]

    for metadata, content in skills:
        lines.extend([
            f"### {metadata.name}",
            "",
            metadata.description,
            "",
            "---",
            "",
            content,
            "",
            "---",
            "",
        ])

    lines.extend([
        "### Skill Usage Rules",
        "",
        "**CRITICAL RULE**: If a skill above is relevant to your task and contains useful guidance",
        "for the operation you need to perform, you MUST follow the skill's instructions. Only act",
        "on your own judgment if:",
        "- No skill is relevant to the task, OR",
        "- The skill does not cover the specific operation you need to perform",
        "",
        "**Skill Authority**: When a skill has guidance for your operation, its instructions take",
        "precedence over your general knowledge.",
        "",
        f"**Resources**: Scripts and other resources referenced in a skill are located in the skill's directory under `{skills_dir}`. Use the full path when running them (e.g., `python {skills_dir}/xlsx/recalc.py`).",
        "",
    ])

    return "\n".join(lines)


SPREADSHEET_SKILL_PRELOADED_CONTEXT = """You have a **bash** action to execute shell commands. Use it to run Python code.

{skills_section}"""


SPREADSHEET_SKILL_PRELOADED_EXAMPLES = """## Recommended Workflow

1. **Analyze**: Read the instruction and spreadsheet_content to understand what needs to be done
2. **Apply Skill Guidance**: Review the skill content loaded above. If any skill covers your operation, follow its guidance
3. **Execute**: Write and run Python code following the skill's guidance when applicable
4. **Verify**: Check that the output file was created at the exact output_path
5. **Complete**: Signal task completion with ACTION: TASK_COMPLETE

**IMPORTANT**: Skill guidance is already loaded in your context above. Follow it when it covers your operation. Only use your own approach when no loaded skill covers your specific task.

## Action Examples

### Execute Python code (following loaded skill guidance when applicable):

Action:
{{
    "name": "bash",
    "arguments": {{"command": "python -c \"import openpyxl; wb = openpyxl.load_workbook('/path/to/input.xlsx'); ws = wb.active; ws['D2'] = '=SUM(B2:C2)'; wb.save('/path/to/output.xlsx'); print('Done')\""}}
}}

### Write and execute a solution script:

Action:
{{
    "name": "bash",
    "arguments": {{"command": "cat <<'EOF' > solution.py\\nimport openpyxl\\nwb = openpyxl.load_workbook('/path/to/input.xlsx')\\nws = wb.active\\n# Your manipulation logic here\\nwb.save('/path/to/output.xlsx')\\nprint('Saved')\\nEOF\\npython solution.py"}}
}}

### Verify output file:

Action:
{{
    "name": "bash",
    "arguments": {{"command": "ls -la /path/to/output.xlsx"}}
}}

### Signal task completion:

When you have successfully created the output file:

ACTION: TASK_COMPLETE

Note: The above examples are just reference actions for inspiration. You should adapt your actions based on context and take any action that you deem appropriate.

Action:
{{
    "name": "bash",
    "arguments": {{"command": "# Any other command you deem appropriate"}}
}}"""


class CLISkillPreloadedAgent(BaseSpreadsheetAgent):
    """
    CLI agent with skill content pre-loaded in the system prompt.

    Unlike CLISkillAgent which lists available skills and instructs the agent to
    read them on demand via bash, this agent reads all skill content at
    initialization and embeds it directly into the context. The agent follows
    skill guidance without needing to decide whether or when to read a skill file.

    Actions:
    - bash: Shell command execution

    Skills:
    - Discovered and fully loaded at initialization
    - Content embedded in system prompt, no runtime file reads needed
    """

    def __init__(
        self,
        client,
        skills_dir: str | None = None,
        max_turns: int = 20,
        temperature: float = 0.0,
        verbose: bool = True,
        timeout: int = 120,
        log_dir: str | None = None,
        log_format: str = "markdown",
    ):
        super().__init__(client, max_turns, temperature, verbose, log_dir, log_format)
        self.timeout = timeout
        self.skills_dir = os.path.abspath(skills_dir if skills_dir is not None else SKILLS_DIR)

        # Discover skills and load their full content at initialization.
        # DynaMix may optionally enable task-conditioned skillbank selection via
        # environment variables; vanilla Trace2Skill behavior remains unchanged
        # when DYNAMIX_SKILLBANK_TOP_K is unset or <= 0.
        metadata_list = discover_skills(self.skills_dir)
        if not metadata_list:
            raise ValueError(
                f"No skills discovered in skills_dir: {self.skills_dir}"
            )
        self._all_skills: list[tuple[SkillMetadata, str]] = [
            (meta, read_skill_content(meta)) for meta in metadata_list
        ]
        self._skills: list[tuple[SkillMetadata, str]] = list(self._all_skills)
        self._active_skill_selection: list[dict] = []
        self._skillbank_selector = None
        try:
            self._skillbank_top_k = int(os.getenv("DYNAMIX_SKILLBANK_TOP_K", "0") or "0")
        except ValueError:
            self._skillbank_top_k = 0
        if self._skillbank_top_k > 0:
            if SkillBankSelector is None:
                raise RuntimeError("DYNAMIX_SKILLBANK_TOP_K was set but dynamix_trace2skill.skillbank is unavailable")
            skillbank_root = os.getenv("DYNAMIX_SKILLBANK_ROOT", self.skills_dir)
            self._skillbank_selector = SkillBankSelector.from_env(default_skillbank_root=skillbank_root)

    @property
    def name(self) -> str:
        return "cli_skill_preloaded_agent"

    def get_system_prompt(self) -> str:
        """Legacy method - kept for backward compatibility."""
        skills_section = render_preloaded_skills_section(self._skills, self.skills_dir)
        return SPREADSHEET_SKILL_PRELOADED_CONTEXT.format(skills_section=skills_section)

    def get_system_template(self) -> str:
        # The official v1 template has a single {skill_content} slot.  DynaMix
        # skillbank selection concatenates the selected top-k SKILL.md files into
        # that slot, so the rest of the Trace2Skill prompt/action protocol stays
        # unchanged.
        active_skills = self._skills or self._all_skills
        if active_skills:
            sections: list[str] = []
            for index, (metadata, content) in enumerate(active_skills, start=1):
                skill_dir = os.path.dirname(os.path.abspath(metadata.file_path))
                sections.extend([
                    f"## Preloaded Skill {index}: {metadata.name}",
                    "",
                    metadata.description,
                    "",
                    f"Skill directory: `{skill_dir}`",
                    f"References directory: `{os.path.join(skill_dir, 'references')}`",
                    f"Scripts directory: `{os.path.join(skill_dir, 'scripts')}`",
                    "Use absolute paths under this skill directory if the skill tells you to inspect references or run helper scripts.",
                    "",
                    content,
                    "",
                ])
            skill_content = "\n".join(sections).strip()
            skill_dir = self.skills_dir
        else:
            skill_content = "(No skill loaded)"
            skill_dir = self.skills_dir

        return render_full_system_prompt(
            "cli_skill_preloaded_full_system_v1.txt",
            skill_content=skill_content,
            skill_dir=skill_dir,
        )

    def _select_skills_for_context(self, context) -> None:
        if self._skillbank_top_k <= 0 or self._skillbank_selector is None:
            self._skills = list(self._all_skills)
            self._active_skill_selection = []
            return
        query = "\n".join([
            str(getattr(context, "instruction", "") or ""),
            str(getattr(context, "instruction_type", "") or ""),
            str(getattr(context, "answer_position", "") or ""),
        ]).strip()
        selections = self._skillbank_selector.select(query, top_k=self._skillbank_top_k)
        selected_pairs: list[tuple[SkillMetadata, str]] = []
        active_manifest: list[dict] = []
        for selection in selections:
            doc = selection.skill
            # Concurrency-safe selection: do NOT copy or mutate shared
            # skills_dir per query.  The skillbank is a run-level immutable root;
            # each selected skill keeps its fixed absolute skill_dir so the agent
            # can read references/ and scripts/ exactly as in Trace2Skill.
            skill_path = os.path.abspath(doc.skill_path)
            skill_dir = os.path.abspath(doc.skill_dir)
            if not os.path.isfile(skill_path):
                raise FileNotFoundError(f"selected SKILL.md does not exist: {skill_path}")
            metadata = SkillMetadata(name=doc.name, description=doc.description, file_path=skill_path)
            content = read_skill_content(metadata)
            selected_pairs.append((metadata, content))
            active_manifest.append({
                "skill_id": doc.skill_id,
                "name": doc.name,
                "score": selection.score,
                "skill_path": skill_path,
                "skill_dir": skill_dir,
                "references_dir": os.path.join(skill_dir, "references"),
                "scripts_dir": os.path.join(skill_dir, "scripts"),
            })
        self._skills = selected_pairs
        self._active_skill_selection = active_manifest
        self._write_skill_selection_record(context=context, query=query, selected=active_manifest)
        # Force ReActAgent/system prompt rebuild for this task's selected skills.
        self._agent = None

    def _write_skill_selection_record(self, *, context, query: str, selected: list[dict]) -> None:
        path = os.getenv("DYNAMIX_SKILL_SELECTION_LOG")
        if not path:
            return
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "instance_id": str(getattr(context, "instance_id", "") or ""),
            "instruction": str(getattr(context, "instruction", "") or ""),
            "instruction_type": str(getattr(context, "instruction_type", "") or ""),
            "answer_position": str(getattr(context, "answer_position", "") or ""),
            "query": query,
            "top_k": self._skillbank_top_k,
            "selected_skill_ids": [item["skill_id"] for item in selected],
            "selected_skill_scores": [item["score"] for item in selected],
            "selected_skill_dirs": [item["skill_dir"] for item in selected],
            "selected_skill_paths": [item["skill_path"] for item in selected],
            "selected": selected,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as handle:
            try:
                import fcntl  # type: ignore
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(line)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                handle.write(line)

    def run(self, context):
        self._select_skills_for_context(context)
        return super().run(context)

    def create_tools(self, working_dir: str) -> list[Tool]:
        return [
            create_bash_tool(working_dir, timeout=self.timeout),
        ]

    def build_task_prompt(self, context) -> str:
        """Build task prompt with absolute paths.

        Unlike CLISkillAgent, this does not include a skills_directory field
        since skill content is already embedded in the system prompt.
        """
        working_dir = os.path.abspath(context.working_dir)
        input_file = os.path.abspath(context.input_file)
        output_file = os.path.abspath(context.output_file)

        return f"""Below is the spreadsheet manipulation question you need to solve:

### working_directory
{working_dir}

### instruction
{context.instruction}

### spreadsheet_path
{input_file}

### spreadsheet_content
{context.spreadsheet_content}

### instruction_type
{context.instruction_type}

### answer_position
{context.answer_position}

### output_path
{output_file}

---
**REMINDER**: Write files ONLY in `{working_dir}`. Save output to exact path: `{output_file}`
---

Solve the question and save the modified spreadsheet to the exact output_path shown above."""

    def get_available_skills(self) -> list[SkillMetadata]:
        """Get list of discovered skills with their loaded content."""
        return [meta for meta, _ in self._skills]
