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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from react_agent import Tool

try:
    from dynamix_trace2skill.skillbank import SkillBankSelector, selected_experience_to_system_content
except Exception:  # pragma: no cover - DynaMix extension is optional for vanilla Trace2Skill use.
    SkillBankSelector = None  # type: ignore[assignment]
    selected_experience_to_system_content = None  # type: ignore[assignment]

from .base import BaseSpreadsheetAgent
from ..tools import create_bash_tool
from ..system_prompts import render_full_system_prompt


SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")


SPREADSHEET_SKILL_PRELOADED_EXAMPLES = """## Recommended Workflow

1. **Analyze**: Read the instruction and spreadsheet_content to understand what needs to be done
2. **Apply Retrieved Experience**: Review the retrieved experience loaded above. If it covers your operation, follow its guidance
3. **Execute**: Write and run Python code in the current directory, using `input.xlsx` and `output.xlsx`
4. **Verify**: Check that the output file was created at the exact output_path
5. **Complete**: Signal task completion with ACTION: TASK_COMPLETE

**IMPORTANT**: Retrieved experience is already loaded in your context above. Follow it when it covers your operation. Only use your own approach when no retrieved experience covers your specific task.
**PYTHON RULE**: Use `python -c` only for short read-only checks. For any workbook edit, loop, formula fill, or multi-step logic, write `solution.py` and run `python solution.py`.

## Action Examples

### Inspect workbook with simple Python:

Action:
{{
    "name": "bash",
    "arguments": {{"command": "python -c \"import openpyxl; wb = openpyxl.load_workbook('input.xlsx', data_only=True); print(wb.sheetnames); print(wb.active.dimensions)\""}}
}}

### Write and execute a solution script for workbook edits:

Action:
{{
    "name": "bash",
    "arguments": {{"command": "cat <<'EOF' > solution.py\\nimport openpyxl\\n\\nwb = openpyxl.load_workbook('input.xlsx')\\nws = wb.active\\nfor row in range(2, ws.max_row + 1):\\n    # Put task-specific workbook edits here.\\n    pass\\nwb.save('output.xlsx')\\nprint('Saved output.xlsx')\\nEOF\\npython solution.py"}}
}}

### Verify output file:

Action:
{{
    "name": "bash",
    "arguments": {{"command": "ls -la output.xlsx"}}
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

        try:
            self._skillbank_top_k = int(os.getenv("DYNAMIX_SKILLBANK_TOP_K", "0") or "0")
        except ValueError:
            self._skillbank_top_k = 0
        if self._skillbank_top_k <= 0:
            raise ValueError("DYNAMIX_SKILLBANK_TOP_K must be > 0 for nodebank retrieval")
        self._active_skill_selection: list[dict] = []
        self._dynamic_skillbank_content = ""
        self._skillbank_selector = None
        if SkillBankSelector is None:
            raise RuntimeError("dynamix_trace2skill.skillbank is unavailable")
        skillbank_root = os.getenv("DYNAMIX_SKILLBANK_ROOT", self.skills_dir)
        self._skillbank_selector = SkillBankSelector.from_env(default_skillbank_root=skillbank_root)

    @property
    def name(self) -> str:
        return "cli_skill_preloaded_agent"

    def get_system_prompt(self) -> str:
        """Legacy method - system_template is used for nodebank retrieval."""
        return ""

    def get_system_template(self) -> str:
        # The official v1 template has a single {skill_content} slot. Nodebank
        # retrieval fills it with task-conditioned experience snippets.
        if self._dynamic_skillbank_content:
            skill_content = self._dynamic_skillbank_content.strip()
            experience_root = os.getenv("DYNAMIX_SKILLBANK_ROOT", self.skills_dir)
        else:
            skill_content = "(No retrieved experience loaded)"
            experience_root = self.skills_dir

        return render_full_system_prompt(
            "cli_skill_preloaded_full_system_v1.txt",
            skill_content=skill_content,
            skill_dir=experience_root,
        )

    def _select_skills_for_context(self, context) -> None:
        if self._skillbank_top_k <= 0 or self._skillbank_selector is None:
            self._dynamic_skillbank_content = ""
            self._active_skill_selection = []
            return
        query = str(getattr(context, "instruction", "") or "").strip()
        selections = self._skillbank_selector.select(query, top_k=self._skillbank_top_k)
        active_manifest: list[dict] = []
        for selection in selections:
            doc = selection.skill
            active_manifest.append({
                "node_id": doc.node_id,
                "item_id": doc.item_id,
                "name": doc.name,
                "score": selection.score,
                "level": doc.level,
                "support_mass": doc.support_mass,
                "confidence": doc.confidence,
                "source_community_id": doc.source_community_id,
                "source_member_count": doc.source_member_count,
            })
        self._dynamic_skillbank_content = selected_experience_to_system_content(selections) if selected_experience_to_system_content else ""
        self._active_skill_selection = active_manifest
        self._write_skill_selection_record(context=context, query=query, selected=active_manifest)
        # Force ReActAgent/system prompt rebuild for this task's selected nodes.
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
            "selected_node_ids": [item["node_id"] for item in selected],
            "selected_node_scores": [item["score"] for item in selected],
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
        """Build task prompt with task-relative paths.

        Unlike CLISkillAgent, this does not include a skills_directory field
        since skill content is already embedded in the system prompt.
        """
        return f"""Below is the spreadsheet manipulation question you need to solve:

### working_directory
.

### instruction
{context.instruction}

### spreadsheet_path
input.xlsx

### spreadsheet_content
{context.spreadsheet_content}

### instruction_type
{context.instruction_type}

### answer_position
{context.answer_position}

### output_path
output.xlsx

---
**REMINDER**: Your bash commands already run inside the current task directory. Read only `input.xlsx`, save the final workbook as `output.xlsx`, and do not search `/tmp` or copy files from other task directories. If answer_position is a range, update every required cell in that range and verify representative target cells before completion.
---

Solve the question and save the modified spreadsheet to the exact output_path shown above."""

    def get_available_skills(self) -> list:
        """Nodebank retrieval does not expose local skill files."""
        return []
