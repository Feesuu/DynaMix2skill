# Nodebank Task-Type Retrieval Experiment

本文档说明当前 DynaMix nodebank 版本的检索协议、运行脚本和远端复现实验参数。目标是让远端服务器 agent 不需要猜测流程，也能按同一套协议执行实验。

## 本次改动

heldout 阶段的 nodebank 检索 query 从只使用 `instruction` 改为：

```text
{instruction}

Task type: {instruction_type}
```

实现位置：

```text
spreadsheet_agent/agents/cli_skill_preloaded_agent.py
```

注意：检索 query 仍然不使用 `answer_position`。`answer_position` 只保留在 SpreadsheetBench generation prompt 中，作为官方任务约束和评测范围提示。

## 为什么这样做

`instruction_type` 只有两类：

```text
Cell-Level Manipulation
Sheet-Level Manipulation
```

它可以帮助检索区分局部单元格操作和整片 sheet/range 操作，属于相对干净的任务类型信号。

相比之下，`answer_position` 可能包含 sheet 名、目标 range、具体单元格位置，例如 `E6:E13` 或 `'Report'!D3:D10`。这些信息对检索很强，但会把目标位置线索注入经验召回，不适合作为当前干净检索协议的一部分。

## Generation Prompt 协议

nodebank 检索 query 和模型真正解题的 generation prompt 是两件事。

当前 generation prompt 仍遵循 Trace2Skill / SpreadsheetBench 风格，包含：

```text
instruction
spreadsheet_content
instruction_type
answer_position
spreadsheet_path
output_path
working_directory
```

这里的 `answer_position` 是官方 SpreadsheetBench 任务约束，不是检索 query 的一部分。

## 推荐运行脚本

新增脚本：

```bash
scripts/run_nodebank_train200_heldout.sh
```

默认流程：

1. 使用 train split `0..200` 跑 Trace2Skill train rollout。
2. 使用 official evaluator 评估 train 输出。
3. 从 train logs/eval 中抽取 `records.json`。
4. 构建 DynaMix tree 和 nodebank。
5. heldout split `200..400` 使用 nodebank top-k 检索经验。
6. 评估 heldout 输出。
7. 写出完整 logs、stage markers、summary 和 selection records。

默认直接运行：

```bash
cd /mnt/data/yaodong/codes/DynaMix2skill
./scripts/run_nodebank_train200_heldout.sh
```

远端常用覆盖方式：

```bash
cd /mnt/data/yaodong/codes/DynaMix2skill

RUN_NAME=qwen35_train200_tasktype_top10_001 \
WORKERS=4 \
MODEL=Qwen3.5-9B \
OPENAI_BASE_URL=http://127.0.0.1:18002/v1 \
EMBEDDING_BASE_URL=http://127.0.0.1:18003/v1 \
THINKING=false \
MAX_TURNS=30 \
SKILLBANK_TOP_K=10 \
./scripts/run_nodebank_train200_heldout.sh
```

如果远端模型服务是 Qwen3-8B-AWQ：

```bash
cd /mnt/data/yaodong/codes/DynaMix2skill

RUN_NAME=qwen3_8b_awq_train200_tasktype_top10_001 \
WORKERS=4 \
MODEL=Qwen3-8B-AWQ \
OPENAI_BASE_URL=http://10.20.56.11:18002/v1 \
EMBEDDING_BASE_URL=http://127.0.0.1:18003/v1 \
THINKING=false \
MAX_TURNS=30 \
SKILLBANK_TOP_K=10 \
./scripts/run_nodebank_train200_heldout.sh
```

## 参数解释

`REPO_ROOT`：DynaMix2skill 仓库路径。默认：

```text
/mnt/data/yaodong/codes/DynaMix2skill
```

`CONDA_ENV`：实验使用的 conda 环境。默认：

```text
/home/yaodong/miniconda3/envs/stableskill-skillrl
```

`DYNAMIX_PYTHON`：所有阶段使用的 Python。脚本会把它所在目录加到 `PATH`，保证 agent bash action 里裸 `python` 也是这个环境。

`DATA_PATH`：SpreadsheetBench 数据目录，必须包含 `dataset.json` 和 `spreadsheet/`。

`RUN_DIR`：本次实验输出目录。如果不显式设置，会根据 `RUN_NAME` 自动写到 `runs/` 下。

`TRAIN_START` / `TRAIN_END`：train split，左闭右开。默认 `0..200`。

`HELDOUT_START` / `HELDOUT_END`：heldout split，左闭右开。默认 `200..400`。

`WORKERS`：并发数，用于 train rollout、tree build 中的 generation/embedding 并发，以及 heldout rollout。应根据 vLLM 实际并发能力设置。

`MODEL`：generation 模型名，传给 OpenAI-compatible chat endpoint。

`OPENAI_BASE_URL`：generation 服务地址。

`OPENAI_API_KEY`：generation 服务 API key。本地 vLLM 通常为 `EMPTY`。

`EMBEDDING_BASE_URL`：embedding 服务地址，用于轨迹/节点 embedding 和 heldout 检索。

`EMBEDDING_MODEL`：embedding 模型名。默认 `Qwen3-Embedding-8B`。

`EMBEDDING_TOKENIZER`：embedding tokenizer 路径，用于 token 统计和长文本截断。

`MAX_TURNS`：每条 spreadsheet task 的最大 ReAct turn 数。

`THINKING`：统一 thinking 设置，允许值：

```text
true
false
null
```

`SKILLBANK_TOP_K`：heldout 每条任务检索并注入的 nodebank 经验数量。

`RESUME`：是否复用当前 `RUN_DIR` 中已经完成的 stage marker。默认 `true`。如果需要在同一目录强制重跑，设置：

```bash
RESUME=false
```

## 输出文件

假设 `RUN_DIR=/mnt/data/yaodong/codes/DynaMix2skill/runs/example_run`，关键输出如下：

```text
experiment_runtime_config.json
experiment_summary.json
split_manifest.json
trace2skill_generation_config.json
trace2skill_train_results.json
trace2skill_train_eval.json
records.json
dynamix_config.json
dynamix_tree/summary.json
dynamix_tree/skills/node_bank_manifest.json
trace2skill_heldout_results.json
trace2skill_heldout_eval.json
raw/skill_selection_records.jsonl
logs/experiment_wrapper.log
logs/01_train_collect.log
logs/02_train_eval.log
logs/03_extract_records.log
logs/04_build_tree.log
logs/06_heldout_collect.log
logs/07_heldout_eval.log
stage_markers/*.done
```

其中最重要的审计文件是：

```text
raw/skill_selection_records.jsonl
```

它记录每条 heldout task 的最终检索 query、top-k 节点 ID、相似度分数和节点元信息。检查这里可以确认 query 是否确实为：

```text
instruction + "\n\nTask type: " + instruction_type
```

## 快速核查命令

检查 Python 环境：

```bash
export CONDA_ENV=/home/yaodong/miniconda3/envs/stableskill-skillrl
export DYNAMIX_PYTHON=$CONDA_ENV/bin/python
export PATH=$CONDA_ENV/bin:$PATH

which python
python -c "import sys; print(sys.executable)"
```

检查脚本语法：

```bash
bash -n scripts/run_nodebank_train200_heldout.sh
```

检查检索 query：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/YOUR_RUN/raw/skill_selection_records.jsonl")
first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
print(first["query"])
PY
```

检查 heldout 分数：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/YOUR_RUN/trace2skill_heldout_eval.json")
data = json.loads(path.read_text(encoding="utf-8"))

if isinstance(data, dict) and "summary" in data:
    print(json.dumps(data["summary"], indent=2, ensure_ascii=False))
else:
    passed = sum(1 for x in data if x.get("hard_restriction") or x.get("passed"))
    print("passed:", passed, "total:", len(data), "acc:", passed / max(1, len(data)))
PY
```

## 实验对比建议

为了判断 `Task type` 是否有帮助，建议至少保留三组结果：

```text
A: query = instruction
B: query = instruction + Task type
C: query = instruction + Task type + answer_position
```

当前代码实现的是 B。A 是旧的 instruction-only 干净协议；C 可能更高分，但会引入目标位置线索，不建议作为主结果，只适合作为上界或诊断对比。

比较结果时不要只看最终准确率，还要同时检查：

```text
heldout runner success 数量
超时/异常数量
SyntaxError 数量
raw/skill_selection_records.jsonl 中 top-k 节点是否相关
trace2skill_heldout_eval.json 的失败类型
是否经过 LibreOffice recalc 后复评
```
