# MobiAgent – AI Agent Playbook

## Project Shape
- Core pieces: [runner/](runner/README.md) (device control + agent loop), [MobiFlow/](MobiFlow/README.md) (DAG-based offline evaluator), [collect/](collect/README.md) (data collection/annotation/build), [agent_rr/](agent_rr/README.md) (retrieval/rerank acceleration), [app/](app/README.md) (Android client), utilities consolidated under [utils/](utils/README.md).
- Models: MobiMind family (Decider/Grounder/Planner or Mixed/Reasoning unified endpoints) served via vLLM; most scripts assume OpenAI-compatible HTTP.
- Weights/configs live under `weights/`; OCR/icon tools now unified in utils (see utils refactor notes).

## Environment & Dependencies
- Python 3.10 recommended; minimal deps `pip install -r requirements_simple.txt`, full stack `pip install -r requirements.txt` and download OmniParser + embedding weights as shown in [README.md](README.md#quick-start).
- Optional OCR accel: install `paddlepaddle-gpu` matching CUDA; tesseract for CPU OCR.
- Device prep: enable ADB, install ADBKeyboard on Android; Harmony supported via device flag.

## Serving Models (vLLM)
- Decider/Grounder/Planner split: `vllm serve IPADS-SAI/MobiMind-Decider-7B --port <decider>`, `...Grounder-3B --port <grounder>`, `Qwen/Qwen3-4B-Instruct --port <planner>`.
- Mixed/Reasoning Qwen3-based: point both decider/grounder to the Mixed port; add `--dtype float16` when serving AWQ quant.

## Running the Agent (single-task)
- Edit tasks in [runner/mobiagent/task.json](runner/mobiagent/task.json) then launch:
  - `python -m runner.mobiagent.mobiagent --service_ip <ip> --decider_port <p> --grounder_port <p> --planner_port <p> --device Android|Harmony --use_qwen3 --use_experience --user_profile on|off --use_graphrag on|off [--clear_memory]`.
- `--use_qwen3` defaults true; keep on when planner/grounder returns 0-1000 coords for proper pixel mapping.
- Preference memory: Mem0 vector store by default; GraphRAG via Neo4j when `--use_graphrag on`. Required env: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MILVUS_URL`, `EMBEDDING_MODEL`, `EMBEDDING_MODEL_DIMS`, `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.

## Multi-Task Runner
- For cross-app workflows use [runner/mobiagent/multi_task/](runner/mobiagent/multi_task/README.md): planner analyzes/creates Plan, executor loops Decider/Grounder per subtask, artifacts refined between subtasks. Start via `python -m runner.mobiagent.multi_task.mobiagent_refactored` with the same ports plus OCR/experience flags (see README for sample command and ExtractArtifactConfig options).

## Data Pipeline (collect)
- Manual UI recorder: `python -m collect.manual.server` then operate via http://localhost:9000; outputs per-step screenshots + actions.json under `collect/manual/data/...`.
- Automatic collector: populate [collect/auto/task.json](collect/auto/task.json) and run `python -m collect.auto.server --model <name> --api_key <key> --base_url <url> [--max_steps 15]`; saves raw logs and normalized trajectories.
- Annotation: `python -m collect.annotate --data_path <dir> --model <name> --api_key <key> --base_url <url>` produces `react.json` with reasoning overlays.
- SFT construction: `python -m collect.construct_sft --data_path <raw> --ss_data_path <single_step> --unexpected_img_path <popups> --out_path <out> [--use_qwen3]`; supports augmentation via [collect/augment_config.json](collect/augment_config.json).

## Evaluation (MobiFlow)
- Offline DAG verifier: configs in [MobiFlow/task_configs/](MobiFlow/task_configs/), icons in `task_configs/icons`. Run `python -m avdag.verifier task_configs/<task>.json <trace_folder>` to check recorded traces (screenshots/XML/actions/react).
- Advanced checks: escalate/juxtaposition/dynamic_match, icon detection via OpenCV templates; see [MobiFlow/README.md](MobiFlow/README.md) for condition semantics.

## AgentRR Acceleration
- Prepare templates/data via `python -m train.prepare_data ...`; train embedder/reranker with ms-swift; run experiments `python run_experiment.py --data_path ... --embedder_path ... --reranker_path ... --ditribution uniform|power_law` (typo in flag preserved).

## Utilities & Weights
- OCR/icon detection now under [utils/](utils/README.md); import via `from utils.tools import OCREngine, get_icon_detection_service`. Weights managed by `utils/weights_manager.py`; default icon/OCR models under `weights/`.

## Android App
- Android client lives in [app/](app/README.md): build with Android Studio (API 26–34, Gradle 8.3); permissions include accessibility and media projection. Configure server URL in `MainActivity.java` to point at deployed MobiAgent server.

## Tips & Debugging
- If actions misplace, confirm `--use_qwen3` alignment and device resolution; check planner/grounder ports match served model.
- For memory issues, start with `--user_profile off` and disable GraphRAG; re-enable after verifying Milvus/Neo4j connectivity.
- Large batch runs generate many screenshots; ensure disk space and periodically prune `data/` directories.

## Quick Test Hooks
- Minimal smoke: run single task with `python -m runner.mobiagent.mobiagent --task_file runner/mobiagent/task.json --decider_port 8000 --grounder_port 8000 --planner_port 8080 --use_qwen3` after serving a Mixed model.
- Evaluate traces with `python -m avdag.verifier ...` and inspect icon/OCR outputs when a node fails.