# Large artifacts (not in git)

sha256 manifests are in `manifests/`. Put each artifact at the path shown (relative to the repo root) to use it.

| Artifact | Size | Path in this repo | Where it is | Rebuild with |
|---|---|---|---|---|
| **DP checkpoint, step 80k** (EMA weights; the reported model) | 1.1 GB | `outputs/dp/multitask_v1/checkpoints/step0080000/state.pt` | private HF repo [`Ctrlgb/dp-imbench-edge-slide-multitask-step80k`](https://huggingface.co/Ctrlgb/dp-imbench-edge-slide-multitask-step80k) (sha256-verified) | `scripts/dp/train_dp.py` (args in `train_args.json`) |
| DP checkpoints, steps 20k / 40k / 60k (EMA weights) | 1.1 GB each | `outputs/dp/multitask_v1/checkpoints/step00{2,4,6}0000/state.pt` | local disk of the training laptop | same |
| DP `last` checkpoint (model + optimizer, = step 80k, for `--resume`) | 4.4 GB | `outputs/dp/multitask_v1/checkpoints/last/state.pt` | local disk of the training laptop | same |
| DP dataset `imbench/dp_multitask_20hz` (LeRobot v3.0, 3453 eps) | 1.5 GB | `demos_synthetic/dp/lerobot/` | same | README step D1–D2 (metadata is already in this repo) |
| DP JPEG training cache (128 px) | 6.0 GB | `demos_synthetic/dp/cache128/` | derived | README step D3 |
| **MolmoAct2 fine-tuned model, step 3000** | 51 GB (inference part ~22 GB) | anywhere; pass `--model <dir>` | private HF repo [`Ctrlgb/molmoact2-imbench-edge-slide-step3000`](https://huggingface.co/Ctrlgb/molmoact2-imbench-edge-slide-step3000) | `training/molmoact2/imbench_ft/train.sh` + `convert.sh` |
| MolmoAct2 dataset `imbench/edge_slide_teleop_20hz` (LeRobot v3.0, 150 eps) | 158 MB | `demos_synthetic/molmoact2/lerobot/` | local disk | `demos_synthetic/molmoact2/run_pipeline.sh` (metadata is already in this repo) |

Checkpoint folders in git keep `config.json`, `stats.json`, `train_args.json` and `PRUNED`. Only `state.pt` is left out.
Both datasets are deterministic in their seeds: re-recording seed 200000 reproduced the original episode bit for bit.
