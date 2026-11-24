# CAC 论文与中期答辩 · PPO + MPC

本目录归档论文 *Bidirectional PPO and MPC Switching Policy for Artificial Pancreas Control via Risk-State and Insulin-on-Board Prediction* 的正式仿真版本，也是中期答辩中 PPO 和风险切换实验的代码来源。研究使用 `simglucose` 虚拟患者。

| 要做什么 | 文件或目录 |
| --- | --- |
| 查看风险双向切换与 IOB 预测 | [`ppo_mpc/controllers.py`](ppo_mpc/controllers.py) |
| 查看 PPO 与 ARX-RLS Zone-MPC 实现 | [`ppo_mpc/ppo.py`](ppo_mpc/ppo.py) · [`mpc_assets/ARX-RLS/arx_mpc_ctrller.py`](mpc_assets/ARX-RLS/arx_mpc_ctrller.py) |
| 训练 PPO | [`scripts/train_ppo.py`](scripts/train_ppo.py) |
| 跑论文评估 | [`reproduce_formal.py`](reproduce_formal.py) · [`scripts/evaluate_ppo_mpc.py`](scripts/evaluate_ppo_mpc.py) |
| 使用正式训练产物 | [最终权重、训练配置及日志](runs/ppo_adult001_005_unann_tanh_v2_2m/) |
| 查论文参数、各次运行指标及患者级统计 | [正式结果](results/finall/formal_iob_prediction_adolescent001_010_72h_r5/) |
| 查审稿补充实验 | [阈值敏感性](results/finall/reviewer2_threshold_sensitivity_72h_r3/) · [闭环安全与稳定性分析](results/finall/reviewer2_safety_stability_formal_72h_r5/) |

## 运行论文实验

从仓库根目录进入 `cac-ppo-mpc`。正式评估使用 Linux x86-64、Python 3.10.15 和随附的依赖锁。正式实验以 CPU 运行，依赖锁记录原服务器使用的 PyTorch 发行版本。

```sh
cd cac-ppo-mpc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock-server-py310-cu126.txt
python -m scripts.verify_runtime
python reproduce_formal.py --dry-run
python reproduce_formal.py --smoke
python reproduce_formal.py
```

快速检查使用 1 个虚拟患者、每种方法 1 次、每次 1 小时。正式运行使用 10 个青少年虚拟患者、每种方法各 5 次、每次计划 72 小时。评估入口校验源码、权重、ARX 参数和依赖锁的 SHA-256。复跑结果默认写入 `reproductions/`，可用 `--output-dir` 指定新目录。

历史正式记录的 100 个任务全部完成处理，其中 11 个提前终止。归档保留配置、逐次指标和论文汇总表。逐步轨迹和决策日志可由评估脚本重新生成。阈值敏感性结果附有各设置的汇总表和制图脚本。复算安全与稳定性分析时，需先生成正式评估的逐步轨迹和决策日志。

## 重新训练 PPO

下面的命令沿用归档训练配置的对象、随机种子、训练步数和 PPO 超参数。日志和新权重写入 `runs/ppo_retrain/`。

```sh
python -m scripts.train_ppo \
  --patients 'adult#001,adult#002,adult#003,adult#004,adult#005' \
  --scenario train --meal-announcement unannounced --sensor-name GuardianRT \
  --steps 288 --seed 61000 --total-steps 2000000 --steps-per-update 2048 \
  --hidden-dim 128 --minibatch-size 256 --update-epochs 10 \
  --policy-lr 0.0003 --value-lr 0.001 --entropy-coef 0.01 \
  --max-basal-multiplier 5.0 --save-every-updates 10 --device cpu \
  --output-dir runs/ppo_retrain
```

原权重的训练源码指纹与论文评估快照的源码指纹不同。复现论文指标时，评估入口使用归档的 `model_final.pt`。训练目录另附当时的配置和完整训练日志。

## 归档说明

`ppo_mpc/`、`scripts/` 和 `mpc_assets/` 中纳入正式评估源码指纹的文件按原路径和字节保存。正式配置记录中的服务器绝对路径已改为相对路径。附带的 [simglucose](https://github.com/jxx123/simglucose) 含项目内的适配，原作者的 [MIT 许可](mpc_assets/simglucose/LICENSE) 随源码保留。
