# sgl-asap-plugin

SGLang 插件：prefill 阶段的 ASAP 式 Attention-FFN 分离。

- **A（attention）**：D 个**纯 DP** 副本（`--dp-size D`）。每个副本是独立的 WORLD，互不等待：没有每步 all-gather，没有 idle batch，没有每层 EP 集合通信。
- **F（FFN）**：一个独立的专家服务，E 个 rank，纯 EP，rank 间零通信，用 `python -m sgl_asap_plugin.ffn_serve` 启动。
- **A↔F**：CUDA IPC 静态 arena + 单边写 + 64 位 doorbell，实现在 `python/sgl_asap_plugin/transport/`（PR-03，见 [docs/PR03.md](docs/PR03.md)）。

本分支 `asap` 是孤儿分支，从零开始，不含旧插件（`master`，基于 attn-DP + DeepEP 的同步方案）的代码。设计与 PR 路线图在 `afd_dev/roadmap/sgl_codewalk/`（`02a`–`02d`、`03`、`03b`）。

## 状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M0 地基** | 插件骨架、hook 装配与自检、`--asap-*` 配置、纯 DP 校验、F 入口骨架与布局计算、验收脚本 | 完成。验收标准见 [docs/M0.md](docs/M0.md)，结果见 [docs/M0_RESULTS.md](docs/M0_RESULTS.md) |
| **M1 正确性** | PR-03 传输库、PR-04 A 侧 RemoteMoE、PR-05 F 服务、PR-06 单 A + F 逐 token 对齐 | 进行中。PR-03 完成：验收标准见 [docs/PR03.md](docs/PR03.md)，结果见 [docs/PR03_RESULTS.md](docs/PR03_RESULTS.md)。PR-04 起未开始 |

## 快速开始

```bash
source /root/paddlejob/inference-public/lhd/sgl_afd/.venv/bin/activate
uv pip install -e . --no-build-isolation      # 注册 sglang.srt.plugins 入口点
export SGLANG_PLUGINS=sgl_asap_plugin         # 只加载本插件（venv 里还装着旧的 sgl_afd_plugin）

# A：2 个纯 DP 副本（DSV4 截 4 层 + dummy 权重，2 张卡）
GPUS=0,1 D=2 T=1 bash scripts/launch_attn_dsv4_tiny.sh

# F：校验并打印布局（不启动）
python -m sgl_asap_plugin.ffn_serve --model-path $MODEL --tp-size 4 --ep-size 4 \
    --asap-num-replicas 4 --asap-replica-tp 1 --asap-dry-run

# 验收
python scripts/m0_acceptance.py --gpus 0,1 --control-gpus 2,3
python scripts/pr03_acceptance.py --gpus 0,1,2,3      # 传输库，需要 4 张卡

# 选项清单（sglang serve --help 不加载插件，要用 launch_server）
python -m sglang.launch_server --help | grep asap
```

## 配置

| 选项 | 环境变量 | 说明 |
|---|---|---|
| `--asap-role {attn,ffn}` | `ASAP_ROLE` | 角色。`attn` 用 `sglang serve`，`ffn` 用 `ffn_serve` |
| `--asap-rdv-addr HOST:PORT` | `ASAP_RDV_ADDR` | F 的握手地址，默认 `127.0.0.1:9600`（要在 10000 以下：本机把 10000-61000 分给外连的源端口，固定端口会被抢） |
| `--asap-n-max N` | `ASAP_N_MAX` | 每个 micro-batch 每个 TP 行的最大 token 数，决定 arena 大小，默认 16384 |
| `--asap-slots S` | `ASAP_SLOTS` | 每个 A rank 的在途 micro-batch 槽位数，默认 2 |
| `--asap-num-replicas D` | `ASAP_NUM_REPLICAS` | A 副本数。F 必填；A 上必须等于 `--dp-size` |
| `--asap-replica-tp T` | `ASAP_REPLICA_TP` | 单个 A 副本的 TP。F 必填；A 上必须等于 `--tp-size` |
| `--asap-ffn-ranks E` | `ASAP_FFN_RANKS` | F 的 rank 数。**给了它，A 才进入 AFD 模式**（专家在 F 上）；不给则专家留在本地，只验证纯 DP 拓扑（M0 的行为）。F 上必须等于 `--ep-size`。F 的第 f 个 rank 监听 `--asap-rdv-addr` 的端口 + f |

启动进程解析配置后导出为 `ASAP_*` 环境变量，子进程（DPC、scheduler）继承。

## Hook 表

hook 都声明在 `python/sgl_asap_plugin/hooks/`，每个带目标路径和期望的参数名。注册前由 `compat.preflight` 校验：目标改名或签名漂移时直接报错，不会像 sglang 的加载器那样只记一条日志就静默失效。

| 目标 | 类型 | 层级 | 作用 |
|---|---|---|---|
| `sglang.srt.server_args.ServerArgs.add_cli_args` | AFTER | base | 增加 `--asap-*` 选项 |
| `sglang.srt.server_args.ServerArgs.from_cli_args` | BEFORE | base | 解析到 `--asap-role` 时激活 role 层并导出配置，早于 ServerArgs 解析 |
| `sglang.srt.arg_groups.validation_hook.check_server_args` | BEFORE | role | 纯 DP 规则校验，早于 sglang 自带检查 |
| `sglang.srt.managers.scheduler.Scheduler.__init__` | AFTER | role | 记录每个 scheduler 进程的 ASAP 身份；AFD 模式下连接 F |
| `sglang.srt.layers.moe.ep_moe.layer.get_moe_impl_class` | REPLACE | afd | 返回无权重的 `AfdRemoteExperts`，不分配任何专家张量 |
| `sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward` | AROUND | afd | 持有 `AfdRemoteExperts` 的层改走 AFD 前向：gate 与 top-k 留在 A，专家交给 F |
| `sglang.srt.model_loader.weight_utils.buffered_multi_thread_safetensors_weights_iterator` | REPLACE | afd | 读 checkpoint 时跳过路由专家张量（约 147 GB 中的绝大部分不读） |
| `sglang.srt.model_loader.weight_utils.safetensors_weights_iterator` | REPLACE | afd | 同上，单线程路径 |

**激活与失败语义**：base 层每个进程都装，只加选项。afd 层只在给了 `--asap-ffn-ranks` 的 A 角色里装。role 层只在 ASAP 部署里装：启动进程看解析后的 CLI，子进程看继承的 `ASAP_ROLE`。用户选择启用后，失败一律响亮：入口点里用 `SystemExit`（sglang 的插件加载器只吞 `Exception`），启动进程里直接抛异常。

## 约束与目录

A、F 的校验规则、目录结构、新增 hook 的步骤见 [docs/M0.md](docs/M0.md)。
