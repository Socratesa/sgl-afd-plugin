# sgl-asap-plugin

SGLang 插件：prefill 阶段的 ASAP 式 Attention-FFN 分离（AFD）。

```
 A 副本 0 (GPU a0) ─┐                        ┌─ F rank 0 (GPU f0): 专家 [0, P)
 A 副本 1 (GPU a1) ─┼── 每层 MoE：x, ids, w ──┼─ F rank 1 (GPU f1): 专家 [P, 2P)
   ...              │   ←── 部分和 ──────────  │   ...
 A 副本 D-1 ────────┘   (CUDA IPC + 流内存操作) └─ F rank E-1
```

- **A（attention）**：`sglang serve --dp-size D --tp-size 1`，D 个**纯 DP** 副本，每个副本是独立的 torch.distributed world，副本之间没有任何同步。每层 MoE 的 gate、top-k、共享专家留在 A，路由专家交给 F。
- **F（expert）**：`python -m sgl_asap_plugin.ffn`，E 个 rank，纯 EP，rank 之间不通信，按到达顺序逐层服务所有副本。
- **A↔F**：静态 arena + 单边拷贝 + 64 位 doorbell + 流内存操作等待；A 的 CPU 从不等 F。内存布局与协议写在 `transport/layout.py` 的模块文档里。

设计与路线图：`afd_dev/roadmap/sgl_codewalk/`（`02a`–`02d`、`03`、`03b`）。本分支 `asap` 是孤儿分支，与 `master`（attn-DP + DeepEP 的锁步方案）无共同代码。

验收框架包含数据比较、必选项报告、测试子进程监督和负载有效性评估；说明及兼容变化见 [验收文档](docs/ACCEPTANCE.md)。无 CUDA 环境可从仓库根目录运行：

```bash
python -B -m unittest discover -s tests -p 'test_acceptance_*.py' -v
```

这组测试验证验收工具本身；模型精度、CUDA IPC 和服务部署仍需独立 GPU 验收。

## 代码

| 文件 | 内容 |
|---|---|
| `python/sgl_asap_plugin/__init__.py` | 入口点 `register()`，不导入 torch |
| `config.py` | `--asap-*` 选项、`ASAP_*` 环境变量、A/F 两个角色的部署规则、MoE 形状 |
| `hooks.py` | 所有 hook 的目标、类型、期望参数、原因（下表即 `hooks.SPECS`）；预检与激活 |
| `checkpoint.py` | 两个角色各读哪些 checkpoint 张量，按名过滤、不读的张量不读盘 |
| `attn.py` | A 侧：无权重的 `RemoteExperts`、`moe_forward`、`AttnClient`、加载完权重后连接 F |
| `ffn.py` | F 侧：`ExpertStack`（stock `FusedMoE`）、rank 进程、启动器 |
| `transport/` | `layout.py` 布局与 doorbell，`channel.py` A 端与 watchdog，`server.py` F 端，`cuda.py` 驱动调用 |

## 快速开始

```bash
source /root/paddlejob/inference-public/lhd/sgl_afd/.venv/bin/activate
uv pip install -e . --no-build-isolation      # 注册 sglang.srt.plugins 入口点
export SGLANG_PLUGINS=sgl_asap_plugin         # venv 里还装着旧的 sgl_afd_plugin，只加载本插件
bash scripts/launch_asap.sh                   # 先起 F（GPU 4-7），再起 A（GPU 0），见脚本头部的变量
```

A 与 F 必须看到同一组 GPU（同一个 `CUDA_VISIBLE_DEVICES`，用 `--base-gpu-id` 分卡），否则 CUDA IPC 打不开对端内存。F 第一次启动时 FlashInfer 会 JIT 编译 MXFP4 MoE 内核（约 15 分钟，之后走缓存）。始终在激活 venv 的环境里启动：JIT 用 PATH 上找到的 ninja，venv 里的 1.13 与 `/usr/local/bin` 的 1.11 互不认对方的依赖日志，来回切换会整模块重编。

## 配置

| 选项 | 环境变量 | 说明 |
|---|---|---|
| `--asap-role {attn,ffn}` | `ASAP_ROLE` | 角色。`attn` 用 `sglang serve`，`ffn` 用 `python -m sgl_asap_plugin.ffn` |
| `--asap-ffn-ranks E` | `ASAP_FFN_RANKS` | A 上给了它才进入 AFD 模式（专家在 F 上）；不给则专家留在本地，只是纯 DP。F 上默认等于 `--ep-size` |
| `--asap-rdv-addr HOST:PORT` | `ASAP_RDV_ADDR` | F rank 0 的握手地址，rank f 用端口 + f。默认 `127.0.0.1:9600`（要在 10000 以下：本机把 10000-61000 分给外连源端口） |
| `--asap-n-max N` | `ASAP_N_MAX` | 一条消息的最大 token 数，决定 arena 大小，默认 16384。A 上 `--chunked-prefill-size` 与 `--max-prefill-tokens` 不能超过它 |
| `--asap-slots S` | `ASAP_SLOTS` | 每个 A rank 的在途 micro-batch 槽位，默认 2（同步前向用 1 个，dual-batch 用 2 个） |
| `--asap-num-replicas D` | `ASAP_NUM_REPLICAS` | A 副本数。F 必填；A 上默认等于 `--dp-size` |
| `--asap-replica-tp T` | `ASAP_REPLICA_TP` | 单个 A 副本的 TP。F 必填；A 上默认等于 `--tp-size`（AFD 目前只支持 1） |

启动进程解析配置后导出为 `ASAP_*` 环境变量，子进程（DP controller、scheduler）继承。

## Hook 表

注册前由 `hooks.preflight` 按期望参数名校验每个目标，改名或签名漂移直接报错；装上后再确认每个补丁真的生效（sglang 的加载器失败时只记日志）。

| 目标 | 类型 | 层级 | 作用 |
|---|---|---|---|
| `sglang.srt.server_args.ServerArgs.add_cli_args` | AFTER | base | 增加 `--asap-*` 选项 |
| `sglang.srt.server_args.ServerArgs.from_cli_args` | BEFORE | base | 见到 `--asap-role` 时激活 role/afd 层并导出配置，早于 ServerArgs 解析 |
| `sglang.srt.arg_groups.parallel_hook.handle_dwdp` | AFTER | role | AFD 的 A 角色声明 `disable_cuda_graph=True`（与 DWDP 用同一个位置：cuda graph 解析之前） |
| `sglang.srt.arg_groups.validation_hook.check_server_args` | BEFORE | role | 检查部署规则（`config.check_attn` / `check_ffn`），早于 sglang 自带检查 |
| `sglang.srt.model_executor.model_runner.ModelRunner.load_model` | AFTER | role | 记录每个 rank；AFD 模式下在 KV 池分配之前连接 F |
| `sglang.srt.layers.moe.ep_moe.layer.get_moe_impl_class` | REPLACE | afd | MoE 层拿到无参数的 `RemoteExperts`，不分配任何专家张量 |
| `sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward` | AROUND | afd | 持有 `RemoteExperts` 的层改走 `attn.moe_forward`，其余层不变 |
| `sglang.srt.model_loader.weight_utils.buffered_multi_thread_safetensors_weights_iterator` | REPLACE | afd | 路由专家与 MTP 张量不读（166.9 GB 里的约 147 GB） |
| `sglang.srt.model_loader.weight_utils.safetensors_weights_iterator` | REPLACE | afd | 同上，单线程路径 |

base 层每个进程都装，只加选项；role 层只在 ASAP 部署里装；afd 层只在给了 `--asap-ffn-ranks` 的 A 角色里装。用户启用后失败一律响亮：入口点里用 `SystemExit`（sglang 的加载器只吞 `Exception`），启动进程里直接抛异常。

## A↔F 契约

A 发送 hidden states、逻辑 top-k id 和**最终**路由权重（A 的 TopK 已乘 `routed_scaling_factor`，`RemoteExperts.should_fuse_routed_scaling_factor_in_topk = True`，与 Hopper 上 stock 的 MXFP4 方法一致）；F rank f 返回 Σ_{k: id_k 属于 f} w_k · expert_{id_k}(x)，A 把 E 个部分和用 fp32 相加。F 不需要知道缩放系数，A 不需要知道 F 的量化方法。握手时 F 逐项核对几何（`Geometry`）和模型指纹（hidden、topk、专家数、MoE 层号），不一致就拒绝并指出字段。

## 验收

`python scripts/acceptance.py <stage>`，每个阶段打印 PASS/FAIL 表并写 `report.md`（默认在 `/tmp/asap_reports/<stage>`：共享盘可能写满）。标准与结果见 [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)。

| 阶段 | 内容 | 命令 |
|---|---|---|
| M0 | 插件骨架、规则、纯 DP 拓扑 | `acceptance.py m0 --gpus 0,1 --control-gpus 2,3` |
| PR-03 | 传输库 | `acceptance.py pr03 --gpus 0,1,2,3` |
| PR-04 | A 侧 MoE 前向（对合成 F） | `acceptance.py pr04 --gpus 0,1,2` |
| PR-05 | F 服务（对 torch 参考实现） | `acceptance.py pr05 --gpus 0,1,2,3` |
| PR-04..07 | 全量模型：A/F 拉起、与 colocate 对齐、多副本无屏障 | `acceptance.py e2e --a-gpus 0,1 --f-gpus 4,5,6,7` |

## 限制

- 单机：传输是 CUDA IPC 加 /dev/shm（RDMA 后端是 PR-18）。
- AFD 的 A 副本只支持 `--tp-size 1`；不支持 CUDA graph（等待值在入队时固定，graph 重放不会推进序号）、投机解码、PP、attention DP。
- 每个 A 向每个 F 广播整份 payload（按路由压缩是 PR-14）；同步前向每层只有一个 micro-batch 在途（dual-batch 是 PR-09b）。
