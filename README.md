# sgl-afd-plugin

SGLang 插件：prefill 阶段的 ASAP 式 Attention-FFN 分离（AFD）。

```
 A 副本 0 (GPU a0) ─┐                        ┌─ F rank 0 (GPU f0): 专家 [0, P)
 A 副本 1 (GPU a1) ─┼── 每层 MoE：x, ids, w ──┼─ F rank 1 (GPU f1): 专家 [P, 2P)
   ...              │   ←── 部分和 ──────────  │   ...
 A 副本 D-1 ────────┘ (IPC/fabric + 流内存操作) └─ F rank E-1
```

插件只提供 A↔F 的通信库（`transport/`）和把它接进 SGLang 的代码；进程、调度、加载、并行都是 SGLang 自己的。包的结构与 `sglang` 相同：`srt/` 下每个模块都放在它挂接或扩展的 SGLang 模块的同一路径上。

- **A（attention）**：`sglang serve --asap-role attn --asap-ffn-size E --dp-size D --tp-size 1`，D 个纯 DP 副本，每个副本是独立的 torch.distributed world，副本之间没有同步。跑 SGLang 自己的模型：`get_moe_impl_class` 返回 `AsapMoE`（没有专家权重的 FusedMoE），它的 `dispatcher` 是 `BaseDispatcher` 的子类 `AsapDispatcher`：`dispatch()` 把这一层的 token 发给 F，`combine()` 等 F 回复并求和。gate、top-k、共享专家照常在 A 上算。shared experts 默认在计算流上按 dispatch → shared GEMM → combine 执行，copy 使用单独的发送流，与 F 的 routed experts 重叠。
- **F（FFN）**：`sglang serve --asap-role ffn --tp-size E --ep-size E`，E 个纯 EP rank，rank 之间不通信，按到达顺序逐层服务所有副本。启动进程只拉 scheduler rank 和 `/health`（同 `--node-rank >= 1` 的节点）；每个 rank 是 `FfnScheduler`（`Scheduler` 子类：初始化到 model worker 为止，不建 KV cache、attention backend、CUDA graph 和请求路径）。模型是 `srt/models/` 里该模型族的 F 侧模型：只有每层的路由专家，由 SGLang 的 loader 加载。
- **A↔F**：静态 arena + 单边拷贝 + 64 位 doorbell + 流内存操作等待；A 的 CPU 从不等 F。内存布局与协议写在 `transport/layout.py` 的模块文档里。arena 和 doorbell 页怎么共享由 `transport/memory.py` 决定（`--asap-transport-mem`）：`ipc` 是 CUDA IPC 加 /dev/shm，只在一个 OS 内（H100、B200 单机）；`fabric` 是 CUDA fabric handle，同一个 NVLink 域内都行，GB200 上 A 与 F 可以在不同 tray。协议本身两者相同。

- **跨 tray（GB200）**A 与 F 可以放在同一个 NVLink 域的不同 tray 上（`nvidia-smi -q` 的 Fabric 里 ClusterUUID 与 CliqueId 相同）。每次启动仍是单机（`--nnodes 1`）：A 要占多台 tray 时，每台各起一个 A 实例，而不是一个 `--nnodes > 1` 的进程组。所有实例共用同一组 F，每个都填同一个 `--asap-attn-dp-size`（全部副本总数），用 `--asap-attn-base-replica` 各占一段不重叠的副本号（实例 i 的副本是 `[B, B + --dp-size)`）。各实例有自己的 scheduler 和 HTTP 端口，副本之间没有 torch.distributed 同步；请求由外面的路由打到这些端口（`scripts/smg.sh` 的 `WORKER_URLS`）。`--asap-bootstrap-addr` 写 F 所在 tray 的 IP（F 绑定它，每个 A 连它）；`--asap-transport-mem` 用默认的 `auto`（GB200 上即 fabric）或显式 `fabric`。A 与 F 不需要相同的 `CUDA_VISIBLE_DEVICES`，一个 A 实例也可以和 F 放在同一台 tray 的不同 GPU 上。A 与 F 必须在同一个 NVLink 域里，没有走网络的后端，跨域时握手拒绝。`scripts/run_asap_cross.sh` 按 `A_HOSTS` 起这些实例。


## 支持的模型

| 模型 | 文件 | 验证 |
|---|---|---|
| DeepSeek-V4-Flash（`DeepseekV4ForCausalLM`，MXFP4 专家；F 上验证过 Hopper 的 `flashinfer_mxfp4`（默认）与 `humming`） | `srt/models/deepseek_v4.py` | 全量 e2e，见 [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) |
| ERNIE 5.5 Flash（`Ernie5_5_MoeForCausalLM`，FP8 block 专家或 MXFP4 专家（`quantization_config` 里 `"expert_dtype": "mxfp4"`），MXFP4 专家在 F 上用 `--moe-runner-backend deep_gemm`（W4A8）；SGLang 不支持 ERNIE 的 batch overlap，A 上不开 DBO） | `srt/models/ernie55.py` | 待验证 |

架构不在 `srt/models/` 里时，两个角色启动时都会拒绝，并列出支持的架构。


## 快速开始

```bash
# sglang main @ 0bf026b1b5 (0.5.24.dev1296) has been installed
source .venv/bin/activate
uv pip install -e . --no-build-isolation      # 注册 sglang.srt.plugins 入口点
bash scripts/launch_asap.sh                   # 2A4F：A 在 GPU 0、1，F 在 GPU 4-7，DBO 开，chunk 16k；启动后等两边 /health
```


## 配置

| 选项 | 环境变量 | 说明 |
|---|---|---|
| `--asap-role {null,attn,ffn}` | `ASAP_ROLE` | 角色，默认 `null`（不启用），两种角色都用 `sglang serve`。`attn`：`--dp-size D` 个纯 DP 副本；`ffn`：专家服务，`--tp-size E --ep-size E` |
| `--asap-bootstrap-addr HOST:PORT` | `ASAP_BOOTSTRAP_ADDR` | F rank 0 的握手地址，rank f 用端口 + f。默认 `127.0.0.1:9600`（要在 10000 以下：本机把 10000-61000 分给外连源端口） |
| `--asap-transport-mem {auto,ipc,fabric}` | `ASAP_TRANSPORT_MEM` | A↔F 共享内存的方式（`transport/memory.py`）。`ipc`：CUDA IPC + /dev/shm，两个角色在同一台机器上、用同一个 `CUDA_VISIBLE_DEVICES`。`fabric`：CUDA fabric handle，同一个 NVLink 域内任意 GPU
| `--asap-max-dispatch-tokens N` | `ASAP_MAX_DISPATCH_TOKENS` | 一次 dispatch（一层 MoE、一次前向）的最大 token 数，决定 arena 大小，默认 16384。A 上 `--chunked-prefill-size` 与 `--max-prefill-tokens` 不能超过它 |
| `--asap-max-micro-batches S` | `ASAP_MAX_MICRO_BATCHES` | 每个 A rank 同时在途的 micro-batch 数（F 上每个 A rank 的接收槽位），默认 2（同步前向用 1 个，dual-batch 用 2 个） |
| `--asap-min-batch-tokens N` | `ASAP_MIN_BATCH_TOKENS` | A（AFD）：等待中的请求要算的 token（prompt 减去 radix cache 命中的前缀）凑够 N 个、或最老的等满 `--asap-max-wait-ms` 才出 prefill 批（length-aware batching）。批大则每层发给 F 的 job 大，F 每 token 便宜得多。超出 `--chunked-prefill-size` 的部分由 SGLang 的 chunked prefill 截断，余下的留到下一步。默认 0，即不等 |
| `--asap-max-wait-ms T` | `ASAP_MAX_WAIT_MS` | A：上一项里请求最多等多久，默认 50 |
| `--asap-tbo-min-tokens N` | `ASAP_TBO_MIN_TOKENS` | A（AFD）加 `--enable-two-batch-overlap` 时，只拆至少 N 个 token 的 prefill 批，默认 6144。小批受 CPU 下发限制，拆成两半下发量翻倍，反而更慢（`docs/DBO.md`） |
| `--asap-ffn-cuda-graph-max-tokens N` | `ASAP_FFN_CUDA_GRAPH_MAX_TOKENS` | F：每层的专家调用按 2 的幂分桶捕获 CUDA graph，job 不超过 N 个 token 时重放，省掉每个 job 0.3–0.9 ms 的下发（`docs/BENCH_MOE.md`）。默认 0，即 eager |
| `--asap-attn-dp-size D` | `ASAP_ATTN_DP_SIZE` | 共用这组 F 的所有 A 实例的副本总数。F 必填；只有一个 A 实例时 A 上可省，取它的 `--dp-size`；多个 A 实例时每个都填同一个总数 |
| `--asap-attn-base-replica B` | `ASAP_ATTN_BASE_REPLICA` | A：本实例第一个副本在全部 D 个副本里的编号，默认 0。本实例的副本是 `[B, B + --dp-size)`，必须在 D 以内，且各实例不重叠；F 拒绝另一台机器上的进程连一个已被占用的编号 |
| `--asap-attn-tp-size T` | `ASAP_ATTN_TP_SIZE` | A 角色的 `--tp-size`。F 必填；A 上由 `--tp-size` 得出（AFD 目前只支持 1） |
| `--asap-ffn-size E` | `ASAP_FFN_SIZE` | F rank 数。A 上给了它才进入 AFD 模式（专家在 F 上），不给则专家留在本地，只是纯 DP。F 上默认等于 `--ep-size` |

启动进程检查完参数后把它们导出为 `ASAP_*` 环境变量，子进程（DP controller、scheduler）继承。另有两个超时：`ASAP_BOOTSTRAP_TIMEOUT`（A 连 F 的握手，默认 1800 s）和 `ASAP_WAITING_TIMEOUT`（A 等 F 回复，默认 60 s），对应 SGLang PD 分离的 `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` 与 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`。


## Hook 表

`register()` 在每个进程里注册 base 三个；启动进程解析到 `--asap-role` 后、ServerArgs 解析之前注册该角色的其余 hook，子进程按继承的 `ASAP_ROLE` 注册。hook 体都在插件里与目标同路径的模块中（参数的解析与检查在 `srt/arg_groups/asap_hook.py`，同 SGLang 自己的写法），`tests/test_cpu.py` 检查这一点和每个目标的签名。SGLang 的加载器对挂不上的 hook 只记日志，所以启动进程会检查每个 hook 都已生效。

| 目标 | 类型 | 角色 | 作用 |
|---|---|---|---|
| `sglang.srt.server_args.ServerArgs.add_cli_args` | AFTER | 所有进程 | 增加 `--asap-*` 选项 |
| `sglang.srt.server_args.ServerArgs.from_cli_args` | BEFORE | 所有进程 | 见到 `--asap-role` 时注册该角色的 hook 并导出 `ASAP_*` |
| `sglang.srt.configs.model_config.ModelConfig.__init__` | AFTER | 所有进程 | ERNIE 5.5 的 FP4 checkpoint 在 fp8 `quantization_config` 里写 `"expert_dtype": "mxfp4"`，SGLang 不认：置 `is_fp4_experts`，专家按 MXFP4 加载。FP8 checkpoint 没有这一项，专家照常按 FP8 加载。不带 `--asap-role` 的同机部署（EP 基线）也要它 |
| `handle_cuda_graph_config`（`register_resolution_hook`） | 解析步骤 | attn, ffn | `handle_asap`：声明 `disable_shared_experts_fusion=True`；AFD 的 A 再声明 `disable_cuda_graph=True` |
| `sglang.srt.arg_groups.validation_hook.check_server_args` | BEFORE | attn, ffn | `check_asap_server_args`：部署规则，早于 SGLang 自带检查；补齐由解析结果得出的值并导出 |
| `sglang.srt.utils.network.get_free_port_below_ephemeral` | REPLACE | attn, ffn | 本机临时端口从 10000 起，SGLang 给每个 DP 副本选进程组端口的范围（22000–29999）落在其中，只好退回 bind(0)，副本启动时常撞 `EADDRINUSE`；改为固定端口：副本 r 用 9500 + r（SGLang 先占住上一个副本的端口再要下一个，所以从 9500 起取第一个空闲的就是下一个副本的） |
| `sglang.srt.model_executor.model_runner.ModelRunner.load_model` | AFTER | attn | 记录每个 rank；AFD 模式下检查模型受支持、安装永久 shared pre-combine hook，并在 KV 池分配之前连接 F |
| `sglang.srt.layers.moe.ep_moe.layer.get_moe_impl_class` | REPLACE | attn（AFD） | MoE 层拿到 `AsapMoE`，不分配专家张量 |
| `sglang.srt.managers.scheduler.Scheduler.get_new_batch_prefill` | AROUND | attn（AFD） | 开 `--asap-min-batch-tokens` 时，等待队列要算的 token（去掉 cache 命中的前缀）不够这么多、最老的请求也没等满 `--asap-max-wait-ms` 就先不出 prefill 批，让随后到的请求合进同一批（length-aware batching）；某个请求正处在 chunked prefill 中途（上一步被截断，下一块必须跑）时不等 |
| `sglang.srt.arg_groups.validation_hook.check_two_batch_overlap` | REPLACE | attn（AFD） | 放行 `--enable-two-batch-overlap`：没有 EP a2a 时 SGLang 要求 attention DP，A 副本不需要 |
| `sglang.srt.model_executor.model_runner.ModelRunner._prepare_eager_forward_batch` | AFTER | attn（AFD） | 开 `--enable-two-batch-overlap` 时在本副本内拆分 prefill 批（SGLang 只在 attention-DP 同步里拆） |
| `sglang.srt.batch_overlap.operations_strategy._compute_moe_deepseek_v4_layer_operations_strategy_tbo` | AROUND | attn（AFD） | `AsapMoE` 层的 TBO 操作：attention → 发送 → shared experts ─让出─ 等回复 → 相加 |
| `sglang.srt.entrypoints.http_server.launch_server` | REPLACE | ffn | 只拉 scheduler rank 和 `/health` |
| `sglang.srt.managers.scheduler.Scheduler` | REPLACE（类） | ffn | 换成 `FfnScheduler` |
| `sglang.srt.model_loader.weight_utils.buffered_multi_thread_safetensors_weights_iterator` | AROUND | attn（AFD）, ffn | 本角色不加载的张量不读：A 不读路由专家（166.9 GB 里的约 147 GB），F rank f 只读自己的路由专家 |
| `sglang.srt.model_loader.weight_utils.safetensors_weights_iterator` | AROUND | attn（AFD）, ffn | 同上，单线程路径 |

F 角色另外在 scheduler 进程里用 SGLang 的 `ModelRegistry.register("sgl_asap_plugin.srt.models", overwrite=True)` 换掉同名架构的模型类；这是 SGLang 给外部模型的接口，不是 hook。


## 加一个模型

在 `srt/models/` 里按 SGLang 的模型文件写法加一个模块（例如 `qwen3_moe.py`）：以 checkpoint 的架构名定义 F 侧模型类，`EntryClass` 指向它。这个类只建 SGLang 对应模型里每个 MoE 层的路由专家：用同样的参数调 `get_moe_impl_class(quant_config)(...)`，参数名与 SGLang 的模型相同（`model.layers.{i}.mlp.experts.*`），`load_weights` 复用该模型的名字映射和 `FusedMoE.make_expert_params_mapping`。`deepseek_v4.py` 是现成的例子。另外要满足：

- A 侧：SGLang 的模型用 `get_moe_impl_class(...)` 建路由专家，并以 `self.experts(hidden_states, topk_output)` 调用，top-k 是标准格式。
- F 侧：量化方法要把 `routed_scaling_factor` 融进 top-k（`FusedMoE.should_fuse_routed_scaling_factor_in_topk`），否则 F 启动时拒绝。
- checkpoint 按专家号存张量（名字里有 `layers.<L>.` 和 `.experts.<E>.`）：A 不读、F 只读自己的。
- F 的 rank 数遵守 SGLang 的 EP 约束：整除专家数。