# 验收标准与结果

一条命令一个阶段：`python scripts/acceptance.py <stage>`。测试自己断言门限，脚本把测试映射到验收 ID、补上实测数字，任何跳过都算失败。日志和 `report.md` 默认写到 `/tmp/asap_reports/<stage>`（共享盘写满过一次，PR-04 草稿的验收就是因此失败的）。

环境（2026-10-30）：8× Hopper 80 GB（NVLink NV18），driver 580.126.20，torch 2.14.1+cu130，sglang 0.5.22（`baa9befa38`），DeepSeek-V4-Flash-0731（43 层，256 专家 top-6，MXFP4 专家，166.9 GB）。venv `lhd/sgl_afd/.venv`（基于 conda 环境 `sgl_afd` 的 Python 3.12）。

## M0：骨架、规则、纯 DP 拓扑

`acceptance.py m0 --gpus 0,1 --control-gpus 2,3`，**PASS**。A 服务用 DSV4 截 4 层 + dummy 权重，D=2、T=1，专家仍在本地（不加 `--asap-ffn-ranks`）。

| ID | 检查 | 实测 | 标准 |
|---|---|---|---|
| U1 | CPU 测试：配置、规则、布局、checkpoint 选择、全新进程里的 hook 激活 | 28 个，全部通过 | ≥ 28，0 失败，0 跳过 |
| U2 | 入口点可被 sglang 发现 | `sgl_asap_plugin:register` | 同左 |
| U3 | 导入保持轻量 | 2 ms，未引入 torch、sglang | < 500 ms |
| G0 | A 服务启动 | 280 s，3 次尝试（前两次 `EADDRINUSE`） | `/health` 200；仅端口竞争时重试 |
| G1 | hook 在启动进程和每个 scheduler 生效 | 启动进程 5/5，两个 rank 5/5 | 全部应用 |
| G2 | 每个副本是独立 world | replica 0、1，world_size 都是 1 | 不是 D·T = 2 |
| G3 | 副本能服务请求 | 10/10 | 全部成功 |
| G4 | `SIGSTOP` 副本 0，副本 1 照常服务 | 5/5，最慢 0.03 s；被停的副本无回复；`SIGCONT` 后恢复 | ≤ 10 s |
| N1 | 反例：stock attention-DP 做同样实验 | 停一个 scheduler 后两个 rank 都不回复 | 证明 G4 的判据能失败 |

G0 的重试与插件无关：sglang 的 DP controller 用 `bind(0)` 选各副本的端口后立即释放，本机把 10000–61000 分给外连源端口，释放到副本绑定之间可能被占。

## PR-03：传输库

`acceptance.py pr03 --gpus 4,5,6,7`，**PASS**（20 个用例，120 s）。A 与各 F rank 是不同 GPU 上的独立进程。重构（13 个模块合成 4 个、watchdog 改为只看最老的等待组）之后重跑。

| ID | 检查 | 实测 | 标准 |
|---|---|---|---|
| C1 | doorbell 字段、几何规则、arena 大小 | 4 个用例通过 | 边界往返无损；越界被拒；标志独占页 |
| C2–C7 | 乒乓 1000 轮、3 个 F 部分和、8 KB–64 MB、seq 回绕、18000 轮压测、3 个发送者 FCFS | 全部 0 错 0 丢；C7 副本切换 > 100 | 同左 |
| B1 / B2 | 反压 / 反例：关掉反压 | 第二条停在队列里，按序送达 / 覆盖被检出 | 同左 |
| F1–F5 | F 死亡、旧 epoch 与重连、kill 一个 A、几何不一致、park 住的等待 | 全部通过；F5 冷路径 2.6 s（2.0 s 后释放），热路径 0.001 s | 同左 |
| P1 | n=1 往返（echo F） | p50 52 us（p99 193）；F 小睡时 p50 180 us | p50 < 150 us；小睡慢 > 40 us |
| P2 | 64 MB 单向与全双工 | 单向 361 GB/s；发送与回复重叠 695 GB/s | > 330；> 650 |
| P3 | 64 MB 往返 | RTT p50 394 us，总带宽 340 GB/s | > 150 GB/s |

P1 从重构前的 63 us 降到 52 us：每组等待只记一个 event（原来两个），watchdog 每 20 ms 只查最老的一组（原来每 2 ms 查全部，占 GIL）。硬件上限：裸 `cuMemcpyAsync` 单向 385 GB/s、双向 769 GB/s。

## PR-04（单元部分）：A 侧 MoE 前向

`acceptance.py pr04 --gpus 4,5,6`，**PASS**。真实的 `attn.moe_forward` 接合成 F（每个 rank 返回 (f+1)·base，和可精确校验），gate、top-k、共享专家用假模块。

| ID | 检查 | 标准 |
|---|---|---|
| A1 | 1 个 F rank：普通层、hash 层（`input_ids_global` 传到 top-k）、不带共享专家、`return_moe_output=True`、空批、超过 n_max 的批（分 3 条消息） | 逐位一致；消息数 = ceil(n / n_max)，空批 0 条 |
| A2 | 同上，2 个 F rank（fp32 相加部分和） | 逐位一致 |

## PR-05（服务部分）：F 服务

`acceptance.py pr05 --gpus 4,5,6,7`，**PASS**（979 s，其中约 15 分钟是 FlashInfer 的 JIT 重编，见 README）。

| ID | 检查 | 实测 | 标准 |
|---|---|---|---|
| S1 | 1 个 rank、1 层的生命周期 | 专家 [0,256)，`Mxfp4FlashinferCutlassMoEMethod`，3.42 GB，SIGTERM 后退出码 0 | 3.42 ± 0.2 GB |
| S2 | 3 个 rank（末尾 2 个补位专家）、3 个合成 A，随机层号乱序到达 | 对去量化的 torch 参考实现，最大相对误差 0.45%；到达顺序 3,3,2,1,1,2,1,0,… | < 2%；≥ 3 层；每条消息都被服务；显存只有专家 |
| S3 | A 的模型与 F 不符 | 握手拒绝，消息指出 `moe_layers` | 同左 |

## PR-04..07（全量模型）：A/F 拉起、与 colocate 对齐、多副本共享 F

`acceptance.py e2e --a-gpus 0,2 --f-gpus 4,5,6,7 --replicas 1,2 --reuse-reference`，**PASS**（2026-10-30 第二轮）。F 是 4 个 rank（每 rank 64 个专家，43 层）；A 每个副本 1 张卡；参考是两种 stock 部署（不加载插件），先单独跑出 `reference.json` / `reference2.json`。提示词是 64 条 128–16384 token 的序列（从 sglang 自己的源码切出，`prompts.json`），每条比较首 token、首 token 的 top-5 logprob 和提示词最后 256 个位置各自的 logprob（共 15613 个位置）。

| ID | 检查 | 实测 | 标准 |
|---|---|---|---|
| R0 | 噪声底：stock `--tp 4 --ep 4` 对 stock attention-DP `--tp 4 --dp 4 --ep 4` | 首 token 62/64；逐位置 \|Δlogprob\| 均值 0.0473，p99 0.630 | 只记录：两者都是正确的部署 |
| E1 | F 服务起来 | 每 rank 36.79 GB（= 147.2 / 4，只有专家），29–33 s | 全部 ready，≈ 36.8 GB |
| E2 | A 副本在全量模型上起来并连上 F | 64 s | `/health` 200 |
| E3 | hook 与连接 | 启动进程 9/9，rank 9/9，连上 4 个 F rank | 同左 |
| E4 | A 保留的权重 | 10.62 GB（stock 同一卡上要 166.9 GB / TP） | < 14 GB |
| E5 | 请求走完每个 MoE 层 | 10 / 500 / 3000 / 6000 / 16000 token：0.17 / 0.18 / 0.24 / 0.28 / 0.79 s | 全部成功（16000 = 2 个 chunk） |
| A1 | 首 token 对参考 | 63/64 相同 | ≥ 噪声底 − 1，0 错 |
| A2 | logprob 偏差对参考 | 均值 0.0484，p99 0.630 | ≤ 1.25 × 噪声底 |
| A3 | 1000 个请求，16 并发，128–4096 token | 0 错，最慢 1.3 s，共 28 s | 0 错，无挂起 |
| E6 | F 的统计 | 每 rank 43 层、9933 个 job、0 丢失、无旧 epoch、无丢弃回复 | 同左 |
| M2.0 | 2 个副本各自独立 world、都连上 F | replica 0、1，world_size 1，9/9 | 同左 |
| M2.1 | 每个副本复现单副本 AFD 的结果（前 16 条） | 两个副本 16/16，所有 logprob 逐位相同 | max \|Δ\| = 0 |
| M2.2 | 2000 个请求，16 并发，sglang 的 `total_tokens` 路由 | 0 错，最慢 1.7 s，共 54 s | 0 错 |
| M2.3 | 无屏障：副本 1 的 1024 token 短请求单独跑 vs 副本 0 同时在 prefill 32768 token 长请求 | TTFT p50 153 ms → 182 ms（×1.19）；长请求 p50 2.37 s | ×≤ 1.5，且远小于长请求耗时 |
| M2.4 | F 交替服务两个副本、且两者处在不同层 | 每 rank 约 24400 次副本切换；相邻且来自不同副本的 job 里约 2150 次层号不同 | 每 rank > 0 |

第一轮（2026-10-30，`--a-gpus 0,1,2,3 --replicas 1,2,4`）另跑了 4 个副本（4A4F，8 张卡）：M4.0 4 个副本 9/9；M4.2 4000 个请求 32 并发 0 错，最慢 2.6 s；M4.3 ×1.23（151 → 185 ms，长请求 p50 2.37 s）；M4.4 每 rank 约 52460 次副本切换。那一轮的 M4.1 用的是旧标准（对参考"首 token 100% 相同且 \|Δ\| ≤ 0.05"），4 个副本结果彼此完全相同（都是 16/16、top-5 最大差 1.875）。第二轮时 GPU 1、3 被其他任务占用，没有重跑 4 副本。

### 怎么读这些数字

- **对齐标准改成相对噪声底**。路线图 PR-06 原定"首 token 100% 一致、\|Δlogprob\| ≤ 0.05"。实测两种 stock 部署之间就做不到：stock `--tp 4 --ep 4`、stock `--tp 4`、stock attention-DP 两两比较，首 token 都是 62/64，逐位置均值都在 0.047–0.049，p99 都在 0.62–0.66；翻转的总是同几条（第 8、41、52 条，参考自身 top-2 差 0.125–0.875）。AFD 对参考的距离与这些 stock 部署之间的距离相同，所以判据是"不超过噪声底的 1.25 倍"。
- **确定性**。同一次启动内结果逐位确定；不同次启动之间，≤ 2048 token 的提示词逐位相同，更长的有小差异（两次 AFD 启动之间均值 0.014），量级在噪声底以内。M2.1 因此只比较前 16 条（都 ≤ 2048 token）。
- **M2.3 的 ×1.19 是共享 F 的排队**，不是屏障：副本 1 的某层到达 F 时，F 可能正在算副本 0 一个 8192 token 的 job。锁步设计下副本 1 每一步都要等副本 0 走完同一步，代价接近长请求本身的耗时（2.37 s）。M2.4 是直接证据：F 上相邻的两个 job 经常来自不同副本、处在不同层。
- **单副本的速度**（E5，不含 logprob）：A 1 张卡 + F 4 张卡，128 token 0.17 s、16K token 0.79 s。每层约 3.7 ms 的固定开销来自 eager 下 DSV4 的大量小 kernel 加一次 F 往返；性能对比是 PR-08 的事。
