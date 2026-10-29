# M0 验收结果

2026-10-29，`python scripts/m0_acceptance.py --gpus 0,1 --control-gpus 2,3`，**全部通过**。

环境：8× CF-NG-HZZ1-O（Hopper，80 GB），driver 580.126.20，torch 2.14.1+cu130，sglang 0.5.22（`baa9befa38`），模型 DeepSeek-V4-Flash-0731（截 4 层、128 专家、dummy 权重、FP8 专家），A 服务 D=2、T=1。

| ID | 检查 | 实测 | 标准 | 结果 |
|---|---|---|---|---|
| U1 | 单元与全新进程 hook 测试 | 44 个用例，0 失败，0 跳过，105 s | ≥ 43 个用例，0 失败，0 跳过 | PASS |
| U2 | 入口点可被 sglang 发现 | `sgl_asap_plugin:register` | 同左 | PASS |
| U3 | 导入保持轻量 | 0 ms，不引入 torch、sglang | < 500 ms，不引入 torch、sglang | PASS |
| G0 | A 服务启动 | 85 s，2 次尝试（第 1 次遇到 `EADDRINUSE`） | 900 s 内 `/health` 200，仅 `EADDRINUSE` 时重试 | PASS |
| G1 | hook 在每个进程生效 | 启动进程 1 行，scheduler 2 行，全部 hooks=4/4 | 启动进程 ≥ 1 行，scheduler 恰好 2 行，hooks 全部应用 | PASS |
| G2 | 纯 DP：每个副本是独立 WORLD | dp0（gpu 0）、dp1（gpu 1），world_size=1，tp_size=1 | dp_rank = {0,1}；world_size = tp_size = 1 | PASS |
| G3 | 副本能服务请求 | 10/10 成功，中位数 21 ms | 全部成功 | PASS |
| G4 | 副本独立性 | `SIGSTOP` 副本 0 后，副本 1：5/5 成功，最慢 0.02 s；发往副本 0 的请求 5 s 无回复；`SIGCONT` 后副本 0 恢复 | 副本 1 全成功且最慢 ≤ 10 s；副本 0 无回复；恢复后可服务 | PASS |
| N1 | 反例：stock attention-DP（tp=2、dp=2） | 停掉 2 个 scheduler 中的 1 个后，两个 rank 都不再回复（停之前都正常） | 停之后两个 rank 都不回复 | PASS |

## 怎么读这些数字

- **G2 + G4 + N1 合起来证明"互不等待"**：纯 DP 下每个副本只有自己的 world（world_size 等于 T，不是 D·T），冻结一个副本不影响另一个；同样的实验在 stock attention-DP 下会让两个 rank 一起停，所以 G4 的判据确实能失败。
- **G0 的重试不是插件问题**：sglang 的 DPC 用 `bind(0)` 选各副本的 rendezvous 端口，释放后副本才去绑定；本机临时端口范围是 10000–61000，sglang 里"避开临时端口"的取值区间（22000–29999）被这个范围吞掉，端口会被别的外连占走。本次验收 4 次启动里有 2 次遇到，重试后通过。
- **推荐入口也验证过**：另用 `sglang serve --asap-role attn --dp-size 2 --tp-size 1`（GPU 4、5）启动一次：hooks 4/4，2 个副本 world_size=1，3 个请求成功。`sglang serve --help` 不会加载插件，看选项请用 `python -m sglang.launch_server --help | grep asap`。
- 验收结束后无残留进程，GPU 显存回到 0。
