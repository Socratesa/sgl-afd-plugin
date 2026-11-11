# 启动：run_asap.sh、run_ep.sh 只管本机；跨机用 *_cross.sh / run_ep_pair.sh，它们在每台 tray 上调用上面两个（经 ssh），自己只管摆放、等健康、起 router
bash scripts/run_asap.sh            # ASAP，本机：A 在 A_GPUS，F 在 F_GPUS；engine 都 healthy 后自动起 smg（ROUTER=0 不起），完成后写 $LOG_DIR/ready
bash scripts/run_ep.sh              # EP 基线，本机：EP 张卡，同上
bash scripts/run_asap_cross.sh      # ASAP 跨机：F_HOST（默认 hostfile 里另一台）、A_HOSTS="HOST:GPU,GPU ..."；每个实例一次 run_asap.sh（ROLE=ffn / attn）
bash scripts/run_ep_cross.sh        # 一个 EP 组的 rank 跨 NODES（默认 hostfile 全部，EP8）；每台一次 run_ep.sh（NNODES / NODE_RANK）
bash scripts/run_ep_pair.sh         # 每台 tray 一个独立 EP4，一个 router；每台一次 run_ep.sh
# run_asap.sh / run_ep.sh 里"一台 tray 的那一份"由几个旋钮控制（ROLE、HOST、BOOTSTRAP、D_TOTAL、A_BASE、A_TAG、NNODES、NODE_RANK ...），
# 默认值就是本机的完整部署，只有 cross 脚本会设它们，单机用不到。
# 公共部分：lib.sh（被 source，不是入口）、smg.sh（EXPECT_RANKS=N 时等 router 发现 N 个 DP rank 才返回）、req.sh（回答里没有 text 就失败）

# prefill 对照：每组同一套流程  起集群 → 等 ready（引擎 healthy 且 router 发现全部 rank）→ req → bench → 停集群
bash scripts/prefill_test.sh                      # DEFAULT_GROUPS
bash scripts/prefill_test.sh asap_2a2f baseline_ep4
bash scripts/prefill_test.sh list
# 每组的日志在 logs/prefill_<组>_<时间>/（launch.log、engine 日志、pd_router.log、req.log、bench.log），bench 结果在 bench-qf/results-t6t7/<时间>_<组>-qf

# nsys：nsys 不能 attach 到已经在跑的进程（CUDA 注入发生在进程启动时），所以服务要在 nsys 下启动、延迟采集，之后 start/stop 随时开关。
# 启动器对 nsys 一无所知，只提供一个通用钩子 LAUNCH_WRAP（见 lib.sh）；nsys_wrap.sh 不读任何启动脚本，也不按脚本名推断任何东西。
bash scripts/nsys_wrap.sh run scripts/run_asap.sh          # 任何启动器都行：run_asap_cross.sh、run_ep.sh、run_ep_pair.sh、prefill_test.sh ...
bash scripts/nsys_wrap.sh start [TAG]       # 所有会话同时开始一次抓取（服务起来之后任何时候）
bash scripts/nsys_wrap.sh stop              # 写 <NSYS_OUT>/<会话>_<启动时间>_<实例>_<TAG|N>[_<后缀>].nsys-rep，服务不停；可以反复 start/stop
bash scripts/nsys_wrap.sh capture 20 mid    # start + 睡 20 s + stop
bash scripts/nsys_wrap.sh status
bash scripts/nsys_wrap.sh shutdown          # 结束会话（服务随之退出）
# 每个服务实例一个 nsys 会话，在它所在的 tray 上（A0/A1/.../F，EP 或 EP0/EP1/...）。单机、跨机一样；后缀默认是部署名（6a2f / 2a2f / ep4），NSYS_SUFFIX= 可置空。

# NSYS_DETAIL=light（默认）去掉 DWARF 回溯、Python 采样、cpuctxsw、osrt、torch autograd NVTX、显存追踪，
# 只留 kernel / CUDA API / NVTX / NCCL / GPU metrics。A 的 prefill 受 CPU launch 限制，full 会拉长要测的间隙。NSYS_DETAIL=full 才有那些。

# 空闲（还没 start）时 nsys 也不是免费的：--trace 列表、--cuda-memory-usage、--pytorch 这些启动期 hook 从进程第一条指令起就在，不管有没有在抓。
# 实测（bench-qf，32 个 session、并发 16、只做 prefill；基线 = 同一部署不套 nsys；每项一次，EP4 基线跑两次差 0.9%）：
#                                         EP4（run_ep.sh）               2A2F（run_asap.sh）
#   NSYS_DETAIL=full                     时长 +50%，平均 TTFT +44%      时长 +24%，TTFT +25%
#   NSYS_DETAIL=light（默认）            +9%，TTFT +7%                  +4%，TTFT +5%
#   light + NSYS_GPU_METRICS=0 + 无 NVTX +6%，TTFT +4%                  （未测）
# 所以默认是 light；full 只在需要 CPU 回溯 / osrt / Python 采样时用，用之前要清楚前半程也会被拖慢。

# nsys 下 F 的专家 CUDA graph 会和 CUPTI 抓取冲突（F 的 GPU 卡在 graph 里不返回，A 60s 后 ChannelPoisoned 退出），
# 所以 wrapper 给 F（--asap-role ffn）追加 --asap-ffn-cuda-graph-max-tokens 0（专家 eager 跑，F 的 kernel 比正常运行慢）。NSYS_FFN_GRAPH_TOKENS=<n> 可改回 graph（会卡）。
# wrapper 还会给服务设 NVTX 环境变量并追加 --enable-layerwise-nvtx-marker（NSYS_NVTX_FLAGS= 可去掉），这些在每次 forward 都会跑，不管有没有在抓。

WORKER_URLS="10.99.91.63 10.99.91.59" 
