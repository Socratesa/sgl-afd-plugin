A_EXTRA="--enable-layerwise-nvtx-marker" F_EXTRA="--enable-layerwise-nvtx-marker" \
  bash scripts/nsys_wrap.sh scripts/run_asap.sh

bash scripts/nsys_wrap.sh start     # 开始
bash scripts/nsys_wrap.sh stop      # 写 logs/nsys/sgl_nsys_<启动时间>_N.nsys-rep，服务不停
bash scripts/nsys_wrap.sh status
bash scripts/nsys_wrap.sh shutdown  # 结束会话（服务随之退出）