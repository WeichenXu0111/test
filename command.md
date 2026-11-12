conda activate soma
export SOMA_ROOT=/inspire/hdd/global_user/czxs25110026/laliwang/SoMA
export RUN_ROOT="$SOMA_ROOT/gentwin_converted_20261110"
cd "$SOMA_ROOT"
export PYTHONPATH="$SOMA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# 一阶段：四卡分布式训练
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  tools/train.py \
  "$RUN_ROOT/configs/single_push_sloth_stage1.py" \
  --work_dir "$RUN_ROOT/work_dirs/single_push_sloth/stage1" \
  --launcher pytorch \
  > "$RUN_ROOT/logs/single_push_sloth_stage1_4gpu.log" 2>&1 &

# 一阶段缓存生成、渲染与评估
nohup env CUDA_VISIBLE_DEVICES=0 python -u tools/test.py \
  "$RUN_ROOT/configs/single_push_sloth_stage1.py" \
  "$RUN_ROOT/work_dirs/single_push_sloth/stage1/epoch_15.pth" \
  --gpu-id 0 \
  --show-dir "$RUN_ROOT/work_dirs/single_push_sloth/stage1_cache_render" \
  --out "$RUN_ROOT/work_dirs/single_push_sloth/stage1_cache_metrics.json" \
  --cfg-options \
    model.flag_save_gaussian=True \
    model.frame_gap=10 \
    model.data_dir="$RUN_ROOT/data/" \
  > "$RUN_ROOT/logs/single_push_sloth_cache.log" 2>&1 < /dev/null &

# 二阶段四卡分布式训练
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  tools/train.py \
  "$RUN_ROOT/configs/single_push_sloth_stage2.py" \
  --work_dir "$RUN_ROOT/work_dirs/single_push_sloth/stage2" \
  --resume-from "$RUN_ROOT/work_dirs/single_push_sloth/stage1/epoch_15.pth" \
  --launcher pytorch \
  > "$RUN_ROOT/logs/single_push_sloth_stage2_4gpu.log" 2>&1 < /dev/null &

## 如果想限制rollout次数，则在这里加入--options model.train_cfg.max_rollout_step=6，这个数是可以变更的

# 二阶段连续预测、评估与渲染
nohup env CUDA_VISIBLE_DEVICES=0 python -u tools/test.py \
  "$RUN_ROOT/configs/single_push_sloth_stage2.py" \
  "$RUN_ROOT/work_dirs/single_push_sloth/stage2/epoch_30.pth" \
  --gpu-id 0 \
  --test-rollout-mode continuous \
  --show-dir "$RUN_ROOT/work_dirs/single_push_sloth/stage2_continuous" \
  --out "$RUN_ROOT/work_dirs/single_push_sloth/stage2_future_metrics.json" \
  > "$RUN_ROOT/logs/single_push_sloth_eval.log" 2>&1 < /dev/null &