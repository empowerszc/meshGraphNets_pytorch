#!/bin/bash
#
# MeshGraphNet 推理性能测试 - 启动脚本
#
# 支持两种机器:
# 1. 304 核心 ARM 服务器 (8 NUMA × 38 核心) - 多进程 CPU 推理
# 2. A100 x86 服务器 - 单进程 GPU 推理
#

set -e

# ==================== 默认配置 ====================
MACHINE_TYPE=""        # arm_304core | a100_x86
BATCH_SIZE=4
NUM_SAMPLES=100
SPLIT="test"
CHECKPOINT=""
OUTPUT_DIR="outputs/perf_test/$(date +%Y%m%d_%H%M%S)"

# ARM 服务器配置
NUMA_PROCESSES=4       # 每个 NUMA 的进程数
BASE_THREADS=9         # 基础线程数（每进程）

# ==================== 参数解析 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --machine)
            MACHINE_TYPE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --processes-per-numa)
            NUMA_PROCESSES="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法：$0 --machine <arm_304core|a100_x86> [选项]"
            echo ""
            echo "必需参数:"
            echo "  --machine TYPE         机器类型：arm_304core | a100_x86"
            echo ""
            echo "推理配置:"
            echo "  --batch-size N         每个 batch 的样本数 (默认 4)"
            echo "  --num-samples N        推理的样本总数 (默认 100)"
            echo "  --split SPLIT          数据集划分：test|valid|train (默认 test)"
            echo "  --checkpoint PATH      模型 checkpoint 路径"
            echo ""
            echo "ARM 服务器专用:"
            echo "  --processes-per-numa N 每个 NUMA 节点启动的进程数 (默认 4)"
            echo ""
            echo "输出:"
            echo "  --output-dir DIR       输出目录"
            exit 0
            ;;
        *)
            echo "未知选项：$1"
            exit 1
            ;;
    esac
done

if [[ -z "$MACHINE_TYPE" ]]; then
    echo "错误：必须指定 --machine 参数"
    exit 1
fi

# ==================== 机器配置 ====================
if [[ "$MACHINE_TYPE" == "arm_304core" ]]; then
    NUMA_NODES=8
    CORES_PER_NUMA=38
    # 每个 NUMA 的核心范围（精细到核心号）
    declare -a CORE_RANGES=(
        "0-37" "38-75" "76-113" "114-151"
        "152-189" "190-227" "228-265" "266-303"
    )
    # 每个 NUMA 对应的内存节点 (numa_id + 16)
    declare -a MEM_NODES=(16 17 18 19 20 21 22 23)
    MODE="cpu_multi_process"

elif [[ "$MACHINE_TYPE" == "a100_x86" ]]; then
    MODE="gpu_single_process"
    echo "A100 GPU 模式：单进程推理"

else
    echo "未知机器类型：$MACHINE_TYPE"
    exit 1
fi

# ==================== 创建输出目录 ====================
mkdir -p "$OUTPUT_DIR/workers"
mkdir -p "$OUTPUT_DIR/logs"

echo "=============================================="
echo "  MeshGraphNet 推理性能测试"
echo "=============================================="
echo "机器类型：$MACHINE_TYPE"
echo "运行模式：$MODE"
echo ""

if [[ "$MODE" == "cpu_multi_process" ]]; then
    echo "ARM 服务器配置:"
    echo "  NUMA 节点数：$NUMA_NODES"
    echo "  每 NUMA 核心数：$CORES_PER_NUMA"
    echo "  每 NUMA 进程数：$NUMA_PROCESSES"

    # 计算线程分配
    BASE_THREADS=$((CORES_PER_NUMA / NUMA_PROCESSES))
    REMAINDER_CORES=$((CORES_PER_NUMA % NUMA_PROCESSES))
    echo "  线程分配：每进程 ${BASE_THREADS} 线程"
    if [[ $REMAINDER_CORES -gt 0 ]]; then
        echo "             前 $REMAINDER_CORES 个进程各多 1 线程 (${BASE_THREADS}+1)"
    fi
    echo ""

    # 计算任务分配
    TOTAL_PROCESSES=$((NUMA_NODES * NUMA_PROCESSES))
    if [[ $NUM_SAMPLES -lt $TOTAL_PROCESSES ]]; then
        ACTUAL_PROCESSES=$NUM_SAMPLES
        echo "注意：样本数 ($NUM_SAMPLES) 少于进程数 ($TOTAL_PROCESSES)，将只启动 $ACTUAL_PROCESSES 个进程"
    else
        ACTUAL_PROCESSES=$TOTAL_PROCESSES
    fi

    SAMPLES_PER_PROCESS=$((NUM_SAMPLES / ACTUAL_PROCESSES))
    REMAINDER_SAMPLES=$((NUM_SAMPLES % ACTUAL_PROCESSES))

    echo ""
    echo "任务分配:"
    echo "  启动进程数：$ACTUAL_PROCESSES"
    echo "  每进程样本数：$SAMPLES_PER_PROCESS (余 $REMAINDER_SAMPLES 个)"
    echo ""

elif [[ "$MODE" == "gpu_single_process" ]]; then
    echo "A100 GPU 配置:"
    echo "  进程数：1"
    echo "  设备：cuda:0"
    echo ""
    echo "推理配置:"
    echo "  Batch size: $BATCH_SIZE"
    echo "  推理样本数：$NUM_SAMPLES"
    echo "  数据集：$SPLIT"
    echo ""
fi

echo "推理配置:"
echo "  Batch size: $BATCH_SIZE"
echo "  推理样本数：$NUM_SAMPLES"
echo "  数据集：$SPLIT"
echo "  Checkpoint: ${CHECKPOINT:-随机权重}"
echo ""
echo "输出目录：$OUTPUT_DIR"
echo "=============================================="

# ==================== 生成配置并启动进程 ====================

START_TIME=$(date +%s.%N)

if [[ "$MODE" == "cpu_multi_process" ]]; then
    # ========== ARM 多进程模式 ==========

    rm -f "$OUTPUT_DIR/pids.txt"
    PROCESS_ID=0

    for NUMA_ID in $(seq 0 $((NUMA_NODES - 1))); do
        if [[ $PROCESS_ID -ge $ACTUAL_PROCESSES ]]; then
            break
        fi

        MEM_NODE="${MEM_NODES[$NUMA_ID]}"
        CORE_START=$((NUMA_ID * CORES_PER_NUMA))

        # 计算该 NUMA 上每个进程的精确核心分配
        # 将 38 个核心均匀分配给 NUMA_PROCESSES 个进程
        # 使用全局偏移量追踪已分配的核心
        NUMA_CORE_OFFSET=0

        for PROC_IN_NUMA in $(seq 0 $((NUMA_PROCESSES - 1))); do
            if [[ $PROCESS_ID -ge $ACTUAL_PROCESSES ]]; then
                break
            fi

            # 计算该进程的基础核心数
            if [[ $PROC_IN_NUMA -lt $REMAINDER_CORES ]]; then
                PROC_CORES=$((BASE_THREADS + 1))
            else
                PROC_CORES=$BASE_THREADS
            fi

            # 生成该进程的核心列表（连续分配）
            CORE_LIST=""
            for ((i=0; i<PROC_CORES; i++)); do
                CORE_IDX=$((CORE_START + NUMA_CORE_OFFSET + i))
                if [[ -n "$CORE_LIST" ]]; then
                    CORE_LIST="${CORE_LIST},${CORE_IDX}"
                else
                    CORE_LIST="${CORE_IDX}"
                fi
            done
            NUMA_CORE_OFFSET=$((NUMA_CORE_OFFSET + PROC_CORES))

            # 计算该进程负责的样本范围
            if [[ $PROCESS_ID -lt $REMAINDER_SAMPLES ]]; then
                PROC_SAMPLES=$((SAMPLES_PER_PROCESS + 1))
                START_SAMPLE=$((PROCESS_ID * (SAMPLES_PER_PROCESS + 1)))
            else
                PROC_SAMPLES=$SAMPLES_PER_PROCESS
                START_SAMPLE=$((REMAINDER_SAMPLES * (SAMPLES_PER_PROCESS + 1) + (PROCESS_ID - REMAINDER_SAMPLES) * SAMPLES_PER_PROCESS))
            fi
            END_SAMPLE=$((START_SAMPLE + PROC_SAMPLES - 1))

            # 生成配置文件
            CONFIG_FILE="$OUTPUT_DIR/workers/process_${PROCESS_ID}.json"
            cat > "$CONFIG_FILE" <<EOF
{
    "process_id": $PROCESS_ID,
    "numa_id": $NUMA_ID,
    "core_list": "$CORE_LIST",
    "mem_node": $MEM_NODE,
    "threads": $PROC_CORES,
    "start_sample": $START_SAMPLE,
    "end_sample": $END_SAMPLE,
    "batch_size": $BATCH_SIZE,
    "split": "$SPLIT",
    "checkpoint": "$CHECKPOINT"
}
EOF

            echo "进程 $PROCESS_ID: NUMA=$NUMA_ID, 核心=[$CORE_LIST], 内存=$MEM_NODE, 样本=$START_SAMPLE-$END_SAMPLE"

            # 启动进程（使用 numactl 精确绑核和内存）
            LOG_FILE="$OUTPUT_DIR/logs/process_${PROCESS_ID}.log"
            RESULT_FILE="$OUTPUT_DIR/workers/result_${PROCESS_ID}.json"

            numactl --cpunodebind=$NUMA_ID --membind=$MEM_NODE \
                OMP_NUM_THREADS=$PROC_CORES \
                python perf_worker.py \
                    --config "$CONFIG_FILE" \
                    --output "$RESULT_FILE" \
                    > "$LOG_FILE" 2>&1 &

            echo $! >> "$OUTPUT_DIR/pids.txt"
            PROCESS_ID=$((PROCESS_ID + 1))
        done
    done

elif [[ "$MODE" == "gpu_single_process" ]]; then
    # ========== A100 GPU 单进程模式 ==========

    CONFIG_FILE="$OUTPUT_DIR/workers/gpu_config.json"
    cat > "$CONFIG_FILE" <<EOF
{
    "process_id": 0,
    "mode": "gpu",
    "device": "cuda:0",
    "batch_size": $BATCH_SIZE,
    "start_sample": 0,
    "end_sample": $((NUM_SAMPLES - 1)),
    "split": "$SPLIT",
    "checkpoint": "$CHECKPOINT"
}
EOF

    echo "启动 GPU 推理进程..."
    LOG_FILE="$OUTPUT_DIR/logs/gpu.log"
    RESULT_FILE="$OUTPUT_DIR/workers/result_gpu.json"

    CUDA_VISIBLE_DEVICES=0 \
        python perf_worker.py \
            --config "$CONFIG_FILE" \
            --output "$RESULT_FILE" \
            > "$LOG_FILE" 2>&1 &

    echo $! >> "$OUTPUT_DIR/pids.txt"
fi

echo ""
echo "等待所有进程完成..."

# ==================== 等待完成 ====================
while read PID; do
    wait $PID
done < "$OUTPUT_DIR/pids.txt"

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# ==================== 收集结果 ====================
echo ""
echo "=============================================="
echo "  推理完成!"
echo "=============================================="
echo ""

python3 - "$OUTPUT_DIR" <<'SCRIPT'
import sys
import json
import os
from pathlib import Path

output_dir = Path(sys.argv[1])
workers_dir = output_dir / "workers"

results = []
for result_file in sorted(workers_dir.glob("result_*.json")):
    try:
        with open(result_file) as f:
            results.append(json.load(f))
    except:
        pass

if not results:
    print("错误：未找到任何结果文件")
    sys.exit(1)

# 计算总体统计
total_samples = sum(r.get("samples_processed", 0) for r in results)
elapsed_times = [r.get("elapsed_time", 0) for r in results]
total_time = max(elapsed_times)  # 并行取最大值
min_time = min(elapsed_times)

if total_time > 0:
    total_samples_per_sec = total_samples / total_time
else:
    total_samples_per_sec = 0

# 找到最慢/最快进程
slowest = max(results, key=lambda r: r.get("elapsed_time", 0))
fastest = min(results, key=lambda r: r.get("elapsed_time", 0))

# 输出汇总
print(f"汇总统计:")
print(f"  完成进程数：{len(results)}")
print(f"  总样本数：{total_samples}")
print(f"  总耗时：{total_time:.2f}s")
print(f"  吞吐量：{total_samples_per_sec:.2f} 样本/s")
if total_samples > 0:
    print(f"  平均延迟：{total_time / total_samples * 1000:.2f}ms/样本")
print(f"")
print(f"最慢进程：PID={slowest.get('process_id')}, 耗时={slowest.get('elapsed_time', 0):.2f}s")
print(f"最快进程：PID={fastest.get('process_id')}, 耗时={fastest.get('elapsed_time', 0):.2f}s")
if fastest.get('elapsed_time', 0) > 0:
    imbalance = (slowest.get('elapsed_time', 0) - fastest.get('elapsed_time', 0)) / fastest.get('elapsed_time', 0) * 100
    print(f"  负载不均衡度：{imbalance:.1f}%")
print(f"")

# 保存汇总结果
summary = {
    "machine_type": "arm_304core" if len(results) > 1 else "a100_x86",
    "total_processes": len(results),
    "total_samples": total_samples,
    "total_time_sec": total_time,
    "throughput_samples_per_sec": total_samples_per_sec,
    "avg_latency_ms": total_time / total_samples * 1000 if total_samples > 0 else 0,
    "slowest_process": slowest.get("process_id"),
    "slowest_time_sec": slowest.get("elapsed_time", 0),
    "fastest_process": fastest.get("process_id"),
    "fastest_time_sec": fastest.get("elapsed_time", 0),
    "imbalance_percent": (slowest.get("elapsed_time", 0) - fastest.get("elapsed_time", 0)) / fastest.get("elapsed_time", 0) * 100 if fastest.get("elapsed_time", 0) > 0 else 0,
    "workers": results
}

with open(output_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"汇总结果已保存：{output_dir / 'summary.json'}")
SCRIPT

echo ""
echo "日志目录：$OUTPUT_DIR/logs"
echo ""
