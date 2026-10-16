# 模型转换（Nemotron-3-Diarization，offline 30.4 s 配置）

从 NeMo Speech 的 `nvidia/Nemotron-3-Diarization` 导出**两张拆图 ONNX**，
用 Pulsar2 量化编译成 AX650/NPU3 的 `preencode.axmodel` + `encoder.axmodel`。

## 环境

```bash
# NeMo Speech 仓库（提供 SortformerEncLabelModel），uv 锁定环境
cd /path/to/nvidia-audio/Speech
uv sync --locked --python 3.13 --extra asr --extra cu12
```

路径通过环境变量指定（`paths.py`）：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `NEMOTRON_NEMO_REPO` | NeMo Speech 仓库 | `<workspace>/nvidia-audio/Speech` |
| `NEMOTRON_MODEL` | HF 模型名 | `nvidia/Nemotron-3-Diarization` |
| `NEMOTRON_NEMO_FILE` | 本地 `.nemo`（优先） | `<OUT>/models/Nemotron-3-Diarization.nemo` |
| `NEMOTRON_OUT` | 产物根目录 | 本工程根 |
| `NEMOTRON_CALIB_DIR` | 校准数据目录 | `<OUT>/calib_data` |
| `NEMOTRON_AMI_DIR` | AMI Mix-Headset + ref RTTM | `<workspace>/ami_eval` |
| `NEMOTRON_ALI_DIR` | AliMeeting 音频 + ref | `<workspace>/alimeeting` |

流式配置（模型卡 quick start 的 30.4 s 输入缓冲，offline 档）：

| 参数 | 值 |
|---|---|
| `chunk_len` / `chunk_left_context` / `chunk_right_context` | 340 / 0 / 40 |
| `fifo_len` / `spkcache_len` | 40 / 264 |
| `spkcache_update_period` | 300（有效值 340，见下） |

> `spkcache_update_period=300 < chunk_len=340` 会触发 NeMo 警告，有效更新周期
> 取 `max(period, chunk_len - fifo + fifo) = 340`；与模型卡一致，属预期行为。

导出为**两张拆图**（主机侧打包，图内无 Scatter/Where 索引链、无动态 shape）：

| 图 | 输入 | 输出 |
|---|---|---|
| `export/preencode.onnx` | `chunk [1,3040,128]` | `chunk_pre_encode_embs [1,380,512]` |
| `export/encoder.onnx` | `seq [1,684,512]`（spkcache\|fifo\|chunk 连续打包+补零）、`total_lengths [1]` float | `high_res_preds [1,5472,8]`（10 ms、sigmoid、已按长度掩码） |

两张图导出后都会跑 onnxsim 常量折叠（encoder 简化后 1422 节点，preencode 2 节点）。

### Nemotron 与 Sortformer v2.1 的转换差异

- 编码器是 **31 层 RoPE Transformer**，上游用 **FlexAttention**（不可导出）；
  `export_onnx.py` 中的 `ExportEncoderStep` 用**显式 matmul+softmax** 重写等价
  前向（复用同一份权重），导出前会与上游 FlexAttention 路径做 CPU 对分
  （实测 maxdiff=0.0）。
- 模型是 **high_resolution** 档：编码器输出经 subpixel Conv1d 上采样 8 倍，
  输出 10 ms 逐帧概率；主机侧再把打包序列平均池化回 80 ms 喂 AOSC 状态机
  （与 `forward_streaming_step` 完全一致）。
- AOSC 压缩使用**可学习静音嵌入**（`use_learnable_sil_emb=True`），需从
  检查点导出 `learnable_sil_emb.npy`。
- 前端 mel 的 **hann 窗与滤波器组在检查点内是量化过的**（与现算 librosa 值
  差 ~1e-3）；`dump_assets.py` 导出这两个缓冲区，numpy 前端必须复用才能
  与 NeMo 推理对齐（feature_parity maxdiff ≤2e-4）。

## 步骤

### ① 导出模型资产

```bash
hf download nvidia/Nemotron-3-Diarization Nemotron-3-Diarization.nemo config.json --local-dir models
python dump_assets.py     # learnable_sil_emb / feature_window / mel_filterbank
```

### ② NeMo FP32 基线

```bash
python make_ami_manifest.py ES2004a IS1009a EN2002d TS3003b EN2002a
python baseline_nemo.py --meetings ES2004a IS1009a EN2002d TS3003b EN2002a --preset offline --collar 0.0
python baseline_nemo.py --dataset ali --preset offline --collar 0.0
```

产出 `baseline/preds/<meeting>.npy`（帧级黄金参照，0.01 s/帧）、
`baseline/feats/<meeting>.npy`（NeMo mel 特征）。

### ③ 导出 ONNX（两张图）

```bash
python export_onnx.py --preset offline
```

### ④ 全链路 FP32 对分

```bash
python host_stream.py ES2004a IS1009a        # 主机流式循环 vs 黄金 preds
python state_parity.py EN2002d --max-steps 12 # numpy 状态机 vs NeMo torch
python feature_parity.py ES2004a              # numpy mel vs NeMo
python postprocess_parity.py ES2004a          # numpy 后处理 vs NeMo
```

### ⑤ 校准数据 + Pulsar2 量化

```bash
python generate_calibration.py --meetings EN2002c TS3003c ES2004b IS1009b --samples 20 --provider cpu --preset offline
bash pack_calib.sh
bash compile.sh        # 编译 preencode + encoder
```

量化配置：U16 激活 / S8 权重、MinMax、真实会议状态校准；逐算子精度
（`parse_precision.py`）encoder min cosine ≥0.99。

### ⑤b 低延迟档（~0.8 s，可选）

```bash
python export_onnx.py --preset low_latency --output-dir ../export_lowlat
python generate_calibration.py --meetings EN2002c TS3003c ES2004b IS1009b --samples 20 \
    --provider cpu --preset low_latency \
    --preencode ../export_lowlat/preencode.onnx --encoder ../export_lowlat/encoder.onnx
# 需要 NEMOTRON_CALIB_DIR 指向 calib_data_lowlat，再打包 + 用 lowlat 配置编译：
NEMOTRON_CALIB_DIR=$PWD/../calib_data_lowlat bash pack_calib.sh
docker run --rm -v "$PWD/..":/workspace -w /workspace \
    docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-09cadfa9 \
    pulsar2 build --config /workspace/model_convert/pulsar2_preencode_lowlat.json
docker run --rm -v "$PWD/..":/workspace -w /workspace \
    docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-09cadfa9 \
    pulsar2 build --config /workspace/model_convert/pulsar2_encoder_lowlat.json
```

> 校准/验证时注意 `--sil-emb` 不要静默缺失：AOSC 压缩用 learnable silence
> embedding，缺失会退化成运行时均值（与 NeMo 不一致），SDK 现在会直接报错。

### ⑥ 板端验证与评分

```bash
# 板端（挂载路径）：python3 python/board_check.py --preencode-axmodel ... --encoder-axmodel ... --wav ...
python score_board.py ES2004a IS1009a EN2002d TS3003b EN2002a
```

## 主机侧必须实现的三件事

1. **`streaming_update` 状态机**：AOSC top-k 压缩（含可学习静音嵌入）、FIFO
   弹出（`python/nemotron_sdk/state.py`，与 NeMo 逐步对齐，13 步对分零分歧）。
2. **high-res → 80 ms 降采样**：`downsample_preds`（mean-pool 8、按长度掩码），
   以及 chunk 输出切片 `start=(spkcache+fifo+lc_enc)*8`。
3. **前端 mel**：复用检查点内导出的 `feature_window.npy` / `mel_filterbank.npy`。
