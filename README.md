# Nemotron-3-Diarization.AXERA

`nvidia/Nemotron-3-Diarization`（NVIDIA Open Model License，openmdw-1.1）端到端说话人日志
在 AX650N 的部署工程：输入 16 kHz 单声道会议音频，输出 RTTM 说话人时序标签
（最多 8 说话人，10 ms 输出分辨率）。

- [x] 模型转换（FlexAttention 重写导出 → 拆图 → 三层对分 → 校准 → Pulsar2 量化）
- [x] Python 板端推理（axengine；offline RTF 0.0096 / low latency RTF 0.189）
- [x] C++ 板端推理（offline RTF 0.0066 / low latency RTF 0.179）

量化模型与运行时资产发布在 HuggingFace：
[HY-2012/Nemotron-3-Diarization.AXERA](https://huggingface.co/HY-2012/Nemotron-3-Diarization.AXERA)，
本仓库仅含全部代码。

## 支持平台

- AX650N（NPU3）

## 模型

- 模型：`nvidia/Nemotron-3-Diarization`（端到端流式，8 说话人，10 ms 输出）
- 量化：8 张拆图 axmodel（4 图 × NPU3 / NPU2 两个核数版本），U16 激活 / S8 权重；两档流式配置：
  - **offline**（精度优先）：输入缓冲 30.4 s（chunk 340 + right context 40 帧 @ 80 ms，
    spkcache 264 / fifo 40）；逐算子精度 encoder min cosine 0.9953 / mean 0.9989
  - **low latency**（实时流式）：输入缓冲 ~0.8 s（chunk 6 + right context 4，
    spkcache 264 / fifo 264，update 222）；逐算子精度 encoder min cosine 0.9588 / mean 0.9977
  - **NPU2 版**（`*_npu2.axmodel`）：同量化参数，仅编译期调度改为双核；输出与 NPU3 版逐帧一致
    （maxdiff 0），单步耗时更高，适合同一 NPU 上与其他模型并跑（留 1 核）
- 主机侧：与 NeMo 对齐的 log-mel 前端（复用检查点内量化的 hann 窗/滤波器组）、
  `streaming_update` AOSC 状态机（learnable silence embedding）、high-res→80 ms
  降采样与后处理（纯 numpy）
- 转换要点：
  - 上游编码器用 **FlexAttention**（不可导出），导出脚本用显式 matmul+softmax
    重写等价前向（复用同一权重），与上游逐层对分 maxdiff=0
  - `concat_and_pad` 的 Scatter/Where 索引链移到主机侧打包，图内无动态 shape
  - 模型为 high-resolution 档：编码器输出经 subpixel Conv1d 上采样 8 倍得 10 ms
    概率；状态机用其 80 ms 降采样版本（与 `forward_streaming_step` 一致）

## 目录结构

```
├── model_convert/        # 拆图 ONNX 导出 → 对分 → 校准 → Pulsar2 量化
├── python/               # 板端 Python 推理（nemotron_sdk + example.py + board_check.py）
├── cpp/                  # C++ 推理源码 + 交叉编译（见 cpp/README.md）
└── models/               # 仅 README（模型从 HuggingFace 下载）
```

## 环境

```bash
# 转换与基线（host）：NeMo Speech 仓库环境（torch + nemo_toolkit[asr]）
pip install -r requirements.txt

# 板端推理
pip install numpy soundfile
# pyaxengine: https://github.com/AXERA-TECH/pyaxengine/releases/latest
```

## 模型转换（三步）

1. **导出**：`python export_onnx.py --preset offline`（preencode + encoder，onnxsim
   常量折叠，torch↔ONNX 与上游 FlexAttention 双重对分）
2. **校准**：`python generate_calibration.py ...` + `bash pack_calib.sh`（真实会议状态采样）
3. **编译**：`bash compile.sh`（Pulsar2 Docker，U16/S8）

详见 [model_convert/README.md](model_convert/README.md)。

## 板端部署

### Python 推理

```bash
# 从 HuggingFace 下载 models/ 到本地
python python/example.py meeting.wav out.rttm
```

## 精度与性能（AX650N 实测）

三方对比（collar=0，pyannote.metrics，本地 AMI Mix-Headset / AliMeeting 重构参考）：

| 数据集 | NeMo FP32（参考） | 主机 ONNX FP32 | 板端 U16/S8 | ΔDER |
|---|---|---|---|---|
| AMI-SDM 5 场（offline） | 29.82% | 29.83% | **29.99%**（C++ 29.97%） | +0.17pp |
| AliMeeting 4 场（远场，offline） | 17.92% | 17.93% | **18.23%**（C++ 18.19%） | +0.31pp |
| AMI 2 场（low latency） | 28.28% | 28.26% | **29.08%** | +0.80pp |

逐帧余弦（板端 vs golden）：AMI offline 0.9995~0.9998，Ali 0.9993~0.9997。

性能（AX650N，主机 CPU 前端 + NPU 编码器）：

| 配置 | 版本 | Python RTF | C++ RTF | C++ ms/步 | 单步 MACs |
|---|---|---|---|---|---|
| offline（seq 684 × 31 层） | NPU3 | 0.0096 | **0.0066** | 177 | 194.6 GMACs |
| offline | NPU2 | — | 0.0075 | 201 | 194.6 GMACs |
| low latency（seq 538 × 31 层） | NPU3 | 0.189 | **0.179** | 85.9 | 143.1 GMACs |
| low latency | NPU2 | — | 0.338 | 162 | 143.1 GMACs |

参考：NeMo FP32 基线 RTF≈0.0017（offline）/ 0.062（low latency）（L4 GPU，未编译）。

## License

本仓库 Apache-2.0（见 [LICENSE](LICENSE)）；上游 Nemotron-3-Diarization 权重为
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)（openmdw-1.1）。
