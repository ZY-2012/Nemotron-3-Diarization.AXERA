# 板端 Python 推理（AX650N）

`nemotron_sdk/` 是纯 numpy 推理 SDK（无 torch / librosa 依赖），前端 mel、
FIFO/AOSC 状态机、后处理全部在主机侧实现；两张 axmodel
（`preencode` + `encoder`）由 axengine 执行。

## 依赖

```bash
pip install numpy soundfile
# pyaxengine: https://github.com/AXERA-TECH/pyaxengine/releases/latest
```

模型资产从 HuggingFace 下载到 `../models/`：

```
models/
├── preencode.axmodel
├── encoder.axmodel
├── learnable_sil_emb.npy    # AOSC 压缩用静音嵌入
├── feature_window.npy       # 检查点内 hann 窗（必须复用）
└── mel_filterbank.npy       # 检查点内 mel 滤波器组（必须复用）
```

## 用法

```bash
# 板端（挂载路径 /root/huyuan/workspace/...）
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
python3 example.py meeting.wav out.rttm

# 计时/评分产物
python3 board_check.py --preencode-axmodel ../models/preencode.axmodel \
    --encoder-axmodel ../models/encoder.axmodel --wav meeting.wav --out-dir out \
    --sil-emb ../models/learnable_sil_emb.npy --assets ../models
```

低延迟档（~0.8 s 输入缓冲，RTF 0.189，需配套的低延迟图）：

```bash
python3 example.py meeting.wav out.rttm \
    --preencode ../models/preencode_lowlat.axmodel \
    --encoder ../models/encoder_lowlat.axmodel \
    --chunk-len 6 --chunk-right-context 4 --fifo-len 264 --spkcache-update-period 222
```
