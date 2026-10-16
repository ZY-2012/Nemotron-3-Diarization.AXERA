# 模型

量化后的 axmodel 与运行时资产发布在 HuggingFace：

- **https://huggingface.co/HY-2012/Nemotron-3-Diarization.AXERA**

包含：

| 文件 | 说明 |
|---|---|
| `models/preencode.axmodel` | mel → chunk embedding，offline 档（U16/S8） |
| `models/encoder.axmodel` | offline：spkcache 264 + fifo 40 + chunk 380（U16/S8，121 MB） |
| `models/preencode_lowlat.axmodel` | mel → chunk embedding，低延迟档（U16/S8） |
| `models/encoder_lowlat.axmodel` | 低延迟：spkcache 264 + fifo 264 + chunk 10（U16/S8，111 MB） |
| `models/learnable_sil_emb.npy` | AOSC 压缩用可学习静音嵌入（512） |
| `models/feature_window.npy` | 检查点内量化后的 400 点 hann 窗（必须复用） |
| `models/mel_filterbank.npy` | 检查点内量化后的 128×257 mel 滤波器组 |
| `models/model_meta.json` | 图接口契约（形状 / 帧率 / mel 参数） |

本仓库不包含任何模型与二进制；转换与编译方法见
[`../model_convert/README.md`](../model_convert/README.md) 与
[`../python/README.md`](../python/README.md)。
