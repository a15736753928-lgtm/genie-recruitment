"""Download and cache the FunASR Paraformer model for Chinese speech recognition.

Run once before starting the server:
    python asr_init.py

The model (~1.8 GB) is downloaded from HuggingFace mirror (fast in China)
and cached locally.  Subsequent calls load from cache instantly.
"""

import os

# Use HuggingFace mirror for faster download in China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

MODEL_NAME = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

print(f"正在下载 FunASR Paraformer 模型: {MODEL_NAME}")
print("（首次下载约 1.8 GB，大文件下载失败会自动重试，请耐心等待...）")

import time

MAX_RETRIES = 5
pipe = None

for attempt in range(1, MAX_RETRIES + 1):
    try:
        pipe = pipeline(
            task=Tasks.auto_speech_recognition,
            model=MODEL_NAME,
        )
        break  # success
    except Exception as e:
        if attempt < MAX_RETRIES:
            wait = attempt * 10
            print(f"\n下载中断 (第 {attempt}/{MAX_RETRIES} 次): {e}")
            print(f"已下载的文件已缓存，{wait} 秒后继续下载剩余文件...\n")
            time.sleep(wait)
        else:
            print(f"\n重试 {MAX_RETRIES} 次后仍失败，请检查网络后重新运行。")
            raise

# Warm up with a quick test
import numpy as np
test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
result = pipe(test_audio)
text = result[0]["text"] if isinstance(result, list) else result["text"]
print(f"\n初始化完成 ✓  测试结果: {text}")
print("模型已就绪，可以启动服务器了。")
