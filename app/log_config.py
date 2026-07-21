"""集中式日志配置。

设计原则（对齐业界实践）：
  1. 单一入口 configure_logging()，全应用一种格式，杜绝多套格式混排。
  2. 白名单式降噪：root 抬到 WARNING，只放行自家 genie.* / app.* 到 INFO。
     未知三方库默认沉默，新引入的库不会再往启动日志里刷噪音。
  3. 已知吵闹的三方库显式压到 WARNING（faiss / modelscope / funasr / …）。
  4. 禁掉 HuggingFace / tqdm 进度条与 modelscope 冗余日志（靠环境变量，
     必须在导入这些库之前设置——见 main.py 顶部）。

只面向控制台/开发终端，不引入 structlog/JSON —— 那是集中式日志采集才需要的。
"""

from __future__ import annotations

import logging
import sys

# 自家日志命名空间：这些放行到 INFO，其余一律按 root(WARNING) 过滤。
_APP_LOGGERS = ("genie", "app")

# 已知会在启动/运行期刷屏的三方库，显式压到 WARNING。
_NOISY_LOGGERS = (
    "faiss", "faiss.loader",
    "modelscope",
    "funasr",
    "sentence_transformers",
    "transformers",
    "milvus_lite", "pymilvus",
    "apscheduler",
    "httpx", "httpcore",
    "urllib3",
    "huggingface_hub",
    "datasets",
    "matplotlib", "PIL", "numba", "jieba",
)


class _ColoredFormatter(logging.Formatter):
    """纯 ANSI 彩色日志格式器 —— 不依赖 TTY 检测，始终输出颜色。"""

    _LEVEL_COLORS = {
        logging.DEBUG:    "\033[36m",     # cyan
        logging.INFO:     "\033[32m",     # green
        logging.WARNING:  "\033[33m",     # yellow
        logging.ERROR:    "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m",   # bold red
    }
    _RESET = "\033[0m"
    _DIM = "\033[2m"
    _WHITE = "\033[37m"

    def format(self, record: logging.LogRecord) -> str:
        # 拷贝 record，避免污染原对象影响其他 handler
        record = logging.LogRecord(
            record.name, record.levelno, record.pathname, record.lineno,
            record.msg, record.args, record.exc_info,
            func=record.funcName, sinfo=record.stack_info,
        )
        color = self._LEVEL_COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname:8}{self._RESET}"
        record.name = f"{self._DIM}{record.name}{self._RESET}"
        record.asctime = self.formatTime(record, self.datefmt)
        # 延迟导入，规避潜在的导入顺序问题
        try:
            from app.middleware.trace import get_trace_id
            trace_id = get_trace_id()
        except Exception:
            trace_id = ""
        trace = f" {self._WHITE}[{trace_id}]{self._RESET}" if trace_id else ""
        return (
            f"{self._DIM}{record.asctime}{self._RESET} "
            f"{record.name} {record.levelname}{trace} {record.getMessage()}"
        )


def configure_logging(level: int = logging.INFO) -> None:
    """初始化全应用日志。幂等：重复调用只重置 root handler，不叠加。

    Args:
        level: 自家 genie.* / app.* 日志级别（默认 INFO）。
    """
    # UTF-8 输出：Windows 默认 GBK 会在 emoji/框线字符上抛 UnicodeEncodeError。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    root = logging.getLogger()
    # 清掉已有 handler（uvicorn/basicConfig/三方库可能提前加过），避免重复打印
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)

    # 白名单式降噪：root 只放 WARNING 及以上，自家命名空间放行到 INFO。
    root.setLevel(logging.WARNING)
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(level)

    # 已知吵闹三方库显式压到 WARNING（即便有的会自挂 handler，也先降级源头）。
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)

    # uvicorn 走自己的 log_config（UVICORN_LOG_CONFIG），这里不动它。


# uvicorn 着色日志配置（uvicorn 在非 TTY 下默认关色，这里强制开启）。
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": True,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "use_colors": True,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error":  {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}
