"""run_code 运行时：协议、桥接、worker 与执行器。"""

from .bridge import Bridge, create_socketpair
from .executor import RunCodeExecutor, RunCodeResult
from .protocol import (
    MAX_FRAME_BYTES,
    decode_frame,
    encode_frame,
    read_frame,
    read_frame_sync,
    write_frame,
    write_frame_sync,
)

__all__ = [
    "Bridge",
    "MAX_FRAME_BYTES",
    "RunCodeExecutor",
    "RunCodeResult",
    "create_socketpair",
    "decode_frame",
    "encode_frame",
    "read_frame",
    "read_frame_sync",
    "write_frame",
    "write_frame_sync",
]
