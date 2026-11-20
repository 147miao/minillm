"""桥接帧编解码：UTF-8 JSON lines，单帧上限 8 MiB。

宿主与 run_code worker 之间只走 socketpair，协议帧不污染 stdout/stderr。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, BinaryIO

from ..errors import BridgeError

MAX_FRAME_BYTES = 8 * 1024 * 1024


def encode_frame(payload: Mapping[str, Any]) -> bytes:
    """把一条消息编码成单行 UTF-8 JSON（含换行）。"""

    try:
        text = json.dumps(
            dict(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        data = text.encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise BridgeError(f"无法编码桥接帧：{exc}") from exc
    if len(data) > MAX_FRAME_BYTES:
        raise BridgeError(f"桥接帧超过上限 {MAX_FRAME_BYTES} 字节")
    return data


def decode_frame(line: bytes) -> dict[str, Any]:
    """把单行字节解码成 dict；空行/超限/非法 JSON 抛 BridgeError。"""

    if not line:
        raise BridgeError("桥接连接已关闭")
    if len(line) > MAX_FRAME_BYTES:
        raise BridgeError(f"桥接帧超过上限 {MAX_FRAME_BYTES} 字节")
    if not line.endswith(b"\n"):
        raise BridgeError("桥接帧缺少换行结尾（可能被截断）")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BridgeError("桥接帧不是合法 UTF-8") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"桥接帧不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise BridgeError("桥接帧必须是 JSON 对象")
    return payload


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """异步读一帧；EOF 返回 None。"""

    try:
        line = await reader.readline()
    except (ValueError, asyncio.LimitOverrunError) as exc:
        raise BridgeError(f"桥接帧读取失败（超过上限或流异常）：{exc}") from exc
    if not line:
        return None
    return decode_frame(line)


async def write_frame(writer: asyncio.StreamWriter, payload: Mapping[str, Any]) -> None:
    writer.write(encode_frame(payload))
    await writer.drain()


def read_frame_sync(stream: BinaryIO) -> dict[str, Any] | None:
    """worker 侧同步读一帧；EOF 返回 None。"""

    try:
        line = stream.readline(MAX_FRAME_BYTES + 2)
    except OSError as exc:
        raise BridgeError(f"桥接读取失败：{exc}") from exc
    if not line:
        return None
    if len(line) > MAX_FRAME_BYTES:
        raise BridgeError(f"桥接帧超过上限 {MAX_FRAME_BYTES} 字节")
    return decode_frame(line)


def write_frame_sync(stream: BinaryIO, payload: Mapping[str, Any]) -> None:
    """worker 侧同步写一帧。"""

    stream.write(encode_frame(payload))
    stream.flush()


__all__ = [
    "MAX_FRAME_BYTES",
    "decode_frame",
    "encode_frame",
    "read_frame",
    "read_frame_sync",
    "write_frame",
    "write_frame_sync",
]
