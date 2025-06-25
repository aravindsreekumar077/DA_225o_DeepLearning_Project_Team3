import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    raw: str

# Matches top-level {...} JSON objects lazily, even across lines
JSON_RE = re.compile(r"\{.*?\}\}", re.DOTALL)

from json import JSONDecoder, JSONDecodeError

def find_calls(text):
    dec = JSONDecoder()
    i = 0
    calls = []
    text = text.replace("'", '"')
    while i < len(text):
        try:
            obj, off = dec.raw_decode(text[i:])
        except JSONDecodeError:
            i += 1
            continue
        raw = text[i:i+off]
        i += off
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("parameters") or obj.get("arguments") or {}
            calls.append(ToolCall(name=obj["name"], args=args, raw=raw))
    return calls

__all__ = ["ToolCall", "find_calls"]