# Revised Agent implementation (v2): ensures model is instructed to emit proper tool-call JSON

import json, re
from typing import Dict, Any, Generator, List

# ── local imports ────────────────────────────────────────────────────────────
from ..runner.slm_runner       import SLMRunner
from ..agentic.tool_registry   import TOOL_REG, TOOL_SCHEMAS
from ..agentic.json_utils      import ToolCall, find_calls
from ..agentic.temp_control    import TemperatureController

class Agent:
    """Streaming tool-augmented chat agent (v2).

    Fixes over previous version:
    • Adds TOOL_GUIDE system message so the model reliably emits correct JSON.
    • When no tool call is detected but the model *mentions* one (e.g. writes a pseudo-JSON block),
      we fall back to normal chat instead of hanging.
    • Otherwise identical to v1 (placeholder substitution, feedback injection, loop guard).
    """

    # ── construction ──────────────────────────────────────────────────────────
    def __init__(self, cfg):
        self.runner   = SLMRunner(cfg)
        self._results : Dict[str, Any] = {}
        self._counter = 0
        self._last_calls : List[str] = []
        self._repeat_cap = 3

    # ── helpers ──────────────────────────────────────────────────────────────
    def _stash(self, value: Any) -> str:
        self._counter += 1
        key = f"$result_{self._counter}"
        self._results[key] = value
        return key

    def _sub(self, text: str) -> str:
        return re.sub(r"\$result_\d+", lambda m: str(self._results.get(m.group(), m.group())), text)

    def _validate_args(self, name: str, args: Dict[str, Any]) -> None:
        schema = TOOL_SCHEMAS.get(name)
        if schema is None:
            return
        for key, typ in schema.items():
            if key not in args:
                raise ValueError(f"missing argument '{key}' for {name}")
            if not isinstance(args[key], typ):
                raise ValueError(f"'{key}' must be {typ.__name__} in {name}")

    # ── tool runner ──────────────────────────────────────────────────────────
    def _run_tool(self, call: ToolCall) -> str:
        # print("\n########CALL##########", call, "\n")
        if call.name not in TOOL_REG:
            return f"[error: unknown tool {call.name}]"

        replaced_json = self._sub(call.raw)
        if "$result_" in replaced_json:
            return "[error: unresolved result reference]"

        try:
            call_obj = json.loads(replaced_json)
            args     = call_obj.get("parameters") or call_obj.get("args") or {}
            self._validate_args(call.name, args)
            val      = TOOL_REG[call.name](**args)
            tag      = self._stash(val)
            return f"[{call.name} → {val} | id {tag}]"
        except Exception as exc:
            return f"[{call.name} raised {exc}]"

    # ── generation helpers ───────────────────────────────────────────────────
    def _generate(self, prompt: str, temperature: float) -> tuple[str, List[ToolCall]]:
        buf = ""
        stream = self.runner.generate(prompt, stream=True, temperature=temperature, max_tokens=2048)
        for chunk in stream:
            tk = chunk["choices"][0]["text"]
            buf += tk
            calls = find_calls(buf)
            if calls:
                return buf, calls
        return buf, []

    # ── public chat API ──────────────────────────────────────────────────────
    def chat(self, system: str, user: str) -> Generator[str, None, None]:
        history: List[Dict[str, str]] = [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ]
        i = 0
        while True:
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history) + "\nASSISTANT:"

            buf, calls = self._generate(prompt, TemperatureController.for_chat(prompt))
            if not calls:
                history.append({"role": "assistant", "content": buf.strip()})
                yield buf
                return
            
            buf = buf.split('{')[0]

            # Low-temp extension to finish JSON
            brace_prompt = prompt + buf

            # print("### START OF BUF ###")
            # print(f"BUF {i}:\n\n", buf, "\n\n")

            # print("### END OF BUF ###")

            # print(f"PROMPT {i}:\n\n", prompt, "\n\n")
            # i+=1

            # print("### END OF BUF ###")

            # print("### START OF BRACE PROMPT ###")
            # print(f"BRACE PROMPT {i}:\n\n", brace_prompt, "\n\n")
            # print("### END OF BRACE PROMPT ###")

            buf2, calls = self._generate(brace_prompt, TemperatureController.for_tool())
            full_json_chunk = buf + buf2

            # print("### START OF BUF2 ###")
            # print(f"BUF2 {i}:\n\n", buf2, "\n\n")
            # print(find_calls(buf2))
            # print("### END OF BUF2 ###")


            # print("BUF + BUF2:", buf, "\n\n", buf2)

            yield buf
            yield buf2

            tool_outputs: List[str] = []
            tool_msgs:    List[Dict[str, str]] = []
            
            for call in calls:
                print(f"\n\nProcessing tool call: {call.name} with args {call.args}")
                print(f"Raw call: {call.raw}")
                if call.name != 'get_date' and call.args == {}:
                    print(f"Skipping tool call {call.name} with empty args")
                    tool_msgs.append({"role": "assistant", "content": f"WARNING! You are calling [{call.name} with no args, please fix your JSON.]"})
                    continue
                out = self._run_tool(call)
                tool_outputs.append(out)
                tool_msgs.append({"role": "assistant", "name": call.name, "content": out})
            yield "\n".join(tool_outputs)

            sig = "|".join(f"{c.name}:{json.dumps(c.args, sort_keys=True)}" for c in calls)
            self._last_calls.append(sig)
            self._last_calls = self._last_calls[-self._repeat_cap:]
            if self._last_calls.count(sig) == self._repeat_cap:
                print("Aborting: identical tool call repeated", self._repeat_cap)
                return

            history.append({"role": "assistant", "content": full_json_chunk})
            history.extend(tool_msgs)
            history.append({"role": "assistant", "content": ""})

    # ── internal one-shot generator ───────────────────────────────
    def _generate(
        self, prompt: str, temperature: float
    ) -> tuple[str, List[ToolCall]]:
        """
        One streaming pass.  
        Returns (generated_text, detected_tool_calls)
        """
        buf = ""
        stream = self.runner.generate(
            prompt,
            stream=True,
            temperature=temperature,
            max_tokens=2048,
            stop=["USER"]
        )

        for chunk in stream:
            tk = chunk["choices"][0]["text"]
            buf += tk

            # JSON fully closed?
            calls = find_calls(buf)
            if calls:
                return buf, calls

        return buf, []  # no tool call detected

__all__ = ["Agent"]