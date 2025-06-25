import ast
import operator as op
import io
import requests
from datetime import datetime
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr

# safe math functions
from math import (
    sin, cos, tan, log, exp, sqrt, floor, ceil,
    asin, acos, atan, degrees, radians, pi, e, pow
)

class Tools:
    SAFE = {
        "sin": sin, "cos": cos, "tan": tan,
        "asin": asin, "acos": acos, "atan": atan,
        "log": log, "exp": exp, "sqrt": sqrt,
        "floor": floor, "ceil": ceil,
        "degrees": degrees, "radians": radians,
        "pi": pi, "e": e, "pow": pow
    }

    _OPS = {
        ast.Add: op.add, ast.Sub: op.sub,  ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg
    }

    # ── mini-evaluator ────────────────────────────────────────────
    @classmethod
    def _eval_node(cls, node):
        if isinstance(node, ast.Constant):           # numbers, strings, etc.
            return node.value
        if isinstance(node, ast.Name):               # variables / constants
            if node.id in cls.SAFE:
                return cls.SAFE[node.id]
            raise ValueError(f"Unrecognized name: {node.id}")
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls._OPS:
            return cls._OPS[type(node.op)](cls._eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in cls._OPS:
            return cls._OPS[type(node.op)](
                cls._eval_node(node.left),
                cls._eval_node(node.right)
            )
        if isinstance(node, ast.Call):               # function calls
            if (isinstance(node.func, ast.Name)
                    and node.func.id in cls.SAFE
                    and callable(cls.SAFE[node.func.id])):
                args = [cls._eval_node(a) for a in node.args]
                return cls.SAFE[node.func.id](*args)
            raise ValueError(f"Forbidden call: {ast.dump(node.func)}")
        raise ValueError("Forbidden expression")

    # ── public API ────────────────────────────────────────────────
    @classmethod
    def calculator(cls, expression: str) -> float:
        print(f"Calculating: {expression}")
        expr = expression.replace("^", "**")  # allow 2^3 style
        tree = ast.parse(expr, mode="eval").body
        return cls._eval_node(tree)

    @staticmethod
    def compare(expressions: Dict[str, str]) -> Dict[str, Any]:
        out = {}
        for label, expr in expressions.items():
            try:
                out[label] = Tools.calculator(expr)
            except Exception as e:
                out[label] = f"Error: {e}"
        return out

    @staticmethod
    def get_date() -> str:
        return datetime.now().strftime("%Y%m%d")

    @staticmethod
    def python_shell(code: str) -> Any:
        code = code.replace('\n', ';')
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        local_vars: Dict[str, Any] = {}
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, {}, local_vars)
        except Exception as e:
            return f"Script error: {e}"
        out = stdout_buf.getvalue().rstrip()
        err = stderr_buf.getvalue().rstrip()
        result = local_vars.get("result", None)
        parts = []
        if out: parts.append(f"Output:\n{out}")
        if err: parts.append(f"Errors:\n{err}")
        if result is not None: parts.append(f"Result: {result}")
        return "\n".join(parts) or None

    @staticmethod
    def get_weather_details(location: str) -> Dict[str, Any]:
        try:
            r = requests.get(f"https://wttr.in/{location}?format=j1", timeout=5)
            r.raise_for_status()
            cur = r.json()["current_condition"][0]
            return {
                "loc": location,
                "temp_C": cur["temp_C"],
                "weather": cur["weatherDesc"][0]["value"]
            }
        except Exception as ex:
            return {"error": str(ex), "loc": location}


TOOL_REG = {
    "calculator": Tools.calculator,
    "compare":    Tools.compare,
    "get_date":   Tools.get_date,
    "python_shell": Tools.python_shell,
    "get_weather_details": Tools.get_weather_details,
}

TOOL_SCHEMAS: Dict[str, Dict[str, type]] = {
    "calculator":         {"expression": str},
    "compare":            {"expressions": dict},
    "get_date":           {},
    "python_shell":       {"code": str},
    "get_weather_details": {"location": str},
}

__all__ = ["Tools", "TOOL_REG", "TOOL_SCHEMAS"]