import re

class TemperatureController:
    """
    Heuristic temperature control for generation:
    - Chat: adaptive based on prompt length and complexity
    - Tool calls: deterministic and low-temp
    """

    @staticmethod
    def for_chat(prompt: str) -> float:
        length = len(prompt.split())
        geek = bool(re.search(r"(def |class |{|\[|\]|\(|\))", prompt))
        base = 0.75 if length < 30 else 0.45
        return base - (0.2 if geek else 0)

    @staticmethod
    def for_tool() -> float:
        return 0.1  # Near-deterministic
        

__all__ = ["TemperatureController"]