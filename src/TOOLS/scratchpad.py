# TOOLS/scratchpad.py

class ScratchpadManager:
    def __init__(self):
        self.enabled = False
        self.notes = []

    def toggle(self, status: bool):
        self.enabled = status
        if not status:
            full_text = "\n".join(self.notes).strip()
            self.notes.clear()
            if full_text:
                return f"📝 Scratchpad summary:\n{full_text}"  # placeholder summary
            else:
                return "📝 Scratchpad was empty."
        else:
            return ""

    def add_note(self, text: str):
        self.notes.append(text)

    def is_enabled(self):
        return self.enabled
