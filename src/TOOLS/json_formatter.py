import json

def format_json_from_text(text: str) -> str:
    try:
        data = json.loads(text)
        return "🧾 Formatted JSON:\n```json\n" + json.dumps(data, indent=2) + "\n```"
    except Exception as e:
        return f"❌ Could not format JSON:\n`{str(e)}`"
