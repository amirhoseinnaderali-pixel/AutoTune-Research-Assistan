import json
from typing import Any, Dict

def parse_json_object(response: str) -> Dict[str, Any]:
    text = (response or '').strip()
    if not text:
        return {}
    if text.startswith('```json'):
        text = text[7:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
    elif text.startswith('```'):
        text = text[3:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}