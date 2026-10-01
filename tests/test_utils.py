from utils import parse_json_object

def test_plain_json():
    assert parse_json_object('{"task": "code"}') == {"task": "code"}

def test_markdown_json():
    assert parse_json_object('```json\n{"task": "reasoning"}\n```') == {"task": "reasoning"}

def test_embedded_json():
    result = parse_json_object('Here is the result:\n{"task": "qa"}\nDone.')
    assert result["task"] == "qa"

def test_invalid_json():
    assert parse_json_object('not json') == {}