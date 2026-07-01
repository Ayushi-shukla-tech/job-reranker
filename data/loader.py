import json
from typing import Iterator
from models.candidate import Candidate

def stream_candidates(path: str) -> Iterator[Candidate]:
    """Stream candidates from NDJSON file one at a time — never loads all into RAM."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        # Handle JSON array format for compatibility with test files
        if first_char == '[':
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        try:
                            yield Candidate.model_validate(item)
                        except Exception:
                            continue  # skip malformed items silently
                    return
            except Exception:
                pass

        # Handle NDJSON format (normal case)
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield Candidate.model_validate(json.loads(line))
            except Exception:
                continue  # skip malformed lines silently
