import csv, tempfile, os, pytest
from models.scoring import FinalRankedCandidate
from output.formatter import write_submission
from output.validator import validate_submission

def _make_100_results() -> list[FinalRankedCandidate]:
    return [
        FinalRankedCandidate(
            candidate_id=f"CAND_{str(i).zfill(7)}",
            rank=i,
            score=round(1.0 - (i - 1) * 0.009, 4),
            reasoning=f"Candidate {i} reasoning."
        )
        for i in range(1, 101)
    ]

def test_write_submission_creates_valid_csv():
    results = _make_100_results()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_submission(results, path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
        assert len(rows) == 101
        assert rows[1][0] == "CAND_0000001"
        assert rows[1][1] == "1"
    finally:
        os.unlink(path)

def test_validate_submission_passes_valid_csv():
    results = _make_100_results()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_submission(results, path)
        errors = validate_submission(path)
        assert errors == []
    finally:
        os.unlink(path)

def test_validate_submission_catches_duplicate_rank():
    results = _make_100_results()
    results[1] = FinalRankedCandidate(
        candidate_id="CAND_0000099", rank=1,
        score=0.9, reasoning="test"
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_submission(results, path)
        errors = validate_submission(path)
        assert any("duplicate" in e.lower() or "rank" in e.lower() for e in errors)
    finally:
        os.unlink(path)
