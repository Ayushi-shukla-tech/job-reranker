import csv, re
from pathlib import Path

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

def validate_submission(csv_path: str) -> list[str]:
    errors = []
    path = Path(csv_path)
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != REQUIRED_HEADER:
                errors.append(f"Header must be {REQUIRED_HEADER}, got {header}")
                return errors
            data_rows = [row for row in reader if any(c.strip() for c in row)]
    except Exception as e:
        return [f"Cannot read file: {e}"]

    if len(data_rows) != 100:
        errors.append(f"Expected 100 data rows, got {len(data_rows)}")

    seen_ids, seen_ranks, by_rank = set(), set(), []

    for i, cells in enumerate(data_rows):
        if len(cells) != 4:
            errors.append(f"Row {i+2}: expected 4 columns, got {len(cells)}")
            continue
        cid, rank_s, score_s, _ = cells
        cid = cid.strip()

        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {i+2}: invalid candidate_id '{cid}'")
        elif cid in seen_ids:
            errors.append(f"Row {i+2}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)

        try:
            rank = int(rank_s.strip())
            if not 1 <= rank <= 100:
                errors.append(f"Row {i+2}: rank {rank} out of range 1-100")
            elif rank in seen_ranks:
                errors.append(f"Row {i+2}: duplicate rank {rank}")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {i+2}: rank must be integer, got '{rank_s}'")
            rank = None

        try:
            score = float(score_s.strip())
            if cid and rank:
                by_rank.append((rank, score))
        except ValueError:
            errors.append(f"Row {i+2}: score must be float, got '{score_s}'")

    missing_ranks = set(range(1, 101)) - seen_ranks
    if missing_ranks:
        errors.append(f"Missing ranks: {sorted(missing_ranks)}")

    by_rank.sort()
    for i in range(len(by_rank) - 1):
        r1, s1 = by_rank[i]
        r2, s2 = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"Non-monotonic scores: rank {r1} score {s1} < rank {r2} score {s2}")

    return errors
