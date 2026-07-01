import csv
from models.scoring import FinalRankedCandidate

def write_submission(results: list[FinalRankedCandidate], output_path: str) -> None:
    sorted_results = sorted(results, key=lambda x: x.rank)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in sorted_results:
            writer.writerow([r.candidate_id, r.rank, f"{r.score:.4f}", r.reasoning])
