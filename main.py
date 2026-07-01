#!/usr/bin/env python3
"""
AI Candidate Ranking Pipeline — Main Orchestrator

Usage:
    python main.py --jd <path_to_jd.txt> --output submission.csv [--candidates <path>]

Stages 1 and 2 run in parallel via ThreadPoolExecutor.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import settings
from data.loader import stream_candidates
from pipeline.stage1_prefilter import run_stage1
from pipeline.stage2_ideal_profile import run_stage2
from pipeline.stage3_reranker import run_stage3
from pipeline.stage4_signal_blender import run_stage4
from pipeline.stage5_llm_ranker import run_stage5
from output.formatter import write_submission
from output.validator import validate_submission


def load_candidates_by_id(candidate_ids: list[str], candidates_file: str) -> dict:
    """Stream candidates file and collect only the ones we need."""
    needed = set(candidate_ids)
    result = {}
    for candidate in stream_candidates(candidates_file):
        if candidate.candidate_id in needed:
            result[candidate.candidate_id] = candidate
        if len(result) == len(needed):
            break
    return result


def main():
    parser = argparse.ArgumentParser(description="AI Candidate Ranking Pipeline")
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--candidates", default=settings.candidates_file)
    args = parser.parse_args()

    jd_text = Path(args.jd).read_text(encoding="utf-8")

    print("[1/5] Running Stage 1 (miniLM pre-filter) and Stage 2 (ideal profile) in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_stage1 = executor.submit(run_stage1, jd_text)
        future_stage2 = executor.submit(run_stage2, jd_text)
        top_candidate_ids = future_stage1.result()
        ideal_profile = future_stage2.result()

    print(f"  Stage 1: {len(top_candidate_ids)} candidates pre-filtered")
    print(f"  Stage 2: ideal profile generated ({len(ideal_profile.required_skills)} required skills)")

    print(f"[2/5] Loading top-{len(top_candidate_ids)} candidate profiles from disk...")
    all_candidates = load_candidates_by_id(top_candidate_ids, args.candidates)
    print(f"  Loaded {len(all_candidates)} candidates")

    print("[3/5] Running Stage 3 (Gemma multi-dim re-scoring)...")
    stage3_results = run_stage3(ideal_profile, top_candidate_ids, all_candidates)
    print(f"  Scored {len(stage3_results)} candidates")

    print("[4/5] Running Stage 4 (Redrob signal blending)...")
    shortlist = run_stage4(stage3_results, all_candidates)
    print(f"  Shortlisted {len(shortlist)} candidates")

    print("[5/5] Running Stage 5 (Gemini Pro final ranking)...")
    final_results = run_stage5(shortlist, all_candidates, ideal_profile)
    print(f"  Final ranking: {len(final_results)} candidates")

    write_submission(final_results, args.output)

    errors = validate_submission(args.output)
    if errors:
        print(f"\nWARNING: Submission validation failed ({len(errors)} issue(s)):")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"\nSubmission written and validated: {args.output}")


if __name__ == "__main__":
    main()
