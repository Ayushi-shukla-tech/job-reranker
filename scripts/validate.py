#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from output.validator import validate_submission

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate.py <submission.csv>")
        sys.exit(1)
    errors = validate_submission(sys.argv[1])
    if errors:
        print(f"Validation failed ({len(errors)} issue(s)):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("Submission is valid.")

if __name__ == "__main__":
    main()
