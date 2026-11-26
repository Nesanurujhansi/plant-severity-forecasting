"""Documentation Rule Validator script for crop-prognosis.

Verifies that all phase documentation files in docs/ (e.g. phase00_setup.md)
contain all 12 mandatory section headers specified in the project documentation rules.
"""

import sys
import glob
import re
from pathlib import Path

REQUIRED_SECTIONS = [
    ("1. What this phase does", [r"1\.\s*in\s*one\s*sentence", r"what\s*this\s*phase\s*does", r"phase\s*overview"]),
    ("2. Why this step exists", [r"2\.\s*why\s*this\s*step\s*exists", r"why\s*this\s*step\s*exists", r"real-world\s*problem"]),
    ("3. Big-picture diagram", [r"3\.\s*big-picture\s*diagram", r"big-picture\s*diagram", r"architecture\s*diagram"]),
    ("4. Walkthrough of files created", [r"4\.\s*walkthrough", r"walkthrough\s*of\s*every\s*file", r"file\s*walkthrough"]),
    ("5. Worked example", [r"5\.\s*worked\s*example", r"worked\s*example"]),
    ("6. How to run it", [r"6\.\s*how\s*to\s*run", r"how\s*to\s*run\s*it", r"execution\s*instructions"]),
    ("7. What you should see", [r"7\.\s*what\s*you\s*should\s*see", r"what\s*you\s*should\s*see", r"expected\s*outputs"]),
    ("8. Common errors and how to fix them", [r"8\.\s*common\s*errors", r"common\s*errors"]),
    ("9. Self-check questions", [r"9\.\s*self-check\s*questions", r"self-check\s*questions"]),
    ("10. Honest limitations", [r"10\.\s*honest\s*limitations", r"honest\s*limitations", r"limitations\s*and\s*what\s*could\s*go\s*wrong"]),
    ("11. Viva questions", [r"11\.\s*viva\s*questions", r"viva\s*questions"]),
    ("12. Definition of Done", [r"12\.\s*definition\s*of\s*done", r"definition\s*of\s*done"])
]

def validate_phase_doc(filepath: Path) -> list[str]:
    """Check a single phase documentation file for required sections."""
    errors = []
    content = filepath.read_text(encoding="utf-8")
    
    for section_name, patterns in REQUIRED_SECTIONS:
        found = False
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found = True
                break
        if not found:
            errors.append(f"Missing section: '{section_name}'")
            
    return errors

def main() -> int:
    docs_dir = Path("docs")
    phase_files = sorted(docs_dir.glob("phase*.md"))
    
    if not phase_files:
        print("WARNING: No phase documentation files (phase*.md) found in docs/")
        return 0
        
    total_errors = 0
    print(f"Validating {len(phase_files)} phase documentation file(s)...")
    
    for phase_file in phase_files:
        print(f"\nChecking {phase_file}...")
        errors = validate_phase_doc(phase_file)
        if errors:
            total_errors += len(errors)
            for err in errors:
                print(f"  [ERROR] {err}")
        else:
            print("  [PASS] All 12 required sections present.")
            
    if total_errors > 0:
        print(f"\nValidation FAILED with {total_errors} missing section(s).")
        return 1
        
    print("\nValidation PASSED. All phase documentation complies with standards.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
