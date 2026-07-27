#!/usr/bin/env bash
# check_project_status.sh
# Run from the root of p300-llm-speller/
# Gives a quick snapshot of repo state, implementation progress, and known issues.

set -uo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

section() {
    echo ""
    echo -e "${BOLD}=== $1 ===${NC}"
}

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
bad()  { echo -e "  ${RED}[MISSING]${NC} $1"; }

# ---------------------------------------------------------------------------
section "Repo root check"
if [ ! -d ".git" ]; then
    warn "No .git directory found here — are you in the repo root?"
else
    ok "Git repo detected"
fi

# ---------------------------------------------------------------------------
section "Git status"
if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
    echo "Current branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo ""
    echo "Uncommitted changes:"
    git status --short | sed 's/^/  /'
    echo ""
    echo "Last 5 commits:"
    git log --oneline -5 2>/dev/null | sed 's/^/  /'
else
    warn "git not available or not a repo"
fi

# ---------------------------------------------------------------------------
section "Expected directory structure"
EXPECTED_DIRS=(
    "src/data"
    "src/preprocessing"
    "src/models"
    "src/evaluation"
    "scripts"
    "results"
    "notebooks"
    "paper"
)
for d in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$d" ]; then
        n=$(find "$d" -type f -name "*.py" 2>/dev/null | wc -l | tr -d ' ')
        ok "$d/  ($n .py files)"
    else
        bad "$d/"
    fi
done

# ---------------------------------------------------------------------------
section "Key implementation files"
KEY_FILES=(
    "src/preprocessing/edf_parser.py"
    "src/preprocessing/segmentation.py"
    "src/models/fusion.py"
    "src/evaluation/run_pipeline.py"
    "requirements.txt"
    "setup_env.sh"
    "README.md"
    "P300_LLM_Fusion_Implementation_Plan.md"
)
for f in "${KEY_FILES[@]}"; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f" 2>/dev/null | tr -d ' ')
        ok "$f  ($lines lines)"
    else
        # try to find it anywhere in repo in case naming differs
        found=$(find . -name "$(basename "$f")" -not -path "*/.git/*" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            warn "$f not at expected path — found at: $found"
        else
            bad "$f"
        fi
    fi
done

# ---------------------------------------------------------------------------
section "Known issue: num_classes hardcoding (6x6 vs 9x8)"
HARDCODE_HITS=$(grep -rn "num_classes" --include="*.py" . 2>/dev/null | grep -E "36|= 36" )
if [ -n "$HARDCODE_HITS" ]; then
    warn "Found possible hardcoded num_classes=36 references:"
    echo "$HARDCODE_HITS" | sed 's/^/    /'
else
    ok "No obvious hardcoded num_classes=36 found (may already be fixed, or check manually)"
fi

echo ""
echo "  All num_classes references (for manual review):"
grep -rn "num_classes" --include="*.py" . 2>/dev/null | sed 's/^/    /' || echo "    none found"

# ---------------------------------------------------------------------------
section "StimulusCode / metadata handling check"
if grep -rq "epochs.metadata" --include="*.py" . 2>/dev/null; then
    ok "epochs.metadata usage found (StimulusCode fix likely present)"
    grep -rn "epochs.metadata" --include="*.py" . 2>/dev/null | sed 's/^/    /'
else
    bad "No epochs.metadata usage found — StimulusCode fix may not be applied"
fi

# ---------------------------------------------------------------------------
section "Segmentation logic check"
if grep -rlq "sequences_per_selection" --include="*.py" . 2>/dev/null; then
    ok "Fixed-block segmentation (sequences_per_selection) found"
fi
if grep -rlq "Dyn" --include="*.py" src/preprocessing 2>/dev/null; then
    ok "Dyn/DynBigram-specific handling found in preprocessing"
else
    warn "No Dyn-specific handling detected in src/preprocessing — verify majority-voting logic is present"
fi

# ---------------------------------------------------------------------------
section "Fusion engine status"
if [ -f "src/models/fusion.py" ]; then
    lines=$(wc -l < "src/models/fusion.py" | tr -d ' ')
    todos=$(grep -c "TODO\|FIXME\|NotImplemented\|pass  #" "src/models/fusion.py" 2>/dev/null || echo 0)
    echo "  fusion.py: $lines lines, $todos TODO/FIXME/stub markers"
    if [ "$lines" -lt 20 ]; then
        warn "fusion.py looks like a stub — core fusion logic likely not implemented yet"
    fi
else
    bad "src/models/fusion.py not found — fusion engine not started"
fi

# ---------------------------------------------------------------------------
section "TODO / FIXME scan (whole repo)"
TODO_COUNT=$(grep -rn "TODO\|FIXME" --include="*.py" . 2>/dev/null | wc -l | tr -d ' ')
echo "  Total TODO/FIXME markers in .py files: $TODO_COUNT"
grep -rn "TODO\|FIXME" --include="*.py" . 2>/dev/null | head -20 | sed 's/^/    /'
if [ "$TODO_COUNT" -gt 20 ]; then
    echo "    ... ($((TODO_COUNT - 20)) more not shown)"
fi

# ---------------------------------------------------------------------------
section "Results / outputs check"
if [ -d "results" ]; then
    n=$(find results -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -eq 0 ]; then
        warn "results/ exists but is empty — no pipeline runs completed yet"
    else
        ok "results/ contains $n file(s)"
        find results -type f 2>/dev/null | sed 's/^/    /'
    fi
fi

# ---------------------------------------------------------------------------
section "Python environment check"
if [ -f "requirements.txt" ]; then
    ok "requirements.txt found ($(wc -l < requirements.txt | tr -d ' ') packages listed)"
fi
if command -v python3 >/dev/null 2>&1; then
    echo "  Python: $(python3 --version 2>&1)"
    for pkg in mne moabb numpy scikit-learn torch; do
        if python3 -c "import $pkg" 2>/dev/null; then
            ver=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
            ok "$pkg installed (v$ver)"
        else
            warn "$pkg not importable in current environment"
        fi
    done
fi

# ---------------------------------------------------------------------------
section "Summary"
echo "  Review the [MISSING] and [WARN] items above."
echo "  Priority per project plan: fix num_classes hardcoding -> fusion engine -> pipeline validation vs Study D."
echo ""