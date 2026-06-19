"""
prompt_lab/wizard.py: Friendly interactive wizard for non-technical users.

When a user runs `eval-grid prompt-lab` with no arguments (or with --wizard),
this guides them step-by-step through:
  1. Pick a prompt from the library  OR  paste a new one
  2. Choose which LLMs to test
  3. Watch live progress
  4. Open the HTML report automatically
"""

from __future__ import annotations

import os
import re
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

from prompt_lab.library import list_prompts, all_categories, Prompt, save_prompt


CATEGORIES_DISPLAY = {
    "requirements_analysis":  "Requirements Analysis",
    "automation_testing":     "Automation Testing",
    "performance_testing":    "Performance Testing",
    "test_governance":        "Test Governance",
    "testing_artefacts":      "Testing Artefacts",
    "day_to_day":             "Day-to-Day Testing",
}

LLM_OPTIONS = {
    "1": "ChatGPT",
    "2": "Gemini",
    "3": "Copilot",
    "4": "All three",
}


def _hr(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _header() -> None:
    print()
    _hr("═")
    print("  🧪  Prompt Lab — Multi-LLM Prompt Tester")
    print("  Quality Engineering Unit — EvalGrid")
    _hr("═")
    print()


def _pick_from_list(items: List[str], title: str) -> int:
    """Show a numbered list and return the 0-based index chosen."""
    print(f"\n{title}")
    _hr()
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    print()
    while True:
        raw = input("  Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return int(raw) - 1
        print(f"  ⚠️  Please enter a number between 1 and {len(items)}.")


def _choose_prompt() -> tuple[str, str, str]:
    """
    Returns (prompt_id, prompt_title, prompt_text).
    User can pick from library or paste their own.
    """
    print("  Where is your prompt?")
    _hr()
    print("  1. Pick from the Prompt Library")
    print("  2. Paste / type a new prompt")
    print()
    choice = input("  Enter 1 or 2: ").strip()

    if choice == "1":
        categories = all_categories()
        if not categories:
            print("\n  ⚠️  Prompt library is empty. Switching to paste mode.\n")
            return _paste_prompt()

        display = [CATEGORIES_DISPLAY.get(c, c.replace("_", " ").title()) for c in categories]
        idx = _pick_from_list(display + ["↩ Back (paste instead)"], "Choose a category:")
        if idx == len(display):
            return _paste_prompt()

        category = categories[idx]
        prompts = list_prompts(category=category)
        if not prompts:
            print(f"\n  ⚠️  No prompts in '{display[idx]}' yet. Switching to paste mode.\n")
            return _paste_prompt()

        prompt_labels = [f"{p.title}  (v{p.version})" for p in prompts]
        pidx = _pick_from_list(prompt_labels, f"Choose a prompt from {display[idx]}:")
        p = prompts[pidx]
        print(f"\n  Loaded: {p.title}\n")
        return p.id, p.title, p.prompt

    return _paste_prompt()


def _paste_prompt() -> tuple[str, str, str]:
    print("\n  Enter a title for this prompt (e.g. 'BDD Scenario Generator'):")
    title = input("  Title: ").strip() or "Untitled Prompt"

    print("\n  Paste your prompt below.")
    print("  When finished, type END on its own line and press Enter.\n")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    prompt_text = "\n".join(lines).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    prompt_id = f"custom-{slug}"

    save_choice = input("\n  Save this prompt to the library? (y/n): ").strip().lower()
    if save_choice == "y":
        cats = all_categories() or ["day_to_day"]
        display = [CATEGORIES_DISPLAY.get(c, c.replace("_", " ").title()) for c in cats]
        cidx = _pick_from_list(display, "Which category?")
        category = cats[cidx]
        p = Prompt(id=prompt_id, title=title, category=category, prompt=prompt_text)
        path = save_prompt(p)
        print(f"  ✅  Saved to {path}")

    return prompt_id, title, prompt_text


def _choose_llms() -> List[str]:
    from prompt_lab.runner import LLM_NAMES
    print("\n  Which LLMs should I test?")
    _hr()
    for k, v in LLM_OPTIONS.items():
        print(f"  {k}. {v}")
    print()
    while True:
        choice = input("  Enter 1, 2, 3, or 4: ").strip()
        if choice == "4":
            return list(LLM_NAMES)
        if choice in ("1", "2", "3"):
            return [LLM_OPTIONS[choice]]
        print("  ⚠️  Please enter 1, 2, 3, or 4.")


def run_wizard(output_dir: str = "output/prompt_lab") -> Optional[str]:
    """
    Run the interactive wizard.  Returns the path to the HTML report.
    """
    _header()

    # Check at least one API key is configured
    has_key = any([
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GITHUB_TOKEN"),
    ])
    if not has_key:
        print("  ⚠️  No API keys found in your .env file.")
        print("  Please add at least one of:")
        print("     OPENAI_API_KEY   — for ChatGPT")
        print("     GEMINI_API_KEY   — for Gemini")
        print("     GITHUB_TOKEN     — for Copilot")
        print()
        print("  Edit your .env file and run again.")
        return None

    prompt_id, prompt_title, prompt_text = _choose_prompt()
    llms = _choose_llms()

    print(f"\n  Testing '{prompt_title}' on: {', '.join(llms)}")
    print("  This may take up to 60 seconds...")
    print()

    # --- Run ---
    from prompt_lab.runner import run_all_sync
    from prompt_lab.evaluator import evaluate
    from prompt_lab.report import generate_html, print_summary

    print("  [1/3] Sending prompt to LLMs...", end=" ", flush=True)
    llm_results = run_all_sync(prompt_text, llms=llms)
    print("done")

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_anthropic:
        print("  [2/3] Scoring — ANTHROPIC_API_KEY not set, skipping auto-score")
    else:
        print("  [2/3] Scoring responses with AI judge...", end=" ", flush=True)

    report = evaluate(
        prompt_id=prompt_id,
        prompt_title=prompt_title,
        prompt_text=prompt_text,
        llm_results=llm_results,
        auto_fix=has_anthropic,
    )
    if has_anthropic:
        print("done")

    print("  [3/3] Generating HTML report...", end=" ", flush=True)
    safe_id = re.sub(r"[^a-z0-9_-]", "_", prompt_id.lower())
    report_path = str(Path(output_dir) / f"{safe_id}_report.html")
    generate_html(report, report_path)
    print("done")

    print_summary(report)
    print(f"  📄 HTML report: {report_path}")

    if report.fixed_prompt:
        print(f"\n  💡 A fixed prompt is included in the report.")
        save_choice = input("  Apply fixed prompt and save as next version? (y/n): ").strip().lower()
        if save_choice == "y":
            _apply_fix(prompt_id, prompt_title, prompt_text, report.fixed_prompt)

    open_choice = input("\n  Open report in browser? (y/n): ").strip().lower()
    if open_choice == "y":
        webbrowser.open(f"file://{Path(report_path).resolve()}")

    return report_path


def _apply_fix(prompt_id: str, title: str, original: str, fixed: str) -> None:
    """Save the fixed prompt as a new version in the library."""
    from prompt_lab.library import load_prompt, save_prompt, Prompt
    try:
        p = load_prompt(prompt_id)
        p.version += 1
        p.notes = f"Auto-fixed from v{p.version-1} by Prompt Lab"
        p.prompt = fixed
        path = save_prompt(p)
        print(f"  ✅  Saved as version {p.version} → {path}")
    except FileNotFoundError:
        # Prompt was pasted, not from library — save it now
        cats = all_categories() or ["day_to_day"]
        p = Prompt(id=prompt_id, title=title, category=cats[0], prompt=fixed, version=2,
                   notes="Auto-fixed by Prompt Lab")
        path = save_prompt(p)
        print(f"  ✅  Saved → {path}")
