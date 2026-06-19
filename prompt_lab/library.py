"""
prompt_lab/library.py: Manage the YAML prompt library.

Prompts are stored as individual YAML files under prompts/<category>/<id>.yaml.
Each file follows this schema:

    id:          req-analysis-001          # unique slug
    title:       Requirements Analysis     # human-readable name
    category:    requirements_analysis     # folder category
    version:     1                         # increment when you update the prompt
    tags:        [requirements, bdd]       # free-form tags for filtering
    prompt: |
        Analyze the following requirements...

This module provides helpers to load, save, list, and search prompts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


PROMPTS_ROOT = Path(__file__).parent.parent / "prompts"


@dataclass
class Prompt:
    id: str
    title: str
    category: str
    prompt: str
    version: int = 1
    tags: List[str] = field(default_factory=list)
    notes: str = ""        # optional human notes
    file_path: str = ""    # set automatically when loaded from disk


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def _prompt_path(category: str, prompt_id: str) -> Path:
    return PROMPTS_ROOT / category / f"{prompt_id}.yaml"


def load_prompt(prompt_id: str, category: Optional[str] = None) -> Prompt:
    """
    Load a prompt by id.  If category is not given, search all subdirectories.
    Raises FileNotFoundError when the prompt cannot be found.
    """
    if category:
        path = _prompt_path(category, prompt_id)
    else:
        matches = list(PROMPTS_ROOT.rglob(f"{prompt_id}.yaml"))
        if not matches:
            raise FileNotFoundError(
                f"No prompt found with id '{prompt_id}'. "
                f"Run 'eval-grid prompt-lab list' to see available prompts."
            )
        path = matches[0]

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    p = Prompt(**{k: v for k, v in data.items() if k in Prompt.__dataclass_fields__})
    p.file_path = str(path)
    return p


def save_prompt(prompt: Prompt) -> str:
    """
    Save (or overwrite) a prompt YAML file.  Returns the path written to.
    """
    dest_dir = PROMPTS_ROOT / prompt.category
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{prompt.id}.yaml"

    data = asdict(prompt)
    data.pop("file_path", None)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    prompt.file_path = str(path)
    return str(path)


# ---------------------------------------------------------------------------
# List / Search
# ---------------------------------------------------------------------------

def list_prompts(category: Optional[str] = None, tag: Optional[str] = None) -> List[Prompt]:
    """Return all prompts, optionally filtered by category and/or tag."""
    if not PROMPTS_ROOT.exists():
        return []

    root = PROMPTS_ROOT / category if category else PROMPTS_ROOT
    prompts = []
    for yaml_file in sorted(root.rglob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            p = Prompt(**{k: v for k, v in data.items() if k in Prompt.__dataclass_fields__})
            p.file_path = str(yaml_file)
            if tag and tag not in p.tags:
                continue
            prompts.append(p)
        except Exception:
            pass  # skip malformed files

    return prompts


def all_categories() -> List[str]:
    """Return category subfolder names that exist under prompts/."""
    if not PROMPTS_ROOT.exists():
        return []
    return sorted(d.name for d in PROMPTS_ROOT.iterdir() if d.is_dir())
