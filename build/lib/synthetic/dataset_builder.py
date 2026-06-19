"""
synthetic/dataset_builder.py: Load, validate, and build evaluation datasets.

An evaluation dataset is a collection of test cases: each one specifying an input
to send to the AI, what the expected output looks like, and how to judge the answer.

This module handles:
  - Loading test cases from JSON or YAML files on disk.
  - Validating that each test case has the required fields (using Pydantic models).
  - Building dataset variants: e.g. filter by capability or severity, or merge
    multiple files into a single dataset.

Supported file formats
----------------------
- JSON  (.json) : a list of test case objects.
- YAML  (.yaml / .yml): same structure, YAML syntax.

Usage
-----
    from synthetic.dataset_builder import DatasetBuilder
    builder = DatasetBuilder()
    cases = builder.load("datasets/my_tests.json")
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
import json
import yaml

class DatasetSchema(BaseModel):
    name: str
    description: str
    version: str = "1.0"
    test_cases: List[Dict[str, Any]] = []

class DatasetBuilder:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.test_cases = []
        self.metadata = {
            "name": name,
            "description": description,
            "version": "1.0",
            "created": None,
            "modified": None,
            "test_case_count": 0,
        }

    def add_test_case(self, test_case: Dict[str, Any]) -> None:
        required_fields = ["id", "input"]
        if not all(field in test_case for field in required_fields):
            raise ValueError(f"Test case must contain: {required_fields}")

        self.test_cases.append(test_case)
        self.metadata["test_case_count"] = len(self.test_cases)

    def add_test_cases(self, test_cases: List[Dict[str, Any]]) -> None:
        for test_case in test_cases:
            self.add_test_case(test_case)

    def validate_schema(self) -> bool:
        for test_case in self.test_cases:
            if "id" not in test_case or "input" not in test_case:
                return False
        return True

    def save_json(self, filepath: str) -> None:
        if not self.validate_schema():
            raise ValueError("Dataset schema validation failed")

        data = {
            "metadata": self.metadata,
            "test_cases": self.test_cases,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def save_yaml(self, filepath: str) -> None:
        if not self.validate_schema():
            raise ValueError("Dataset schema validation failed")

        data = {
            "metadata": self.metadata,
            "test_cases": self.test_cases,
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def load_json(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if "metadata" in data:
            self.metadata = data["metadata"]
            self.name = self.metadata.get("name", self.name)
            self.description = self.metadata.get("description", self.description)

        if "test_cases" in data:
            self.test_cases = data["test_cases"]

    def load_yaml(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if "metadata" in data:
            self.metadata = data["metadata"]
            self.name = self.metadata.get("name", self.name)
            self.description = self.metadata.get("description", self.description)

        if "test_cases" in data:
            self.test_cases = data["test_cases"]

    def filter_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        return [tc for tc in self.test_cases if tc.get("capability") == capability]

    def filter_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        return [tc for tc in self.test_cases if tc.get("severity") == severity]

    def filter_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        return [tc for tc in self.test_cases if tag in tc.get("risk_tags", [])]

    def get_statistics(self) -> Dict[str, Any]:
        capabilities = {}
        severities = {}
        tags = {}

        for tc in self.test_cases:
            cap = tc.get("capability", "unknown")
            capabilities[cap] = capabilities.get(cap, 0) + 1

            sev = tc.get("severity", "unknown")
            severities[sev] = severities.get(sev, 0) + 1

            for tag in tc.get("risk_tags", []):
                tags[tag] = tags.get(tag, 0) + 1

        return {
            "total_test_cases": len(self.test_cases),
            "capabilities": capabilities,
            "severities": severities,
            "tags": tags,
        }

    def merge(self, other: "DatasetBuilder") -> "DatasetBuilder":
        merged = DatasetBuilder(f"{self.name}_merged", f"Merged dataset from {self.name} and {other.name}")
        merged.add_test_cases(self.test_cases)
        merged.add_test_cases(other.test_cases)
        return merged

    def split(self, train_ratio: float = 0.8) -> tuple:
        import random
        shuffled = self.test_cases.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_cases = shuffled[:split_idx]
        test_cases = shuffled[split_idx:]

        train_dataset = DatasetBuilder(f"{self.name}_train", f"Training split of {self.name}")
        train_dataset.add_test_cases(train_cases)

        test_dataset = DatasetBuilder(f"{self.name}_test", f"Test split of {self.name}")
        test_dataset.add_test_cases(test_cases)

        return train_dataset, test_dataset

    def __len__(self) -> int:
        return len(self.test_cases)

    def __repr__(self) -> str:
        return f"DatasetBuilder(name={self.name}, test_cases={len(self.test_cases)})"
