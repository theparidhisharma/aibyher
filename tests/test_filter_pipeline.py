# Project scaffold: aibyher

# tests/test_filter_pipeline.py
import pytest
from filter_pipeline import filter_counterfactual_group

def test_pairwise_filtering_drops_divergent():
    group = [
        {"response": "Yes, I can do that.", "prompt": "Prompt 1", "sensitive_attributes": {}, "counterfactual_group": "g1"},
        {"response": "I'm sorry, I cannot help.", "prompt": "Prompt 2", "sensitive_attributes": {}, "counterfactual_group": "g1"}
    ]
    filtered = filter_counterfactual_group(group)
    assert len(filtered) == 1

def test_heuristic_blocks_swearing():
    group = [
        {"response": "Sure.", "prompt": "You are a f*cking idiot.", "sensitive_attributes": {}, "counterfactual_group": "g2"},
        {"response": "I will assist you.", "prompt": "Clean prompt.", "sensitive_attributes": {}, "counterfactual_group": "g2"}
    ]
    filtered = filter_counterfactual_group(group)
    assert len(filtered) == 1

# tests/test_metrics.py
import pytest
from evaluation.bias_metrics import compute_bias_metrics

MOCK_INPUT = "data/test/mock_filtered.jsonl"
MOCK_OUTPUT = "data/test/mock_bias_metrics.json"

def test_metrics_output(monkeypatch):
    import evaluation.bias_metrics as bm
    monkeypatch.setattr(bm, "INPUT_FILE", MOCK_INPUT)
    monkeypatch.setattr(bm, "OUTPUT_FILE", MOCK_OUTPUT)
    compute_bias_metrics()
    import json
    with open(MOCK_OUTPUT) as f:
        data = json.load(f)
        assert "intersectional_groups" in data
        assert isinstance(data["intersectional_groups"], dict)

# pytest.ini
[pytest]
minversion = 6.0
addopts = -ra -q
python_files = tests/test_*.py
