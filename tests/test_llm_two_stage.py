"""Two-stage prompting (phee.llm.two_stage)."""

import json

from phee.data import load_samples
from phee.evaluation import evaluate
from phee.llm import two_stage
from phee.llm.prompts import extract_json, gold_json, to_predictions

DEV = "data/phee2/processed/dev.jsonl"


def _two_stage_gold(sample):
    predictions, events, valid = two_stage.parse_stage1(two_stage.stage1_gold_json(sample))
    assert valid and len(events) == len(sample["events"])
    for (event_index, summary), event in zip(events, sample["events"]):
        assert summary == two_stage.gold_event_summary(event)
        subs, ok = two_stage.parse_stage2(two_stage.stage2_gold_json(event), event_index, summary["event_type"])
        assert ok
        predictions += subs
    return predictions


def test_gold_round_trip_equals_the_one_step_prompt():
    samples = load_samples(DEV)
    two = evaluate(samples, [{"id": s["id"], "pred": _two_stage_gold(s)} for s in samples])
    one = evaluate(samples, [{"id": s["id"], "pred": to_predictions(extract_json(gold_json(s)))[0]} for s in samples])
    assert two["summary"] == one["summary"]


def test_prompts_split_the_fields():
    assert '"subject": null' in two_stage.STAGE1_PROMPT
    assert "treatment_drug" not in two_stage.STAGE1_PROMPT
    assert "treatment_drug (detail of the treatment)" in two_stage.STAGE2_PROMPT
    assert '"event_type"' not in two_stage.STAGE2_PROMPT.split("in this form:")[1]


def test_stage2_messages_carry_the_event_and_demonstrations():
    samples = load_samples(DEV)
    sample, demo = samples[0], samples[1]
    summary = two_stage.gold_event_summary(sample["events"][0])
    pairs = [(demo, two_stage.demonstration_event(demo, None))]
    conversation = two_stage.stage2_messages(sample, summary, pairs)
    assert [m["role"] for m in conversation] == ["system", "user", "assistant", "user"]
    assert json.dumps(summary, ensure_ascii=False) in conversation[-1]["content"]
    assert set(json.loads(conversation[2]["content"])) == set(two_stage.SUB_KEYS)


def test_stage2_parser_accepts_the_stage1_form_and_ignores_junk():
    text = 'Answer: {"events": [{"treatment_drug": ["insulin"], "treatment_dosage": "10 U", "subject_age": [""]}]}'
    predictions, valid = two_stage.parse_stage2(text, 1, "ADE")
    assert valid
    assert sorted((p["label"], p["text"], p["event"]) for p in predictions) == [
        ("TREATMENT_DOSAGE", "10 U", 1), ("TREATMENT_DRUG", "insulin", 1)]
    assert two_stage.parse_stage2("no json", 0, "ADE") == ([], False)


class _FakeTokenizer:
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True, **kwargs):
        return json.dumps(conversation, ensure_ascii=False)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_runner_wires_both_stages(tmp_path, monkeypatch):
    """A fake model answering every prompt with the gold answer: the run must reach the
    same scores as the gold round trip (stage-2 answers attached to the right events)."""
    import phee.llm.runner as runner
    from phee.io import experiment_config

    samples = {s["context"]: s for split in ("dev", "test") for s in load_samples(f"data/phee2/processed/{split}.jsonl")}

    class FakeBackend:
        def __init__(self, model_config, max_new_tokens):
            self.tokenizer = _FakeTokenizer()

        def generate(self, prompts):
            texts = []
            for prompt in prompts:
                last = json.loads(prompt)[-1]["content"]
                sentence = last.split("\n")[0][len("Sentence: "):]
                sample = samples[sentence]
                if "\nEvent: " in last:
                    summary = json.loads(last.split("\nEvent: ", 1)[1])
                    event = next(e for e in sample["events"] if two_stage.gold_event_summary(e) == summary)
                    texts.append(two_stage.stage2_gold_json(event))
                else:
                    texts.append(two_stage.stage1_gold_json(sample))
            return texts, [1] * len(texts)

    monkeypatch.setattr(runner, "VllmBackend", FakeBackend)
    monkeypatch.setattr(runner, "_model_weights_gb", lambda name: None)
    monkeypatch.setattr(runner, "_reserved_gpu_memory_gb", lambda config: None)
    config = experiment_config("configs/llm/qwen3_8b_two_stage.yaml", 1,
                               overrides=["data.limit=60", "shots=[0, 2]"])
    runner.run(config, 1, tmp_path)
    metrics = json.load(open(tmp_path / "metrics" / "shots=2__dev.json"))
    assert metrics["summary"]["overall_classification_f1"] > 0.99
    assert metrics["events"]["arguments"]["f1"] > 0.99
