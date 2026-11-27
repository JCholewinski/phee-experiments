"""Zero-shot / few-shot extraction with a local open-weight LLM (no fine-tuning on PHEE).

Config (configs/llm/*.yaml):
  method: llm
  model:
    name: Qwen/Qwen3-8B
    backend: vllm            # vllm (recommended) | hf (transformers.generate, slow; tests)
    max_model_len: 6144      # vllm
    gpu_memory_utilization: 0.85
    dtype: bfloat16
    quantization: null       # e.g. fp8 / awq for larger models
    chat_template_kwargs: {enable_thinking: false}   # passed to the chat template
  shots: [0, 5]              # every value is one variant (BM25 demonstrations from train)
  demonstrations: per_type   # per_type: `shots` most similar sentences with an ADE and
                             # `shots` with a PTE (Sun et al. 2024, BM25-Type; used by both
                             # prompts, so they see the same examples); top: `shots` most
                             # similar sentences of any type
  max_new_tokens: 1024

Greedy decoding (temperature 0), so a single run is enough: the seed only matters when
it chooses the pool of demonstrations (`data.train_sample.seed: run`).
"""

from phee.experiment import RunRecorder, load_splits
from phee.llm.bm25 import BM25
from phee.llm import authors_prompt, two_stage
from phee.llm.prompts import extract_json, messages, to_predictions
from phee.utils import GpuMemory, Timer, log


def _model_weights_gb(name):
    """Size of the model weights (safetensors files), i.e. the GPU memory taken by the
    weights in bf16. vLLM reserves `gpu_memory_utilization` of the GPU up front (the rest
    goes to its KV cache), so its real peak usage cannot be measured like for the other
    methods; the weight size is reported instead."""
    try:
        from pathlib import Path

        path = Path(name)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(name, allow_patterns=["*.safetensors", "*.json"]))
        size = sum(file.stat().st_size for file in path.glob("*.safetensors"))
        return size / 1024 ** 3 if size else None
    except Exception:  # noqa: BLE001
        return None


def _reserved_gpu_memory_gb(model_config):
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            return model_config.get("gpu_memory_utilization", 0.85) * total / 1024 ** 3
    except Exception:  # noqa: BLE001
        pass
    return None


def _prompts(tokenizer, samples, demonstrations, chat_template_kwargs, build_messages=messages):
    prompts = []
    for sample, demos in zip(samples, demonstrations):
        prompts.append(
            tokenizer.apply_chat_template(
                build_messages(sample, demos),
                tokenize=False,
                add_generation_prompt=True,
                **(chat_template_kwargs or {}),
            )
        )
    return prompts


class VllmBackend:
    def __init__(self, model_config, max_new_tokens):
        from vllm import LLM, SamplingParams

        kwargs = {
            "model": model_config["name"],
            "dtype": model_config.get("dtype", "bfloat16"),
            "max_model_len": model_config.get("max_model_len", 6144),
            "gpu_memory_utilization": model_config.get("gpu_memory_utilization", 0.85),
            "seed": 0,
        }
        if model_config.get("quantization"):
            kwargs["quantization"] = model_config["quantization"]
        if model_config.get("tensor_parallel_size"):
            kwargs["tensor_parallel_size"] = model_config["tensor_parallel_size"]
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def generate(self, prompts):
        # The chat template already contains the special tokens (e.g. Llama's
        # <|begin_of_text|>); a text prompt would get a second BOS from vLLM's tokenizer
        # call, so the prompts are tokenized here without special tokens.
        token_prompts = [
            {"prompt_token_ids": self.tokenizer(prompt, add_special_tokens=False)["input_ids"]}
            for prompt in prompts
        ]
        outputs = self.llm.generate(token_prompts, self.sampling)
        return [output.outputs[0].text for output in outputs], [len(output.outputs[0].token_ids) for output in outputs]


class HfBackend:
    def __init__(self, model_config, max_new_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.max_new_tokens = max_new_tokens
        self.batch_size = model_config.get("batch_size", 8)

    def generate(self, prompts):
        import torch

        texts, lengths = [], []
        for start in range(0, len(prompts), self.batch_size):
            batch = self.tokenizer(prompts[start:start + self.batch_size], return_tensors="pt",
                                   padding=True, add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = generated[:, batch["input_ids"].shape[1]:]
            for row in new_tokens:
                ids = [t for t in row.tolist() if t != self.tokenizer.pad_token_id]
                texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
                lengths.append(len(ids))
        return texts, lengths


def run(config, seed, run_dir):
    data = load_splits(config)
    model_config = config["model"]
    shots_list = config.get("shots", [0, 5])
    max_new_tokens = config.get("max_new_tokens", 1024)
    splits = config.get("splits", ["dev", "test"])
    recorder = RunRecorder(run_dir, config, seed)

    log(f"llm: loading {model_config['name']} ({model_config.get('backend', 'vllm')})")
    with Timer() as load_timer:
        backend_class = VllmBackend if model_config.get("backend", "vllm") == "vllm" else HfBackend
        backend = backend_class(model_config, max_new_tokens)
    weights_gb = _model_weights_gb(model_config["name"])
    reserved_gb = _reserved_gpu_memory_gb(model_config)
    # prompt_style: own (our prompt and answer format) or authors (Sun et al. 2024: their
    # instruction, answer format and parsing). The demonstrations are chosen the same way
    # for both prompts (`demonstrations`), so the two versions differ only in the prompt.
    prompt_style = config.get("prompt_style", "own")
    if prompt_style not in ("own", "authors", "two_stage"):
        raise ValueError(f"Unknown prompt_style: {prompt_style}")
    authors_style = prompt_style == "authors"
    two_stages = prompt_style == "two_stage"     # phee.llm.two_stage: events, then sub-arguments per event
    if two_stages:
        build_messages, parse = two_stage.stage1_messages, None
    elif authors_style:
        build_messages, parse = authors_prompt.messages, lambda text: authors_prompt.to_predictions(authors_prompt.extract_events(text))
    else:
        build_messages, parse = messages, lambda text: to_predictions(extract_json(text))
    selection = config.get("demonstrations", "per_type")
    if selection == "per_type":
        retriever = authors_prompt.TypeRetriever(data["train"])
    elif selection == "top":
        retriever = BM25([sample["context"] for sample in data["train"]])
    else:
        raise ValueError(f"Unknown demonstrations: {selection}")
    log(f"llm: prompt {prompt_style}, demonstrations {selection}, "
        f"pool of {len(data['train'])} training sentences")

    for shots in shots_list:
        for split in splits:
            samples = data[split]
            log(f"llm: {shots}-shot on {split} ({len(samples)} sentences)")
            # timed from raw text: demonstration retrieval, prompt building and generation
            with GpuMemory() as memory, Timer() as timer:
                if shots and selection == "per_type":
                    # `shots` demonstrations per event type, most similar first (authors)
                    demonstrations = [retriever.demonstrations(sample, shots) for sample in samples]
                elif shots:
                    # most similar demonstration last (closest to the query)
                    demonstrations = [
                        [data["train"][i] for i in reversed(retriever.top(sample["context"], shots))]
                        for sample in samples
                    ]
                else:
                    demonstrations = [[] for _ in samples]
                prompts = _prompts(backend.tokenizer, samples, demonstrations,
                                   model_config.get("chat_template_kwargs"), build_messages)
                texts, lengths = backend.generate(prompts)
                if two_stages:
                    # stage 2 inside the timer: one call per event found in stage 1
                    stage1 = [two_stage.parse_stage1(text) for text in texts]
                    prompts2, owners = [], []
                    for index, (sample, demos, (_, events, _)) in enumerate(zip(samples, demonstrations, stage1)):
                        # per_type: the first `shots` demonstrations were retrieved for an ADE, the rest for a PTE
                        types = (["ADE"] * shots + ["PTE"] * len(demos))[:len(demos)] if selection == "per_type" else [None] * len(demos)
                        pairs = [(demo, two_stage.demonstration_event(demo, event_type)) for demo, event_type in zip(demos, types)]
                        for event_index, summary in events:
                            prompts2.append(backend.tokenizer.apply_chat_template(
                                two_stage.stage2_messages(sample, summary, pairs), tokenize=False,
                                add_generation_prompt=True, **(model_config.get("chat_template_kwargs") or {}),
                            ))
                            owners.append((index, event_index, summary["event_type"]))
                    texts2, lengths2 = backend.generate(prompts2) if prompts2 else ([], [])
            records = []
            invalid = 0
            if two_stages:
                extra, raw2, invalid2 = [[] for _ in samples], [[] for _ in samples], 0
                for (index, event_index, event_type), text in zip(owners, texts2):
                    predictions, valid = two_stage.parse_stage2(text, event_index, event_type)
                    invalid2 += 0 if valid else 1
                    extra[index].extend(predictions)
                    raw2[index].append(text)
                for index, (sample, text) in enumerate(zip(samples, texts)):
                    predictions, _, valid = stage1[index]
                    invalid += 0 if valid else 1
                    records.append({"id": sample["id"], "pred": predictions + extra[index], "raw": text,
                                    "raw_stage2": raw2[index], "valid_json": valid})
            else:
                for sample, text in zip(samples, texts):
                    predictions, valid = parse(text)
                    invalid += 0 if valid else 1
                    records.append({"id": sample["id"], "pred": predictions, "raw": text, "valid_json": valid})
            prompt_tokens = [len(backend.tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
            if two_stages:
                prompt_tokens2 = [len(backend.tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts2]
            inference = {
                "seconds": timer.seconds,
                # vLLM runs the model in its own process and pre-allocates
                # gpu_memory_utilization x total memory: no comparable peak (see _model_weights_gb)
                "peak_gpu_memory_gb": memory.peak_gb if isinstance(backend, HfBackend) else None,
                "model_weights_gb": weights_gb,
                "reserved_gpu_memory_gb": None if isinstance(backend, HfBackend) else reserved_gb,
                "invalid_json": invalid,
                "invalid_json_rate": invalid / max(1, len(samples)),
                "mean_prompt_tokens": sum(prompt_tokens) / max(1, len(prompt_tokens)),
                "mean_output_tokens": sum(lengths) / max(1, len(lengths)),
            }
            if two_stages:
                # per sentence, both stages together (comparable with the one-step prompt)
                inference.update({
                    "mean_prompt_tokens": (sum(prompt_tokens) + sum(prompt_tokens2)) / max(1, len(samples)),
                    "mean_output_tokens": (sum(lengths) + sum(lengths2)) / max(1, len(samples)),
                    "stage2_calls": len(prompts2),
                    "stage2_invalid_json": invalid2,
                    "mean_stage1_prompt_tokens": sum(prompt_tokens) / max(1, len(prompt_tokens)),
                    "mean_stage2_prompt_tokens": sum(prompt_tokens2) / max(1, len(prompt_tokens2)),
                })
            recorder.add_variant(f"shots={shots}", split, samples, records, inference)
    return recorder.finish({"model_load_seconds": load_timer.seconds})
