"""Common training procedure for every trained model (work plan, section 2).

  * AdamW, linear warm-up and decay over `max_epochs`;
  * evaluation on the dev set after every epoch with a method-specific F1;
  * early stopping: training ends when `patience` (default 2) consecutive evaluations
    do not improve the best dev F1; the best checkpoint is restored at the end;
  * bf16 autocast on GPU, gradient clipping, gradient accumulation;
  * the seed controls model initialisation and data order.
"""

import math
import time

from phee.utils import GpuMemory, autocast_context, get_device, log, set_seed

DEFAULT_TRAINING = {
    "learning_rate": 5e-5,
    "head_learning_rate": None,   # separate LR for parameters whose name starts with "head."
    "crf_learning_rate": None,    # separate LR for the CRF transition scores ("head.crf.")
    "batch_size": 16,
    "eval_batch_size": 64,
    "grad_accum": 1,
    "max_epochs": 20,
    "patience": 2,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "precision": "bf16",
    "log_every": 50,
}


def training_settings(config):
    settings = dict(DEFAULT_TRAINING)
    settings.update(config or {})
    return settings


def _parameter_groups(model, settings):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "layernorm.weight")
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = settings["learning_rate"]
        if name.startswith("head.") and settings["head_learning_rate"]:
            lr = settings["head_learning_rate"]
        if name.startswith("head.crf.") and settings["crf_learning_rate"]:
            lr = settings["crf_learning_rate"]
        decay = 0.0 if any(marker in name for marker in no_decay) or param.ndim == 1 else settings["weight_decay"]
        groups.setdefault((lr, decay), []).append(param)
    return [
        {"params": params, "lr": lr, "weight_decay": decay}
        for (lr, decay), params in groups.items()
    ]


def _move(batch, device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def train_model(model, train_dataset, collate_fn, evaluate_fn, settings, seed, name="model"):
    """Train `model` and return (model with the best weights, report).

    train_dataset  a sequence of examples
    collate_fn     list of examples -> dict of tensors; must contain what model(**batch) needs,
                   the model must return an object with `.loss`
    evaluate_fn    callable(model) -> dict with at least "f1" (dev score, higher is better)
    """
    import torch
    from torch.utils.data import DataLoader

    settings = training_settings(settings)
    set_seed(seed)
    device = get_device()
    model.to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )

    steps_per_epoch = math.ceil(len(loader) / settings["grad_accum"])
    total_steps = max(1, steps_per_epoch * settings["max_epochs"])
    warmup_steps = int(settings["warmup_ratio"] * total_steps)

    optimizer = torch.optim.AdamW(_parameter_groups(model, settings))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_score = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    train_seconds = 0.0
    eval_seconds = 0.0
    global_step = 0

    with GpuMemory() as memory:
        for epoch in range(1, settings["max_epochs"] + 1):
            model.train()
            epoch_start = time.perf_counter()
            running_loss = 0.0
            running_count = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader, start=1):
                batch = _move(batch, device)
                with autocast_context(settings["precision"]):
                    outputs = model(**batch)
                    loss = outputs.loss / settings["grad_accum"]
                loss.backward()
                running_loss += loss.item() * settings["grad_accum"]
                running_count += 1

                if step % settings["grad_accum"] == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if settings["log_every"] and global_step % settings["log_every"] == 0:
                        log(f"{name} epoch {epoch} step {global_step}: loss {running_loss / running_count:.4f}")

            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_train_seconds = time.perf_counter() - epoch_start
            train_seconds += epoch_train_seconds

            eval_start = time.perf_counter()
            model.eval()
            with torch.no_grad():
                scores = evaluate_fn(model)
            epoch_eval_seconds = time.perf_counter() - eval_start
            eval_seconds += epoch_eval_seconds

            score = float(scores["f1"])
            improved = score > best_score
            history.append({
                "epoch": epoch,
                "train_loss": running_loss / max(1, running_count),
                "dev": scores,
                "train_seconds": epoch_train_seconds,
                "eval_seconds": epoch_eval_seconds,
                "improved": improved,
            })
            log(
                f"{name} epoch {epoch}: train loss {running_loss / max(1, running_count):.4f}, "
                f"dev f1 {score:.4f}{' (best)' if improved else ''}"
            )

            if improved:
                best_score = score
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().to("cpu", copy=True)
                    for key, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= settings["patience"]:
                    log(f"{name}: early stopping after epoch {epoch} (best epoch {best_epoch})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    report = {
        "name": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_dev_f1": best_score,
        "train_seconds": train_seconds,
        "dev_eval_seconds": eval_seconds,
        "train_examples": len(train_dataset),
        "peak_gpu_memory_gb": memory.peak_gb,
        "settings": settings,
        "history": history,
    }
    return model, report
