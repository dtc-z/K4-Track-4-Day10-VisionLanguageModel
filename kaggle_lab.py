"""Runnable Kaggle implementation for Lab Day 10 (E0--E4).

The reference utilities under scripts/ are intentionally left untouched. This
module contains the notebook-side training and inference implementation.
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from scripts.labcommon import read_jsonl, write_jsonl


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
CLASSES = ("pothole", "crack", "manhole")
PROMPT = (
    "Detect road damage in this image. Detect only these classes: pothole, crack, manhole. "
    "Return only a valid JSON list. Each item must have keys label and bbox_2d. "
    "Coordinates must be integers on a normalized 0-1000 scale, in [x1,y1,x2,y2] order. "
    'Example: [{"label":"pothole","bbox_2d":[100,200,300,400]}]. Return [] if no objects are visible.'
)
PIXEL_PROMPT = (
    "Detect road damage in this image. Detect only these classes: pothole, crack, manhole. "
    "Return only a valid JSON list. Each item must have keys label and bbox_2d. "
    "Coordinates must be integer pixel values in [x1,y1,x2,y2] order. "
    'Example: [{"label":"pothole","bbox_2d":[64,72,192,144]}]. Return [] if no objects are visible.'
)

_LLM_TARGET = re.compile(
    r".*language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)
_VIT_TARGET = re.compile(
    r".*visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
)
_PROJECTOR_PARTS = ("visual.merger", "visual.deepstack_merger_list")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def yolo_to_box(label: str, yolo: list[float], coord: str = "norm1000") -> dict[str, Any]:
    """Convert a normalized YOLO xywh box to a Qwen grounding xyxy box."""
    xc, yc, width, height = (float(v) for v in yolo)
    scale_x, scale_y = (1000.0, 1000.0) if coord == "norm1000" else (640.0, 360.0)
    x1 = round(max(0.0, min(scale_x, (xc - width / 2) * scale_x)))
    y1 = round(max(0.0, min(scale_y, (yc - height / 2) * scale_y)))
    x2 = round(max(0.0, min(scale_x, (xc + width / 2) * scale_x)))
    y2 = round(max(0.0, min(scale_y, (yc + height / 2) * scale_y)))
    if x2 <= x1:
        x2 = min(int(scale_x), x1 + 1)
    if y2 <= y1:
        y2 = min(int(scale_y), y1 + 1)
    return {"label": label, "bbox_2d": [int(x1), int(y1), int(x2), int(y2)]}


def objects_to_answer(objects: list[dict[str, Any]], coord: str = "norm1000") -> str:
    boxes = [yolo_to_box(obj["label"], obj["yolo"], coord=coord) for obj in objects]
    boxes.sort(key=lambda box: (box["bbox_2d"][1], box["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)


def make_user_message(image: Image.Image, prompt: str = PROMPT) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]


def encode_example(
    image: Image.Image,
    answer: str,
    processor: Any,
    prompt: str = PROMPT,
) -> dict[str, torch.Tensor]:
    """Build a multimodal causal-LM example and mask user/image loss."""
    user_message = make_user_message(image, prompt)
    prompt_encoding = processor.apply_chat_template(
        user_message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_encoding = processor.apply_chat_template(
        user_message + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_encoding["input_ids"][0]
    input_ids = full_encoding["input_ids"][0]
    if input_ids.numel() < prompt_ids.numel() or not torch.equal(
        input_ids[: prompt_ids.numel()], prompt_ids
    ):
        raise ValueError("Qwen chat template changed the prompt prefix; cannot apply assistant-only loss mask")

    labels = input_ids.clone()
    labels[: prompt_ids.numel()] = -100
    row: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": full_encoding["attention_mask"][0],
        "labels": labels,
    }
    for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        if key in full_encoding:
            value = full_encoding[key]
            row[key] = value[0] if key == "mm_token_type_ids" and value.ndim == 2 else value
    return row


class RoadDamageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], image_root: str | Path):
        self.rows = rows
        self.image_root = Path(image_root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.image_root / row["image_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return {"row": row, "image": image}


def collate_examples(samples: list[dict[str, Any]], processor: Any, coord: str = "norm1000") -> dict[str, torch.Tensor]:
    encoded = [
        encode_example(
            sample["image"],
            objects_to_answer(sample["row"]["objects"], coord=coord),
            processor,
            prompt=PROMPT if coord == "norm1000" else PIXEL_PROMPT,
        )
        for sample in samples
    ]
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    result: dict[str, torch.Tensor] = {}
    result["input_ids"] = pad_sequence(
        [x["input_ids"] for x in encoded], batch_first=True, padding_value=pad_id
    )
    result["attention_mask"] = pad_sequence(
        [x["attention_mask"] for x in encoded], batch_first=True, padding_value=0
    )
    result["labels"] = pad_sequence(
        [x["labels"] for x in encoded], batch_first=True, padding_value=-100
    )
    if "mm_token_type_ids" in encoded[0]:
        result["mm_token_type_ids"] = pad_sequence(
            [x["mm_token_type_ids"] for x in encoded], batch_first=True, padding_value=0
        )
    for key in ("pixel_values", "image_grid_thw"):
        if key in encoded[0]:
            result[key] = torch.cat([x[key] for x in encoded], dim=0)
    return result


def _load_base_model() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU chưa được bật. Mở Notebook Settings → Accelerator → GPU rồi khởi động lại.")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # Explicit skip list prevents quantizing the vision tower/projector and lm_head.
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    return Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )


def load_processor() -> Any:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def _target_module_names(model: Any, setup: str) -> list[str]:
    names = []
    for name, _module in model.named_modules():
        if _LLM_TARGET.fullmatch(name) or (setup == "E3" and _VIT_TARGET.fullmatch(name)):
            names.append(name)
    expected = 348 if setup == "E3" else 252
    if len(names) != expected:
        examples = names[:5]
        raise RuntimeError(
            f"LoRA target mismatch for {setup}: found {len(names)} modules, expected {expected}. "
            f"First matches: {examples}. Inspect model.named_modules() before training."
        )
    return names


def _projector_parameters(model: Any) -> dict[str, torch.nn.Parameter]:
    return {
        name.removeprefix("base_model.model."): param
        for name, param in model.named_parameters()
        if any(part in name for part in _PROJECTOR_PARTS)
    }


def _save_run(model: Any, output_dir: Path, setup: str, step: int | None = None) -> None:
    suffix = f"step_{step:04d}" if step is not None else "final"
    dest = output_dir / "checkpoints" / setup / suffix
    adapter_path = dest / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path, safe_serialization=True)
    extras = _projector_parameters(model)
    if extras:
        torch.save({name: param.detach().cpu().clone() for name, param in extras.items()}, dest / "projector.pt")


def train_sft(
    setup: str,
    train_rows: list[dict[str, Any]],
    image_root: str | Path,
    output_dir: str | Path = "work",
    *,
    checkpoint_name: str | None = None,
    rank: int = 16,
    alpha: int | None = None,
    max_steps: int = 300,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    seed: int = 42,
    checkpoint_every: int = 100,
) -> dict[str, Any]:
    """Train one LoRA/projector setup and save adapter/projector checkpoints."""
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError(f"unsupported SFT setup: {setup}")
    run_name = checkpoint_name or setup
    seed_everything(seed)
    output_dir = Path(output_dir)
    processor = load_processor()
    model = _load_base_model()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    targets = _target_module_names(model, setup)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank * 2,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if setup in {"E2", "E3"}:
        extras = _projector_parameters(model)
        if not extras:
            raise RuntimeError("Could not find visual.merger / visual.deepstack_merger_list parameters")
        for param in extras.values():
            param.data = param.data.float()
            param.requires_grad = True

    # GradScaler.unscale_ rejects FP16 gradients on optimizer parameters.
    # Keep all trainable/master weights in FP32 while autocast handles the
    # forward pass in FP16. This also covers LoRA/projector dtypes that vary
    # between PEFT and Transformers versions.
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()

    trainable = [param for param in model.parameters() if param.requires_grad]
    trainable_params = sum(param.numel() for param in trainable)
    trainable_dtypes = {str(dtype): sum(p.numel() for p in trainable if p.dtype == dtype)
                        for dtype in {p.dtype for p in trainable}}
    model.print_trainable_parameters()
    print(f"{setup}: LoRA target modules={len(targets)}, trainable params={trainable_params:,}")
    print(f"{setup}: trainable parameter dtypes={trainable_dtypes}")

    dataset = RoadDamageDataset(train_rows, image_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: collate_examples(batch, processor),
    )
    if not len(loader):
        raise ValueError("Training split has no batches")

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    warmup_steps = max(1, math.ceil(max_steps * 0.05))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    step = 0
    micro_step = 0
    loss_window = 0.0

    while step < max_steps:
        for batch in loader:
            batch = {key: value.to("cuda:0", non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{setup} loss became non-finite at optimizer step {step + 1}")
            loss_window += float(loss.detach())
            scaler.scale(loss / grad_accum).backward()
            micro_step += 1
            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % 10 == 0:
                    print(f"{setup} step {step:03d}/{max_steps} loss={loss_window / grad_accum:.4f}")
                    loss_window = 0.0
                if checkpoint_every and step % checkpoint_every == 0:
                    _save_run(model, output_dir, run_name, step=step)
                if step >= max_steps:
                    break

    elapsed_minutes = (time.perf_counter() - start) / 60.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    _save_run(model, output_dir, run_name, step=None)
    result = {
        "setup": setup,
        "checkpoint_name": run_name,
        "rank": rank,
        "alpha": alpha if alpha is not None else rank * 2,
        "trainable_params": int(trainable_params),
        "peak_vram_gb": round(float(peak_vram_gb), 4),
        "train_minutes": round(float(elapsed_minutes), 4),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    run_path = output_dir / "runs" / f"{run_name}_rank{rank}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    del model, processor, optimizer, scheduler, scaler, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _load_projector(model: Any, checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing trained projector weights: {checkpoint}")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older torch releases
        state = torch.load(checkpoint, map_location="cpu")
    current = _projector_parameters(model)
    missing = sorted(set(state) - set(current))
    if missing:
        raise KeyError(f"projector checkpoint keys are not in the loaded model: {missing[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))


def load_trained_model(
    setup: str | None = None,
    output_dir: str | Path = "work",
    *,
    rank: int = 16,
    checkpoint_name: str | None = None,
) -> tuple[Any, Any]:
    processor = load_processor()
    processor.tokenizer.padding_side = "left"
    base = _load_base_model()
    if setup is not None:
        run_name = checkpoint_name or setup
        adapter_path = Path(output_dir) / "checkpoints" / run_name / "final" / "adapter"
        base = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        if setup in {"E2", "E3"}:
            _load_projector(base, Path(output_dir) / "checkpoints" / run_name / "final" / "projector.pt")
    base.eval()
    return base, processor


def predict_split(
    rows: list[dict[str, Any]],
    image_root: str | Path,
    output_jsonl: str | Path,
    *,
    setup: str | None = None,
    output_dir: str | Path = "work",
    rank: int = 16,
    checkpoint_name: str | None = None,
    max_new_tokens: int = 256,
    coord: str = "norm1000",
) -> Path:
    model, processor = load_trained_model(
        setup, output_dir=output_dir, rank=rank, checkpoint_name=checkpoint_name
    )
    prompt = PROMPT if coord == "norm1000" else PIXEL_PROMPT
    predictions = []
    for index, row in enumerate(rows, start=1):
        with Image.open(Path(image_root) / row["image_path"]) as source:
            image = source.convert("RGB")
        inputs = processor.apply_chat_template(
            make_user_message(image, prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to("cuda:0") for key, value in inputs.items() if key != "token_type_ids"}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completion = processor.batch_decode(
            generated[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        predictions.append({"image_id": row["image_id"], "raw": completion})
        if index % 25 == 0 or index == len(rows):
            print(f"{setup or 'E0'} inference: {index}/{len(rows)}")
    write_jsonl(output_jsonl, predictions)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return Path(output_jsonl)


def render_ground_truth(row: dict[str, Any], image_root: str | Path) -> Image.Image:
    """Return a copy of an image with its ground-truth boxes and labels drawn."""
    with Image.open(Path(image_root) / row["image_path"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for obj in row["objects"]:
        xc, yc, width, height = obj["yolo"]
        box = (
            (xc - width / 2) * row["width"],
            (yc - height / 2) * row["height"],
            (xc + width / 2) * row["width"],
            (yc + height / 2) * row["height"],
        )
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0] + 2, max(0, box[1] - 12)), obj["label"], fill="red")
    return image


def render_comparisons(
    rows: list[dict[str, Any]],
    prediction_jsonl: str | Path,
    image_root: str | Path,
    output_dir: str | Path = "report/figures",
    *,
    limit: int = 6,
    coord: str = "norm1000",
) -> list[Path]:
    """Save side-by-side GT/prediction overlays for report evidence."""
    from scripts.score import parse_output

    predictions = {row["image_id"]: row["raw"] for row in read_jsonl(prediction_jsonl)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for row in rows[:limit]:
        with Image.open(Path(image_root) / row["image_path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in row["objects"]:
            xc, yc, width, height = obj["yolo"]
            box = (
                (xc - width / 2) * row["width"],
                (yc - height / 2) * row["height"],
                (xc + width / 2) * row["width"],
                (yc + height / 2) * row["height"],
            )
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0] + 2, max(0, box[1] - 12)), f"GT {obj['label']}", fill="red")
        parsed = parse_output(predictions.get(row["image_id"]))
        if parsed:
            for obj in parsed:
                x1, y1, x2, y2 = obj["bbox"]
                if coord == "norm1000":
                    x1, x2 = x1 * row["width"] / 1000, x2 * row["width"] / 1000
                    y1, y2 = y1 * row["height"] / 1000, y2 * row["height"] / 1000
                draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
                draw.text((x1 + 2, min(row["height"] - 12, y2 + 1)), f"PRED {obj['label']}", fill="lime")
        path = output_dir / f"{row['image_id']}.jpg"
        image.save(path, quality=92)
        saved.append(path)
    return saved


def validate_split_sessions(split_rows: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in split_rows.items():
        for row in rows:
            previous = seen.setdefault(row["session"], split)
            if previous != split:
                raise ValueError(f"session leakage: {row['session']} occurs in {previous} and {split}")


def compute_iou_reward(completion: str, gt_boxes: list[dict[str, Any]]) -> float:
    """Optional E5 IoU reward; uses the lab scorer's parser and IoU implementation."""
    from scripts.score import iou_xyxy, parse_output

    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total_iou = 0.0
    labels = {item["label"] for item in gt_boxes} | {item["label"] for item in pred}
    for label in labels:
        gt = [item["bbox"] for item in gt_boxes if item["label"] == label]
        guessed = [item["bbox"] for item in pred if item["label"] == label]
        pairs = sorted(
            ((iou_xyxy(a, b), i, j) for i, a in enumerate(gt) for j, b in enumerate(guessed)),
            reverse=True,
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for value, i, j in pairs:
            if value < 0.5:
                break
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            total_iou += value
    denominator = max(len(gt_boxes), len(pred), 1)
    return 0.2 + 0.8 * total_iou / denominator


def grpo_iou_reward(completions: list[Any], gt_boxes: list[list[dict[str, Any]]], **_: Any) -> list[float]:
    """TRL reward callback for conversational completions and per-sample GT boxes."""
    rewards = []
    for completion, ground_truth in zip(completions, gt_boxes):
        if isinstance(completion, list) and completion:
            content = completion[-1].get("content", "")
        elif isinstance(completion, dict):
            content = completion.get("content", "")
        else:
            content = str(completion)
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        rewards.append(compute_iou_reward(content, ground_truth))
    return rewards


def _grpo_dataset_rows(rows: list[dict[str, Any]], image_root: str | Path):
    for row in rows:
        with Image.open(Path(image_root) / row["image_path"]) as source:
            image = source.convert("RGB")
        gt_boxes = []
        for obj in row["objects"]:
            box = yolo_to_box(obj["label"], obj["yolo"], coord="norm1000")
            gt_boxes.append({"label": obj["label"], "bbox": box["bbox_2d"]})
        yield {
            "prompt": PROMPT,
            "image": image,
            "gt_boxes_json": json.dumps(gt_boxes, separators=(",", ":")),
        }


def train_grpo_e5(
    train_rows: list[dict[str, Any]],
    image_root: str | Path,
    output_dir: str | Path = "work",
    *,
    max_steps: int = 60,
    learning_rate: float = 5e-6,
    seed: int = 42,
) -> dict[str, Any]:
    """Optional E5 GRPO fine-tune initialized from the SFT E1 adapter."""
    from datasets import Dataset, Features, Image as DatasetImage, Value
    from peft import prepare_model_for_kbit_training
    from trl import GRPOConfig, GRPOTrainer

    seed_everything(seed)
    output_dir = Path(output_dir)
    features = Features({
        "prompt": Value("string"),
        "image": DatasetImage(),
        "gt_boxes_json": Value("string"),
    })
    dataset = Dataset.from_generator(
        _grpo_dataset_rows,
        gen_kwargs={"rows": train_rows, "image_root": str(image_root)},
        features=features,
    )
    base = _load_base_model()
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    adapter_path = output_dir / "checkpoints" / "E1" / "final" / "adapter"
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    processor = load_processor()
    processor.tokenizer.padding_side = "left"
    model.config.use_cache = False
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    args = GRPOConfig(
        output_dir=str(output_dir / "grpo" / "E5"),
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_completion_length=128,
        max_steps=max_steps,
        beta=0.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, max_steps // 2),
        save_total_limit=2,
        report_to="none",
        use_vllm=False,
        seed=seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=grpo_iou_reward,
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.train()
    final_adapter = output_dir / "checkpoints" / "E5" / "final" / "adapter"
    trainer.save_model(str(final_adapter))
    elapsed_minutes = (time.perf_counter() - start) / 60.0
    result = {
        "setup": "E5",
        "initialized_from": "E1",
        "trainable_params": int(trainable_params),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 4),
        "train_minutes": round(float(elapsed_minutes), 4),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "num_generations": 4,
        "max_completion_length": 128,
        "seed": seed,
        "log_history": trainer.state.log_history,
    }
    run_path = output_dir / "runs" / "E5_grpo.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    del trainer, model, base, processor, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result
