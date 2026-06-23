from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from moondream_mini import MiniConfig, MiniMoondream
from moondream_mini.data import PPEVQADataset
from moondream_mini.prompts import build_prompt, normalize_text


def parse_args():
    p = argparse.ArgumentParser(description="Train Moondream-mini on PPE VQA data")
    p.add_argument("--train-jsonl", type=Path, default=Path("data/moondream_ppe_vqa/train.jsonl"))
    p.add_argument("--val-jsonl", type=Path, default=Path("data/moondream_ppe_vqa/val.jsonl"))
    p.add_argument("--image-root", type=Path, default=Path("data/moondream_ppe_vqa"))
    p.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/moondream-mini"))
    p.add_argument("--run-name", type=str, default="", help="Optional run name used in checkpoint filename.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accumulation-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-text-len", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--use-amp", action="store_true", default=True, help="Use automatic mixed precision on CUDA.")
    p.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    p.add_argument("--print-every", type=int, default=20, help="Print training progress every N steps.")
    p.add_argument("--freeze-vision", action="store_true", default=False)
    p.add_argument("--unfreeze-last-vision-layer", action="store_true")
    p.add_argument(
        "--freeze-epochs",
        type=int,
        default=3,
        help="How many initial epochs to keep vision frozen before unfreezing it.",
    )
    p.add_argument("--no-freeze-vision", dest="freeze_vision", action="store_false")
    return p.parse_args()


def collate(batch):
    return batch


def build_answer(answer: str) -> str:
    return normalize_text(answer)


def build_batch(batch, tokenizer, device, image_size=160):
    images = []
    input_ids = []
    labels = []
    attention_masks = []
    for item in batch:
        pil = item["image"].convert("RGB").resize((image_size, image_size))
        image = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().div(255.0)
        prompt_ids = tokenizer.encode(build_prompt(item["question"], item.get("label_space"))).ids
        answer_ids = tokenizer.encode(build_answer(item["answer"])).ids
        if not answer_ids or answer_ids[-1] != tokenizer.token_to_id("</s>"):
            eos_id = tokenizer.token_to_id("</s>") or tokenizer.token_to_id("<eos>")
            if eos_id is not None:
                answer_ids = answer_ids + [eos_id]
        seq = prompt_ids + answer_ids
        ids = torch.tensor(seq[:-1], dtype=torch.long)
        tgt = torch.tensor(seq[1:], dtype=torch.long)
        prompt_target_end = max(len(prompt_ids) - 1, 0)
        if prompt_target_end > 0:
            tgt[:prompt_target_end] = -100
        images.append(image)
        input_ids.append(ids)
        labels.append(tgt)
        attention_masks.append(torch.ones_like(ids))

    images = torch.stack(images, dim=0).to(device, non_blocking=True)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device, non_blocking=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device, non_blocking=True)
    attention_mask = pad_sequence(attention_masks, batch_first=True, padding_value=0).to(device, non_blocking=True)
    return images, input_ids, labels, attention_mask


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    correct = ((preds == labels) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)


def run_eval(model, loader, tokenizer, device, image_size, use_amp):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            images, input_ids, labels, attention_mask = build_batch(batch, tokenizer, device, image_size=image_size)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(images, input_ids, attention_mask=attention_mask)
                text_logits = logits[:, model.cfg.num_image_tokens :, :]
                loss = F.cross_entropy(
                    text_logits.reshape(-1, text_logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            total_loss += float(loss.detach().cpu())
            total_acc += accuracy_from_logits(text_logits, labels)
            steps += 1
    return total_loss / max(steps, 1), total_acc / max(steps, 1)


def linear_warmup_cosine(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return base_lr * cosine


def set_vision_trainable(model, trainable: bool, unfreeze_last_layer: bool = False) -> None:
    if not trainable:
        for p in model.vision.parameters():
            p.requires_grad = False
        return

    if unfreeze_last_layer and len(model.vision.blocks) > 0:
        for p in model.vision.parameters():
            p.requires_grad = False
        for p in model.vision.blocks[-1].parameters():
            p.requires_grad = True
        for module in (model.vision.ln, model.vision.proj):
            for p in module.parameters():
                p.requires_grad = True
        return

    for p in model.vision.parameters():
        p.requires_grad = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if args.grad_accumulation_steps < 1:
        raise ValueError("--grad-accumulation-steps must be at least 1")
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))

    print("[init] loading dataset and model...", end="\r", flush=True)
    train_ds = PPEVQADataset(args.train_jsonl, image_root=args.image_root)
    val_ds = PPEVQADataset(args.val_jsonl, image_root=args.image_root)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    model = MiniMoondream(
        MiniConfig(
            vocab_size=tokenizer.get_vocab_size(),
            image_size=args.image_size,
            num_image_tokens=(args.image_size // 16) ** 2 + 1,
            max_text_len=args.max_text_len,
        )
    ).to(device)

    if args.freeze_vision:
        set_vision_trainable(model, False)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    optimizer_steps_per_epoch = max(math.ceil(len(train_loader) / args.grad_accumulation_steps), 1)
    total_steps = args.epochs * optimizer_steps_per_epoch
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    best_val_loss = float("inf")
    best_path = None
    vision_unfrozen = not args.freeze_vision

    print(
        f"[init] device={device} train_size={len(train_ds)} val_size={len(val_ds)} "
        f"batches={len(train_loader)} batch_size={args.batch_size} accum={args.grad_accumulation_steps}"
    )
    start_time = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        if args.freeze_vision and not vision_unfrozen and epoch >= args.freeze_epochs:
            set_vision_trainable(model, True, unfreeze_last_layer=args.unfreeze_last_vision_layer)
            vision_unfrozen = True
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")
            print(f"[stage] unfreezing vision at epoch {epoch+1}")

        model.train()
        total = 0.0
        opt.zero_grad(set_to_none=True)
        print(f"[epoch {epoch+1}/{args.epochs}] start")
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch_start = time.time()
            images, input_ids, labels, attention_mask = build_batch(
                batch, tokenizer, device, image_size=args.image_size
            )
            with torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda"):
                logits = model(images, input_ids, attention_mask=attention_mask)
                text_logits = logits[:, model.cfg.num_image_tokens :, :]
                loss = F.cross_entropy(
                    text_logits.reshape(-1, text_logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss / args.grad_accumulation_steps
            scaler.scale(loss).backward()
            total += float(loss.detach().cpu()) * args.grad_accumulation_steps

            optimizer_stepped = batch_idx % args.grad_accumulation_steps == 0 or batch_idx == len(train_loader)
            if optimizer_stepped:
                if args.max_grad_norm is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                lr = linear_warmup_cosine(global_step, warmup_steps, total_steps, args.lr)
                for group in opt.param_groups:
                    group["lr"] = lr
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            elapsed = time.time() - batch_start
            if optimizer_stepped and args.print_every and global_step > 0 and global_step % args.print_every == 0:
                msg = (
                    f"[train] epoch={epoch+1}/{args.epochs} step={batch_idx}/{len(train_loader)} "
                    f"global_step={global_step} loss={float(loss.detach().cpu()) * args.grad_accumulation_steps:.4f} "
                    f"lr={opt.param_groups[0]['lr']:.2e} batch_time={elapsed:.2f}s"
                )
                print(f"\r{msg}", end="", flush=True)

        avg = total / max(len(train_loader), 1)
        val_loss, val_acc = run_eval(model, val_loader, tokenizer, device, args.image_size, args.use_amp)
        epoch_time = time.time() - start_time
        print(
            f"[epoch {epoch+1}/{args.epochs}] done avg_train_loss={avg:.4f} "
            f"val_loss={val_loss:.4f} val_token_acc={val_acc:.4f} elapsed={epoch_time:.1f}s"
        )

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = args.run_name.strip() or stamp
        ckpt_path = args.output_dir / f"moondream_mini_{run_name}_epoch{epoch+1}_{stamp}.pt"
        payload = {
            "model": model.state_dict(),
            "config": dict(model.cfg.__dict__),
            "training_args": vars(args),
            "seed": args.seed,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(payload, ckpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_path
            torch.save(payload, args.output_dir / f"moondream_mini_{run_name}_best.pt")
            print(f"[save] best checkpoint updated -> {args.output_dir / f'moondream_mini_{run_name}_best.pt'}")

    if best_path is not None:
        print(f"[done] best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
