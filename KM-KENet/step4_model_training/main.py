"""Step 4 — KM-KENet Model Training.

Fine-tunes a multilingual encoder (XLM-R) on the weakly labelled dataset for
joint Named Entity Recognition and Relation Extraction, keeps the best
checkpoint on the dev split and reports test metrics.

    python -m step4_model_training.main --config configs/config.yaml
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.cli import build_arg_parser, init_step
from common.io_utils import read_jsonl, require_file, write_json
from common.utils import resolve_device
from step4_model_training.data import KMKENetCollator, KMKENetDataset, LabelSpace
from step4_model_training.model import KMKENet
from step4_model_training.trainer import Trainer, TrainingConfig

STEP_NAME = "step4_model_training"


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(itertools.islice(read_jsonl(path), limit))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Step 4 - train KM-KENet (joint NER + RE on XLM-R)")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)
    model_cfg = cfg.section("model")
    train_cfg = TrainingConfig.from_dict(cfg.section("training"))

    weak_dir = cfg.path("weak_dir")
    model_dir = cfg.path("model_dir")
    device = resolve_device(cfg.device)
    logger.info("Device: %s", device)

    train_records = load_records(
        require_file(weak_dir / "train.jsonl", "Run step 3 first."), train_cfg.max_train_examples
    )
    dev_records = load_records(weak_dir / "dev.jsonl", train_cfg.max_eval_examples)
    test_records = load_records(weak_dir / "test.jsonl", train_cfg.max_eval_examples)
    if not train_records:
        logger.error("The training split is empty")
        return 1

    encoder_name = str(model_cfg.get("encoder", "xlm-roberta-base"))
    max_length = int(model_cfg.get("max_length", 160))
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    label_space = LabelSpace.default()

    def make_dataset(records: list[dict[str, Any]], training: bool) -> KMKENetDataset:
        return KMKENetDataset(
            records,
            tokenizer,
            label_space,
            max_length,
            use_confidence_weights=train_cfg.use_confidence_weights,
            max_negative_relations=train_cfg.max_negative_relations if training else None,
            no_relation_weight=train_cfg.no_relation_weight if training else 1.0,
            seed=cfg.seed,
        )

    train_set = make_dataset(train_records, training=True)
    dev_set = make_dataset(dev_records, training=False) if dev_records else None
    test_set = make_dataset(test_records, training=False) if test_records else None
    logger.info(
        "Examples: train=%d dev=%d test=%d",
        len(train_set), len(dev_set) if dev_set else 0, len(test_set) if test_set else 0,
    )
    if len(train_set) == 0:
        logger.error("No training example survived encoding")
        return 1

    collator = KMKENetCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(cfg.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collator,
        num_workers=train_cfg.num_workers, generator=generator, pin_memory=pin_memory,
    )

    def eval_loader(dataset: KMKENetDataset | None) -> DataLoader | None:
        if dataset is None or len(dataset) == 0:
            return None
        return DataLoader(
            dataset, batch_size=train_cfg.eval_batch_size, shuffle=False, collate_fn=collator,
            num_workers=train_cfg.num_workers, pin_memory=pin_memory,
        )

    model = KMKENet(
        encoder_name,
        label_space,
        dropout=float(model_cfg.get("dropout", 0.1)),
        relation_hidden_size=int(model_cfg.get("relation_hidden_size", 512)),
        type_embedding_dim=int(model_cfg.get("type_embedding_dim", 32)),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(model_dir / "training_config.json", {
        "encoder": encoder_name,
        "max_length": max_length,
        "training": train_cfg.__dict__,
    })

    trainer = Trainer(model, tokenizer, train_loader, eval_loader(dev_set), train_cfg, device, model_dir)
    summary = trainer.train()
    logger.info("Best epoch: %d (score %.4f)", summary["best_epoch"], summary["best_score"])

    test_loader = eval_loader(test_set)
    if test_loader is not None:
        best = KMKENet.from_pretrained(model_dir)
        trainer.model = best.to(device)
        test_metrics = trainer.evaluate(test_loader)
        write_json(model_dir / "test_metrics.json", test_metrics)
        logger.info(
            "Test | NER P=%.4f R=%.4f F1=%.4f | RE P=%.4f R=%.4f F1=%.4f",
            test_metrics["ner"]["precision"], test_metrics["ner"]["recall"], test_metrics["ner"]["f1"],
            test_metrics["relations"]["precision"], test_metrics["relations"]["recall"],
            test_metrics["relations"]["f1"],
        )
    logger.info("Model saved in %s", model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
