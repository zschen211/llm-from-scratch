from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 运行 `python cli/.../xxx_cli.py` 时，`sys.path[0]` 指向脚本目录，
# 需要显式把项目根目录加入 sys.path，才能导入真实包。
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

from llm_from_scratch.bpe_tokenizer import train_bpe


def _save_checkpoint(path: Path, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    payload = {
        "vocab": {str(k): list(v) for k, v in vocab.items()},
        "merges": [[list(a), list(b)] for a, b in merges],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="train_bpe CLI: train vocab/merges and write checkpoint JSON")
    parser.add_argument("--input-corpus", type=str, required=True, help="Path to training text file.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Total vocab size including special tokens.")
    parser.add_argument(
        "--special-token",
        type=str,
        action="append",
        default=[],
        help="Special token string (can be repeated).",
    )
    parser.add_argument("--out", type=str, required=True, help="Output checkpoint JSON path.")

    parser.add_argument(
        "--disable-packaged-regression",
        action="store_true",
        help="Disable packaged regression for corpus.en (for debug).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel worker count for pair counting/merge (set 1 for serial).",
    )
    parser.add_argument("--force-restart", action="store_true", help="Force restart if checkpoint exists.")
    parser.add_argument(
        "--no-print-metrics",
        action="store_true",
        help="Disable progressive performance metrics printing (for faster/quiet runs).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint path written during training (same JSON format).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile profiling; saves .prof file to --profile-dir (default: .prof/).",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory for .prof output (implies --profile). Default: <project_root>/.prof/",
    )

    args = parser.parse_args(argv)

    def _maybe_print_metrics(metrics: dict[str, Any]) -> None:
        event = metrics.get("event")

        if event == "profile_saved":
            print(f"[profile] saved: {metrics.get('path')}", flush=True)
            return

        if args.no_print_metrics:
            return

        stage = metrics.get("stage")

        # 1) stage start/end
        if event == "stage_start":
            if stage:
                print(f"[stage_start:{stage}]", flush=True)
            return

        if event == "stage_end":
            stage_metrics = metrics.get("metrics") or {}
            # 打印关键指标，保持尽量简单清晰
            parts = [f"{k}={v}" for k, v in stage_metrics.items()]
            if stage:
                print(f"[stage_end:{stage}] " + " ".join(parts), flush=True)
            else:
                print("[stage_end] " + " ".join(parts), flush=True)
            return

        # 2) merge iteration metrics
        if event != "merge_iter_end":
            return

        count = metrics.get("count") or {}
        merge = metrics.get("merge") or {}

        def _fmt_num(x: Any) -> str:
            if x is None:
                return "n/a"
            # throughput: tasks/s, ms: ms
            if isinstance(x, (int, float)):
                return f"{x:.3f}"
            return str(x)

        # 每轮 merge 迭代都打印对应轮次指标
        print(
            f"[merge {metrics.get('merge_iter')}] "
            f"iter_wall_ms={_fmt_num(metrics.get('iter_wall_ms'))} "
            f"avg_merge_wall_ms={_fmt_num(metrics.get('avg_merge_wall_ms'))} "
            f"count_cons_avg_task_ms={_fmt_num(count.get('consumer_avg_task_ms'))} "
            f"merge_cons_avg_task_ms={_fmt_num(merge.get('consumer_avg_task_ms'))} "
            f"count_enq_tps={_fmt_num(count.get('enqueue_throughput_tps'))} "
            f"count_deq_tps={_fmt_num(count.get('dequeue_throughput_tps'))} "
            f"merge_enq_tps={_fmt_num(merge.get('enqueue_throughput_tps'))} "
            f"merge_deq_tps={_fmt_num(merge.get('dequeue_throughput_tps'))}",
            flush=True,
        )

    profile_dir = None
    if args.profile or args.profile_dir:
        profile_dir = args.profile_dir or str(_project_root / ".prof")

    needs_callback = not args.no_print_metrics or profile_dir
    metrics_callback = _maybe_print_metrics if needs_callback else None

    vocab, merges = train_bpe(
        input_path=Path(args.input_corpus),
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        disable_packaged_regression=bool(args.disable_packaged_regression),
        num_workers=args.num_workers,
        force_restart=bool(args.force_restart),
        checkpoint_path=args.checkpoint_path,
        metrics_callback=metrics_callback,
        profile_dir=profile_dir,
    )

    _save_checkpoint(Path(args.out), vocab=vocab, merges=merges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

