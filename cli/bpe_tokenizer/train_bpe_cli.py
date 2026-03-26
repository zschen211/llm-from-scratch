from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# 运行 `python cli/.../xxx_cli.py` 时，`sys.path[0]` 指向脚本目录，
# 需要显式把项目根目录加入 sys.path，才能导入真实包。
_project_root = Path(__file__).resolve().parents[2]
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
    parser.add_argument(
        "--stream-chunk-chars",
        type=int,
        default=1_000_000,
        help="Streaming mode: chars to read per I/O pass (default: 1000000). Set 0 to disable streaming.",
    )
    parser.add_argument(
        "--stream-memory-target-percent",
        type=float,
        default=85.0,
        help="Memory usage threshold (0-100) for chunk spilling during stream mode (default: 85).",
    )

    args = parser.parse_args(argv)

    # Progress state. Kept in a dict to avoid Python scoping/nonlocal complexity.
    _progress_state: dict[str, Any] = {
        "pretok_start_t": None,
        "pretok_bytes_read": 0,
        "pretok_file_size": 0,
        "pretok_count_done": 0,
        "merge_start_t": None,
        "merges_target": None,
        "merges_done": 0,
        "last_render_t": 0.0,
    }

    def _progress_bar(frac: float, width: int = 24) -> str:
        frac = max(0.0, min(1.0, frac))
        done = int(frac * width)
        return "[" + ("#" * done) + ("." * (width - done)) + "]"

    def _render_line(line: str, *, newline: bool = False) -> None:
        # Only render progress when metrics are enabled.
        if args.no_print_metrics:
            return
        if newline:
            print(line, file=sys.stderr, flush=True)
            return
        sys.stderr.write("\r" + line + " " * 8)
        sys.stderr.flush()

    label_width = 12  # 让进度条 '[' 在 pretok/merge 两行对齐

    def _maybe_print_metrics(metrics: dict[str, Any]) -> None:
        event = metrics.get("event")

        if event == "profile_saved":
            print(f"[profile] saved: {metrics.get('path')}", flush=True)
            return

        if args.no_print_metrics:
            return

        stage = metrics.get("stage")
        now_t = time.perf_counter()
        # Throttle only pretok progress (very frequent), keep merge events unthrottled
        # so the progress bar always refreshes.
        if event == "pretok_progress" and now_t - float(_progress_state["last_render_t"]) < 0.05:
            return

        # --------------- pretok progress ---------------
        if stage == "data_pretokenization":
            if event == "stage_start":
                _progress_state["pretok_start_t"] = time.perf_counter()
                _progress_state["pretok_total_chunks"] = None
                _progress_state["pretok_done_chunks"] = 0
                _progress_state["pretok_count_done"] = 0
                _render_line("Pretokenize...", newline=False)
                _progress_state["last_render_t"] = now_t
                return

            if event == "pretok_progress":
                bytes_read = int(metrics.get("bytes_read") or 0)
                file_size = int(metrics.get("file_size") or 0)
                pretok_count_done = int(metrics.get("pretok_count_done") or 0)
                _progress_state["pretok_bytes_read"] = bytes_read
                _progress_state["pretok_file_size"] = file_size
                _progress_state["pretok_count_done"] = pretok_count_done

                pretok_start_t = _progress_state["pretok_start_t"]
                elapsed_s = (time.perf_counter() - pretok_start_t) if isinstance(pretok_start_t, (int, float)) else 0.0
                tok_s = (pretok_count_done / elapsed_s) if elapsed_s > 0 else 0.0

                frac = (bytes_read / file_size) if file_size > 0 else 0.0
                bar = _progress_bar(frac)
                mb_read = bytes_read / (1024 * 1024)
                mb_total = file_size / (1024 * 1024)
                label = f"{'Pretokenize':<{label_width}}"
                line = (
                    f"{label}{bar} {mb_read:.1f}/{mb_total:.1f}MB "
                    f"tokens={pretok_count_done} tok/s={tok_s:.0f}"
                )
                _render_line(line, newline=False)
                _progress_state["last_render_t"] = now_t
                return

            if event == "stage_end":
                stage_metrics = metrics.get("metrics") or {}
                pretok_count = int(stage_metrics.get("pretok_count") or 0)
                pretok_ms = float(stage_metrics.get("pretok_ms") or 0.0)
                _render_line(
                    f"Pretokenize done: tokens={pretok_count} time={pretok_ms:.2f}ms", newline=True
                )
                _progress_state["last_render_t"] = now_t
                return

        # --------------- merge progress ---------------
        # 注意：train_bpe 的 merge_iter_end 事件里不包含 stage 字段；
        # 因此这里只根据 event 名称来更新进度条。
        if event == "stage_start" and stage == "byte_pair_merge_iter":
            _progress_state["merge_start_t"] = time.perf_counter()
            mt = metrics.get("merges_target")
            _progress_state["merges_target"] = int(mt) if mt is not None else None
            _progress_state["merges_done"] = 0

            target = _progress_state["merges_target"]
            target_display = str(target) if isinstance(target, int) else "?"
            bar = _progress_bar(0.0)
            label = f"{'Merge':<{label_width}}"
            _render_line(f"{label}{bar} 0/{target_display} merges", newline=False)
            _progress_state["last_render_t"] = now_t
            return

        if event == "merge_iter_end":
            _progress_state["merges_done"] = int(
                metrics.get("merges_done") or metrics.get("merge_iter") or 0
            )
            if (
                _progress_state["merges_target"] is None
                and metrics.get("merges_target") is not None
            ):
                _progress_state["merges_target"] = int(metrics.get("merges_target") or 0)

            target = _progress_state["merges_target"]
            done = int(_progress_state["merges_done"])
            target_val = int(target) if isinstance(target, int) else 0

            merge_start_t = _progress_state["merge_start_t"]
            elapsed_s = (
                (time.perf_counter() - merge_start_t)
                if isinstance(merge_start_t, (int, float))
                else 0.0
            )
            merges_per_s = (done / elapsed_s) if elapsed_s > 0 else 0.0

            frac = (done / target_val) if target_val > 0 else 0.0
            bar = _progress_bar(frac)

            avg_ms = metrics.get("avg_merge_wall_ms")
            avg_ms_s = float(avg_ms) if avg_ms is not None else 0.0
            target_display = str(target_val) if target_val > 0 else "?"
            label = f"{'Merge':<{label_width}}"
            line = (
                f"{label}{bar} {done}/{target_display} merges "
                f"avg_merge={avg_ms_s:.2f}ms ({merges_per_s:.2f} merges/s)"
            )
            _render_line(line, newline=False)
            _progress_state["last_render_t"] = now_t
            return

        if event == "stage_end" and stage == "byte_pair_merge_iter":
            stage_metrics = metrics.get("metrics") or {}
            total_merges_executed = int(stage_metrics.get("total_merges_executed") or 0)
            total_iter_wall_ms = float(stage_metrics.get("total_iter_wall_ms") or 0.0)
            _render_line(
                f"Merge done: merges={total_merges_executed} total_time={total_iter_wall_ms:.2f}ms",
                newline=True,
            )
            _progress_state["last_render_t"] = now_t
            return

        # Fallback: ignore other events to keep output clean when progress is enabled.

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
        stream_chunk_chars=args.stream_chunk_chars,
        stream_memory_target_percent=args.stream_memory_target_percent,
    )

    _save_checkpoint(Path(args.out), vocab=vocab, merges=merges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

