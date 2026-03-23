"""train_bpe cProfile 结果的火焰图可视化 CLI。

对应 public 方法: train_bpe (profile_dir kwarg 生成的 .prof 文件)。
基于 pstats（标准库）读取 .prof，生成交互式 d3-flame-graph HTML 火焰图。

用法:
    python train_bpe_flamegraph_cli.py .prof/train_bpe_xxx.prof
    python train_bpe_flamegraph_cli.py .prof/train_bpe_xxx.prof --out fg.html --open
"""

from __future__ import annotations

import argparse
import json
import pstats
import sys
import webbrowser
from pathlib import Path

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>train_bpe Flame Graph</title>
<link rel="stylesheet" type="text/css"
      href="https://cdn.jsdelivr.net/npm/d3-flame-graph@4/dist/d3-flamegraph.css">
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 16px; background: #fafafa;
  }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
  #chart {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 4px; padding: 8px; }}
</style>
</head>
<body>
<h1>train_bpe Flame Graph</h1>
<p class="meta">Source: {prof_name} &middot; Total cumulative time: {total_time_s:.3f}s</p>
<div id="chart"></div>
<script src="https://d3js.org/d3.v7.min.js" charset="utf-8"></script>
<script src="https://cdn.jsdelivr.net/npm/d3-flame-graph@4/dist/d3-flamegraph.min.js"></script>
<script>
var data = {data_json};
var chart = flamegraph()
  .width(Math.max(960, window.innerWidth - 48))
  .cellHeight(18)
  .transitionDuration(200)
  .tooltip(true);
d3.select("#chart").datum(data).call(chart);
window.addEventListener("resize", function() {{
  chart.width(Math.max(960, window.innerWidth - 48));
  d3.select("#chart").datum(data).call(chart);
}});
</script>
</body>
</html>
"""


def _func_label(key: tuple[str, int, str]) -> str:
    filename, lineno, funcname = key
    short = filename.rsplit("/", 1)[-1] if "/" in filename else filename
    if short.startswith("<") or short == "~":
        return funcname
    return f"{funcname} ({short}:{lineno})"


def _build_flamegraph_tree(prof_path: str) -> tuple[dict, float]:
    """Read a .prof file via pstats and return (d3_tree_dict, total_seconds)."""
    stats = pstats.Stats(prof_path)

    parent_to_children: dict[tuple, list[tuple[tuple, float]]] = {}
    for func_key, (_, _, _, _, callers) in stats.stats.items():
        for caller_key, caller_info in callers.items():
            if isinstance(caller_info, tuple):
                c_ct = caller_info[3] if len(caller_info) >= 4 else float(caller_info[0])
            else:
                c_ct = 0.0
            parent_to_children.setdefault(caller_key, []).append((func_key, c_ct))

    roots = [f for f, (_, _, _, _, callers) in stats.stats.items() if not callers]

    def _build(func_key: tuple, cumtime: float, visited: frozenset) -> dict:
        if func_key in visited:
            return {"name": _func_label(func_key), "value": max(1, int(cumtime * 1e6))}
        visited = visited | {func_key}
        children = []
        for child_key, child_ct in parent_to_children.get(func_key, []):
            children.append(_build(child_key, child_ct, visited))
        node: dict = {
            "name": _func_label(func_key),
            "value": max(1, int(cumtime * 1e6)),
        }
        if children:
            node["children"] = children
        return node

    root_children = []
    total_time = 0.0
    for r in roots:
        ct = stats.stats[r][3]
        total_time += ct
        root_children.append(_build(r, ct, frozenset()))

    return {
        "name": "all",
        "value": max(1, int(total_time * 1e6)),
        "children": root_children,
    }, total_time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate interactive flame graph HTML from train_bpe .prof file",
    )
    parser.add_argument("prof_file", type=str, help="Path to .prof file.")
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Output HTML path. Default: <prof_file>.html",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default browser.",
    )

    args = parser.parse_args(argv)
    prof_path = Path(args.prof_file)

    if not prof_path.is_file():
        print(f"Error: file not found: {prof_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else prof_path.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tree, total_time = _build_flamegraph_tree(str(prof_path))
    except Exception as exc:
        print(f"Error reading profile: {exc}", file=sys.stderr)
        return 1

    html = _HTML_TEMPLATE.format(
        prof_name=prof_path.name,
        total_time_s=total_time,
        data_json=json.dumps(tree),
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Flame graph saved: {out_path}")

    if args.open:
        webbrowser.open(str(out_path.resolve()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
