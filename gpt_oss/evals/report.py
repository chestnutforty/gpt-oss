import json
import os
import threading
from collections import defaultdict
from dataclasses import asdict
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable

import jinja2
import numpy as np
from tqdm import tqdm

from .types import Checkpoint, EvalResult, Message, SingleEvalResult


HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""


def _compute_stat(values: list, stat: str):
    if stat == "mean":
        return np.mean(values)
    elif stat == "std":
        return np.std(values)
    elif stat == "min":
        return np.min(values)
    elif stat == "max":
        return np.max(values)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        return np.std(
            [np.mean(np.random.choice(values, len(values))) for _ in range(1000)]
        )
    else:
        raise ValueError(f"Unknown {stat =}")


def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str, ...] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)
        htmls.append(single_eval_result.html)
        convos.append(single_eval_result.convo)
        metadata.append(single_eval_result.example_level_metadata)
    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)
    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
        single_eval_results=single_eval_results,
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = 128,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap_unordered(f, xs), total=len(xs)))


jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""


def message_to_html(message: Message) -> str:
    """
    Generate HTML snippet (inside a <div>) for a message.
    """
    return jinja_env.from_string(_message_template).render(
        role=message.get("role", message.get("type", None)),
        content=message.get("content", ""),
        variant=message.get("variant", None),
    )


jinja_env.globals["message_to_html"] = message_to_html


_report_template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
</html>
"""


def make_report(eval_result: EvalResult) -> str:
    """
    Create a standalone HTML report from an EvalResult.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )


_checkpoint_lock = threading.Lock()


def _save_checkpoint_unlocked(path: Path, checkpoint: Checkpoint):
    """Save checkpoint to disk atomically (assumes lock is held)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    try:
        with open(temp, "w") as f:
            json.dump({"results": {str(k): v for k, v in checkpoint.results.items()}}, f)
        temp.replace(path)
    except Exception as e:
        if temp.exists():
            temp.unlink()
        raise e


def _save_checkpoint(path: Path, checkpoint: Checkpoint):
    """Save checkpoint to disk atomically with locking."""
    with _checkpoint_lock:
        _save_checkpoint_unlocked(path, checkpoint)


def _load_checkpoint(path: Path) -> Checkpoint | None:
    """Load checkpoint from disk."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return Checkpoint(results={int(k): v for k, v in data["results"].items()})
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def with_checkpoint(checkpoint_path: Path | None):
    """
    Decorator that adds checkpoint/resume capability to map_with_progress.

    Usage:
        @with_checkpoint(Path("checkpoint.json"))
        def process_examples(f, xs, num_threads):
            return map_with_progress(f, xs, num_threads)
    """
    def decorator(map_fn):
        def wrapper(f: Callable, xs: list[Any], num_threads: int = 128, pbar: bool = True):
            if checkpoint_path is None:
                # No checkpointing, use original function
                return map_fn(f, xs, num_threads, pbar)

            # Load existing checkpoint
            checkpoint = _load_checkpoint(checkpoint_path)
            results_by_idx = {}

            if checkpoint:
                results_by_idx = {idx: SingleEvalResult(**data) for idx, data in checkpoint.results.items()}
                print(f"Resuming: {len(results_by_idx)}/{len(xs)} completed")

            # Filter to remaining work
            remaining = [(i, x) for i, x in enumerate(xs) if i not in results_by_idx]

            if not remaining:
                print("All examples completed!")
                return [results_by_idx[i] for i in range(len(xs))]

            # Wrapper that checkpoints after each result
            def f_with_checkpoint(idx_and_x):
                idx, x = idx_and_x
                result = f(x)
                
                # Update dict and save checkpoint atomically
                with _checkpoint_lock:
                    results_by_idx[idx] = result
                    checkpoint_data = Checkpoint(results={i: asdict(r) for i, r in results_by_idx.items()})
                    _save_checkpoint_unlocked(checkpoint_path, checkpoint_data)

                return result

            # Run remaining work
            map_fn(f_with_checkpoint, remaining, num_threads, pbar)

            # Return all results in order
            return [results_by_idx[i] for i in range(len(xs))]

        return wrapper
    return decorator
