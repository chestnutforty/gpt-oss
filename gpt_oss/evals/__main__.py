import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from . import report
from .basic_eval import BasicEval
# from .gpqa_eval import GPQAEval
# from .aime_eval import AIME25Eval
# from .healthbench_eval import HealthBenchEval
from .polymarket_eval import PolymarketEval
from .chat_completions_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionsSampler,
)
from .responses_sampler import ResponsesSampler
from .api_sampler import ApiSampler
from .metaculus_eval import MetaculusEval
from .recursive_sampler import RecursiveSampler


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate the models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-20b",
        help="Select a model by name. Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        help="Reasoning effort (low, medium, high). Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Verbosity (low, medium, high).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["responses", "chat_completions", "api", "recursive"],
        default="responses",
        help="Sampler backend to use for models.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the API.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="gpqa,healthbench,healthbench_hard,healthbench_consensus,aime25",
        help="Select an eval by name. Options: basic, gpqa, healthbench, healthbench_hard, healthbench_consensus, aime25, polymarket. Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="Number of threads to run.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode"
    )
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for VLLM sampler",
    )
    parser.add_argument(
        "--mcp",
        default="",
        help="Comma-separated list of MCP tools (wikipedia,browser,python).",
    )
    parser.add_argument(
        "--enable-internal-browser",
        action="store_true",
        help="Enable internal browser tool (web_search) handled by API server",
    )
    parser.add_argument(
        "--enable-internal-python",
        action="store_true",
        help="Enable internal python tool (code_interpreter) handled by API server",
    )
    parser.add_argument(
        "--developer-message",
        type=str,
        help="Developer message md file in prompts/ directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to store evaluation checkpoints",
    )

    # Polymarket eval arguments
    parser.add_argument(
        "--polymarket-data-path",
        type=str,
        default="data/polymarket_politics_resolve_nov15.jsonl",
        help="Path to Polymarket JSONL dataset",
    )
    parser.add_argument(
        "--polymarket-cutoff-types",
        nargs="+",
        choices=["day", "week", "month"],
        default=["month"],
        help="Which cutoff dates to use for Polymarket eval",
    )
    
    # Metaculus eval arguments
    parser.add_argument(
        "--metaculus-data-path",
        type=str,
        default="data/forecast_snapshot_metaculus",
        help="Path to Metaculus dataset",
    )
    parser.add_argument(
        "--metaculus-cutoff-types",
        nargs="+",
        choices=["1day", "3day", "1week", "2week", "1month"],
        default=["1month"],
        help="Which cutoff dates to use for Metaculus eval",
    )
    parser.add_argument(
        "--metaculus-latest-only",
        action="store_true",
        help="Only use latest snapshot date for each question in Metaculus eval",
    )

    # Recursive sampler arguments
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Max depth for recursive sampler",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Max turns for recursive sampler",
    )

    args = parser.parse_args()
    
    developer_message = None
    if args.developer_message:
        developer_message = Path(__file__).parent.parent / "prompts" / f"{args.developer_message}.md"
        developer_message = developer_message.read_text(encoding="utf-8")
        
    # Build list of MCP servers to connect to
    mcp_servers = []
    if args.mcp:
        mcp_servers = args.mcp.split(",")
        mcp_servers = [{
            "wikipedia": ("wikipedia", 8003),
            "browser": ("browser", 8001),
            "python": ("python", 8002),
            "google-trends": ("google-trends", 8004),
            "metaculus": ("metaculus", 8005),
            "financial-datasets": ("financial-datasets", 8006),
            "datacommons": ("datacommons", 8007),
        }[mcp_server] for mcp_server in mcp_servers]
        print(f"MCP servers: {mcp_servers}")

    # Create models/samplers
    models = {}
    if args.sampler == "responses":
        sampler_cls = ResponsesSampler
    elif args.sampler == "chat_completions":
        sampler_cls = ChatCompletionsSampler
    elif args.sampler == "api":
        sampler_cls = ApiSampler
    elif args.sampler == "recursive":
        sampler_cls = RecursiveSampler
    else:
        raise ValueError(f"Invalid sampler: {args.sampler}")

    for model_name in args.model.split(","):
        for reasoning_effort in args.reasoning_effort.split(","):
            models[f"{model_name}-{reasoning_effort}"] = sampler_cls(
                model=model_name,
                reasoning_model=True,
                reasoning_effort=reasoning_effort,
                verbosity=args.verbosity,
                temperature=args.temperature,
                base_url=args.base_url,
                max_tokens=131_072,
                developer_message=developer_message,
                mcp_servers=mcp_servers if mcp_servers else None,
                enable_internal_browser=args.enable_internal_browser,
                enable_internal_python=args.enable_internal_python,
                max_depth=args.max_depth,
                max_turns=args.max_turns,
            )

    print(f"Running with args {args}")

    grading_sampler = ChatCompletionsSampler(
        model="gpt-4.1-2025-04-14",
        developer_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        base_url="https://api.openai.com/v1",
    )

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "basic":
                return BasicEval()
            case "polymarket":
                return PolymarketEval(
                    data_path=args.polymarket_data_path,
                    num_examples=num_examples,
                    cutoff_types=args.polymarket_cutoff_types,
                    num_threads=args.n_threads,
                )
            case "metaculus":
                return MetaculusEval(
                    data_path=args.metaculus_data_path,
                    num_examples=num_examples,
                    cutoff_types=args.metaculus_cutoff_types,
                    num_threads=args.n_threads,
                    latest_only=args.metaculus_latest_only,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {}
    for eval_name in args.eval.split(","):
        evals[eval_name] = get_evals(eval_name, args.debug)

    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {evals}")
    print(f"Running evals for the following models: {models}")

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")

    # Prepare checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for model_name, sampler in models.items():
        model_name = model_name.replace("/", "__")
        for eval_name, eval_obj in evals.items():
            file_stem = f"{eval_name}_{model_name}_reasoning_effort_{args.reasoning_effort}_temp{args.temperature}_python_{args.enable_internal_python}_browser_{args.enable_internal_browser}_mcp_{args.mcp}"
            checkpoint_path = checkpoint_dir / (file_stem + ".json")

            # If not resuming, delete any existing checkpoint to start fresh
            if not args.resume and checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"Starting fresh (deleted existing checkpoint)")

            result = eval_obj(sampler, checkpoint_path=checkpoint_path)
            # ^^^ how to use a sampler

            # Clean up checkpoint on successful completion
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"Checkpoint deleted: {checkpoint_path}")
            tools_metadata = {
                "sampler_type": args.sampler,
                "mcp_servers": [{"name": name, "port": port} for name, port in mcp_servers] if mcp_servers else [],
                "enable_internal_browser": args.enable_internal_browser,
                "enable_internal_python": args.enable_internal_python,
            }
            if result.metadata is None:
                result.metadata = {}
            result.metadata["tools"] = tools_metadata
            result.metadata["developer_message"] = args.developer_message

            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(report.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            single_eval_results_dict = None
            if result.single_eval_results:
                single_eval_results_dict = [
                    {
                        "score": ser.score,
                        "metrics": ser.metrics,
                        "html": ser.html,
                        "convo": ser.convo,
                        "example_level_metadata": ser.example_level_metadata,
                        "spans": ser.spans if ser.spans else None,
                    }
                    for ser in result.single_eval_results
                ]

            full_result_filename = f"/tmp/{file_stem}{debug_suffix}_allresults.json"
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                    "single_eval_results": single_eval_results_dict,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            local_result_path = f"results/{file_stem}{debug_suffix}_allresults.json"
            if not os.path.exists(local_result_path):
                os.makedirs(os.path.dirname(local_result_path), exist_ok=True)
            with open(local_result_path, "w") as f:
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {local_result_path}")

            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    print(merge_metrics)
    return merge_metrics


if __name__ == "__main__":
    main()
