import re
import ast
import numpy as np
from datasets import load_from_disk
from datetime import datetime, timedelta

from . import report
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult


def extract_binary_probability(response_text: str) -> float | None:
    r"""Extract binary probability from response using \prediction{0.XX} pattern."""
    match = re.search(r'\\prediction\{([0-9.]+)\}', response_text)
    if match:
        point_estimate = float(match.group(1))
        # Handle percentage format (convert to 0-1 range if needed)
        if point_estimate > 1.0:
            point_estimate = point_estimate / 100.0
        return point_estimate
    return None


def extract_multiple_choice_probabilities(response_text: str, outcomes: list[str]) -> dict[str, float] | None:
    r"""Extract multiple choice probabilities from response.

    Looks for pattern: \prediction{[{'option_a': 0.xx}, {'option_b': 0.yy}, ...]}
    """
    # Find \prediction{ and then match braces
    pattern = r'\\prediction\{'
    start_match = re.search(pattern, response_text)

    if not start_match:
        return None

    # Find matching braces starting from the first {
    start_idx = start_match.end()
    brace_count = 1
    idx = start_idx

    while idx < len(response_text) and brace_count > 0:
        if response_text[idx] == '{':
            brace_count += 1
        elif response_text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count != 0:
        return None

    try:
        # Extract the list string and parse it as Python literal
        list_str = response_text[start_idx:idx-1]
        option_list = ast.literal_eval(list_str)

        if not isinstance(option_list, list):
            return None

        # Convert list of dicts to single dict with probabilities
        converted_probs = {}
        for item in option_list:
            if isinstance(item, dict):
                for key, value in item.items():
                    prob = float(value)
                    if prob > 1.0:
                        prob = prob / 100.0
                    converted_probs[str(key)] = prob

        return converted_probs if converted_probs else None
    except (ValueError, SyntaxError, TypeError):
        return None


def extract_numeric_percentiles(response_text: str) -> list[dict] | None:
    r"""Extract numeric percentiles from response.

    Looks for pattern: \prediction{[{"percentile": 10, "value": 5.2}, ...]}
    Returns: [{"percentile": 10, "value": 5.2}, ...]
    """
    # Find \prediction{ and then match braces
    pattern = r'\\prediction\{'
    start_match = re.search(pattern, response_text)

    if not start_match:
        return None

    # Find matching braces starting from the first {
    start_idx = start_match.end()
    brace_count = 1
    idx = start_idx

    while idx < len(response_text) and brace_count > 0:
        if response_text[idx] == '{':
            brace_count += 1
        elif response_text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count != 0:
        return None

    try:
        # Extract the list string and parse it as Python literal
        list_str = response_text[start_idx:idx-1]
        percentiles = ast.literal_eval(list_str)

        if not isinstance(percentiles, list):
            return None

        # Validate structure and convert to expected format
        validated_percentiles = []
        for item in percentiles:
            if isinstance(item, dict) and "percentile" in item and "value" in item:
                validated_percentiles.append({
                    "percentile": float(item["percentile"]),
                    "value": float(item["value"])
                })

        if not validated_percentiles:
            return None

        # Sort by percentile
        validated_percentiles.sort(key=lambda x: x["percentile"])

        return validated_percentiles
    except (ValueError, SyntaxError):
        return None


def calculate_binary_log_score(probability: float, ground_truth: float) -> float:
    """Calculate log score for binary question.

    Args:
        probability: Predicted probability (0-1)
        ground_truth: Actual outcome (0.0 or 1.0)

    Returns:
        log(p) if y=1, log(1-p) if y=0
    """
    # Clip to avoid log(0)
    probability = np.clip(probability, 1e-10, 1 - 1e-10)

    if ground_truth == 1.0:
        return float(np.log(probability))
    else:
        return float(np.log(1 - probability))


def calculate_brier_score(probability: float, ground_truth: float) -> float:
    """Calculate Brier score for binary question.

    Args:
        probability: Predicted probability (0-1)
        ground_truth: Actual outcome (0.0 or 1.0)

    Returns:
        -((p - y)^2)
    """
    return -((probability - ground_truth) ** 2)


def calculate_multiple_choice_log_score(option_probs: dict[str, float], correct_outcome: str) -> tuple[float, dict[str, float]]:
    """Calculate log score for multiple choice question.

    Args:
        option_probs: Dict mapping outcome -> probability
        correct_outcome: The correct outcome string

    Returns:
        Tuple of (log_score, normalized_probs)
    """
    # Normalize probabilities to sum to 1.0
    total = sum(option_probs.values())
    if total == 0:
        # Fallback: uniform distribution
        normalized_probs = {k: 1.0 / len(option_probs) for k in option_probs}
    else:
        normalized_probs = {k: v / total for k, v in option_probs.items()}

    p_correct = normalized_probs.get(correct_outcome, 0.0)

    # Clip to avoid log(0)
    p_correct = np.clip(p_correct, 1e-10, 1.0)

    return float(np.log(p_correct)), normalized_probs


def calculate_numeric_log_score(
    percentiles_raw: list[dict],
    resolution: float,
    range_min: float,
    range_max: float,
    open_lower_bound: bool,
    open_upper_bound: bool,
    zero_point: float | None,
) -> float:
    """Calculate log score for numeric question using Metaculus CDF method.

    This matches the implementation in src/forecasting_agent/art_agent/utils.py
    """
    from .utils import (
        convert_agent_percentiles_to_metaculus_format,
        validate_and_get_boundary_probabilities,
        generate_continuous_cdf,
        standardize_cdf,
        generate_cdf_x_axis,
        calculate_log_score_from_cdf,
    )

    # Build question_data dict in Metaculus format
    question_data = {
        "type": "numeric",
        "scaling": {
            "range_min": range_min,
            "range_max": range_max,
            "zero_point": zero_point,
        },
        "open_lower_bound": open_lower_bound,
        "open_upper_bound": open_upper_bound,
    }

    percentiles_dict = convert_agent_percentiles_to_metaculus_format(percentiles_raw)

    below_lower_bound, above_upper_bound = validate_and_get_boundary_probabilities(
        percentiles_raw, range_min, range_max
    )

    cdf_y_values = generate_continuous_cdf(
        percentiles_dict,
        question_data,
        below_lower_bound=below_lower_bound,
        above_upper_bound=above_upper_bound,
    )

    cdf_y_values = standardize_cdf(cdf_y_values, question_data)

    cdf_x_values = generate_cdf_x_axis(question_data, num_points=201)

    log_score, _ = calculate_log_score_from_cdf(
        cdf_y_values, cdf_x_values, resolution, question_data
    )

    return float(log_score)


class MetaculusEval(Eval):

    def __init__(
        self,
        data_path: str = "data/forecast_snapshot_metaculus",
        num_examples: int | None = None,
        cutoff_types: list[str] = ["1day", "3day", "1week", "2week", "1month"],
        num_threads: int = 4,
    ):
        super().__init__(num_threads)

        dataset = load_from_disk(data_path)

        # Convert to list of dicts
        questions = []
        for item in dataset:
            questions.append(dict(item))

        # Filter out unresolved questions
        filtered_questions = []
        for q in questions:
            question_type = q["question_type"]

            if question_type == "binary":
                # Keep only if resolution is 'yes' or 'no'
                if q.get("resolution") in ["yes", "no"]:
                    filtered_questions.append(q)

            elif question_type == "multiple_choice":
                # Keep only if outcome is not null
                if q.get("outcome") is not None and q.get("outcomes") is not None:
                    filtered_questions.append(q)

            elif question_type == "numeric":
                # Keep only if resolution_numeric is not NaN
                resolution_numeric = q.get("resolution_numeric")
                if resolution_numeric is not None and isinstance(resolution_numeric, float):
                    filtered_questions.append(q)

        questions = filtered_questions

        # Subsample if requested
        if num_examples is not None:
            questions = questions[:num_examples]

        # Expand to multiple examples per cutoff type
        self.examples = []
        for question in questions:
            for cutoff_type in cutoff_types:
                # Map cutoff_type to dataset field
                cutoff_field_map = {
                    "1day": ("community_pred_1day", 1),
                    "3day": ("community_pred_3day", 3),
                    "1week": ("community_pred_1week", 7),
                    "2week": ("community_pred_2week", 14),
                    "1month": ("community_pred_1month", 30),
                }

                if cutoff_type not in cutoff_field_map:
                    continue

                field_name, days_ago = cutoff_field_map[cutoff_type]
                cutoff_price = question.get(field_name)

                # Skip if cutoff price is not available
                if cutoff_price is None or (isinstance(cutoff_price, float) and np.isnan(cutoff_price)):
                    continue

                # Calculate cutoff date
                snapshot_datetime = question.get("snapshot_datetime")
                if snapshot_datetime:
                    snapshot_dt = datetime.fromisoformat(snapshot_datetime)
                    cutoff_dt = snapshot_dt - timedelta(days=days_ago)
                    cutoff_date = cutoff_dt.strftime("%Y-%m-%d")
                else:
                    cutoff_date = None

                example = {
                    "question": question["question"],
                    "question_type": question["question_type"],
                    "description": question.get("description", ""),
                    "resolution_criteria": question.get("resolution_criteria", ""),
                    "url": question.get("url", ""),
                    "categories": question.get("categories", "[]"),
                    "cutoff_type": cutoff_type,
                    "cutoff_price": cutoff_price,
                    "cutoff_date": cutoff_date,
                    "snapshot_datetime": snapshot_datetime,
                    "close_datetime": question.get("close_datetime"),
                }

                # Add question-type-specific fields
                if question["question_type"] == "binary":
                    example["resolution"] = question["resolution"]
                    example["resolution_numeric"] = float(question["resolution_numeric"])

                elif question["question_type"] == "multiple_choice":
                    example["outcome"] = question["outcome"]
                    example["resolution"] = question["resolution"]
                    example["outcomes"] = question["outcomes"]

                elif question["question_type"] == "numeric":
                    example["resolution_numeric"] = float(question["resolution_numeric"])
                    example["range_min"] = question.get("range_min")
                    example["range_max"] = question.get("range_max")
                    example["zero_point"] = question.get("zero_point")
                    example["open_lower_bound"] = question.get("open_lower_bound", False)
                    example["open_upper_bound"] = question.get("open_upper_bound", False)
                    example["unit"] = question.get("unit", "")

                self.examples.append(example)

    def __call__(self, sampler: SamplerBase, checkpoint_path=None) -> EvalResult:
        def fn(row: dict):
            user_message = f"""Question:
{row['question']}

Description:
{row['description']}

Resolution Criteria:
{row['resolution_criteria']}

Question Type:
{row['question_type']}
"""
            if row['question_type'] == 'multiple_choice':
                user_message += f"""\n\nPossible Options:\n{row['outcomes']}"""
                
            if row['question_type'] == 'numeric':
                if not row['open_lower_bound']:
                    user_message += f"\n\nLower Bound:\n{row['range_min']}"
                if not row["open_upper_bound"]:
                    user_message += f"\n\nUpper Bound:\n{row['range_max']}"

            sampler_response = sampler([
                sampler._pack_message(content=user_message, role="user")
            ])
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list

            question_type = row["question_type"]
            metrics = {}
            prediction = None

            if question_type == "binary":
                point_estimate = extract_binary_probability(response_text)
                prediction = point_estimate

                metrics["extraction_success"] = 1.0 if point_estimate is not None else 0.0

                if point_estimate is not None:
                    ground_truth = row["resolution_numeric"]
    
                    metrics["log_score"] = calculate_binary_log_score(point_estimate, ground_truth)
                    metrics["brier_score"] = calculate_brier_score(point_estimate, ground_truth)
                    metrics["abs_diff_from_resolution"] = abs(point_estimate - ground_truth)

                    score = metrics["log_score"]
                else:
                    metrics["log_score"] = -10.0
                    metrics["brier_score"] = -1.0
                    metrics["abs_diff_from_resolution"] = 1.0
                    score = -10.0

            elif question_type == "multiple_choice":
                correct_outcome = row["outcome"]
                option_probs = extract_multiple_choice_probabilities(response_text, [correct_outcome])

                prediction = option_probs
                metrics["extraction_success"] = 1.0 if option_probs is not None else 0.0

                if option_probs is not None and correct_outcome in option_probs:
                    log_score, normalized_probs = calculate_multiple_choice_log_score(
                        option_probs, correct_outcome
                    )

                    metrics["log_score"] = log_score
                    metrics["probability_on_correct"] = normalized_probs.get(correct_outcome, 0.0)

                    score = metrics["log_score"]
                else:
                    metrics["log_score"] = -10.0
                    metrics["probability_on_correct"] = 0.0
                    score = -10.0

            elif question_type == "numeric":
                percentiles_raw = extract_numeric_percentiles(response_text)
                prediction = percentiles_raw

                metrics["extraction_success"] = 1.0 if percentiles_raw is not None else 0.0

                if percentiles_raw is not None:
                    try:
                        log_score = calculate_numeric_log_score(
                            percentiles_raw=percentiles_raw,
                            resolution=row["resolution_numeric"],
                            range_min=row["range_min"],
                            range_max=row["range_max"],
                            open_lower_bound=row["open_lower_bound"],
                            open_upper_bound=row["open_upper_bound"],
                            zero_point=row["zero_point"],
                        )

                        metrics["log_score"] = log_score
                        metrics["num_percentiles"] = len(percentiles_raw)

                        score = metrics["log_score"]
                    except Exception:
                        # CDF generation failed
                        metrics["log_score"] = -10.0
                        score = -10.0
                else:
                    metrics["log_score"] = -10.0
                    score = -10.0

            html = self._generate_html_report(
                row=row,
                response_text=response_text,
                prediction=prediction,
                metrics=metrics,
                actual_queried_prompt_messages=actual_queried_prompt_messages,
            )

            convo = actual_queried_prompt_messages + [
                dict(role="assistant", content=response_text)
            ]

            example_level_metadata = {
                "question": row["question"],
                "question_type": question_type,
                "cutoff_type": row["cutoff_type"],
                "cutoff_date": row["cutoff_date"],
                "cutoff_community_pred": row["cutoff_price"],
                "prediction": str(prediction),
            }

            if question_type == "binary":
                example_level_metadata["resolution"] = row["resolution"]
                example_level_metadata["resolution_numeric"] = row["resolution_numeric"]
            elif question_type == "multiple_choice":
                example_level_metadata["correct_outcome"] = row["outcome"]
            elif question_type == "numeric":
                example_level_metadata["resolution"] = row["resolution_numeric"]
                example_level_metadata["range"] = f"[{row['range_min']}, {row['range_max']}]"

            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
                example_level_metadata=example_level_metadata,
            )

        if checkpoint_path:
            map_fn = report.with_checkpoint(checkpoint_path)(report.map_with_progress)
            results = map_fn(fn, self.examples, num_threads=self.num_threads)
        else:
            results = report.map_with_progress(fn, self.examples, num_threads=self.num_threads)

        return report.aggregate_results(results)

    def _generate_html_report(
        self,
        row: dict,
        response_text: str,
        prediction: any,
        metrics: dict,
        actual_queried_prompt_messages: list,
    ) -> str:
        """Generate HTML report for a single evaluation."""

        return ""