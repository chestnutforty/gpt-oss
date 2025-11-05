import json
import re
from datetime import datetime

from . import report
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult


def extract_probability(response_text: str) -> tuple[float | None, float | None, float | None]:
    """
    Extract probability estimate and uncertainty band from model response.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    # Try to extract \pointestimate{0.XX} pattern
    point_estimate = None
    match = re.search(r'\\pointestimate\{([0-9.]+)\}', response_text)
    if match:
        point_estimate = float(match.group(1))

    # Try to extract \uncertaintyband{0.XX-0.YY} pattern
    lower_bound = None
    upper_bound = None
    match = re.search(r'\\uncertaintyband\{([0-9.]+)-([0-9.]+)\}', response_text)
    if match:
        lower_bound = float(match.group(1))
        upper_bound = float(match.group(2))

    # If no point estimate found, try other patterns
    if point_estimate is None:
        # Try "X% probability" or "X percent probability"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:probability|chance|likelihood)', response_text.lower())
        if match:
            point_estimate = float(match.group(1)) / 100.0

        # Try "probability of X%" or "probability of 0.X"
        if point_estimate is None:
            match = re.search(r'probability\s+(?:of|is)\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%', response_text.lower())
            if match:
                point_estimate = float(match.group(1)) / 100.0

        # Try "probability of 0.XX" (decimal without %)
        if point_estimate is None:
            match = re.search(r'probability\s+(?:of|is)\s+(?:approximately\s+)?(0\.\d+)', response_text.lower())
            if match:
                point_estimate = float(match.group(1))

        # Try decimal probability "0.XX probability"
        if point_estimate is None:
            match = re.search(r'(0\.\d+)\s+(?:probability|chance|likelihood)', response_text.lower())
            if match:
                point_estimate = float(match.group(1))

    # Handle percentage format (convert to 0-1 range if needed)
    if point_estimate is not None and point_estimate > 1.0:
        point_estimate = point_estimate / 100.0

    if lower_bound is not None and lower_bound > 1.0:
        lower_bound = lower_bound / 100.0

    if upper_bound is not None and upper_bound > 1.0:
        upper_bound = upper_bound / 100.0

    return point_estimate, lower_bound, upper_bound


def calculate_brier_score(prediction: float, ground_truth: float) -> float:
    """
    Calculate Brier score (lower is better).
    Treats ground_truth as the "true" probability.
    """
    return (prediction - ground_truth) ** 2


def calculate_directional_accuracy(prediction: float, cutoff_price: float, latest_price: float) -> int:
    """
    Check if prediction direction matches market movement direction.

    Returns 1 if correct, 0 if incorrect.
    """
    # Market movement direction
    market_moved_up = latest_price > cutoff_price
    market_moved_down = latest_price < cutoff_price

    # Prediction direction (relative to cutoff)
    predicted_up = prediction > cutoff_price
    predicted_down = prediction < cutoff_price

    # Check if directions match
    if market_moved_up and predicted_up:
        return 1
    elif market_moved_down and predicted_down:
        return 1
    elif latest_price == cutoff_price:
        # If market didn't move, consider correct if prediction is close
        return 1 if abs(prediction - cutoff_price) < 0.05 else 0
    else:
        return 0


class PolymarketEval(Eval):
    """
    Evaluation for Polymarket forecasting.
    """

    def __init__(
        self,
        data_path: str = "data/polymarket_politics_resolve_nov15.jsonl",
        num_examples: int | None = None,
        cutoff_types: list[str] = ["day", "week", "month"],
    ):
        """
        Args:
            data_path: Path to JSONL dataset
            num_examples: Number of questions to sample (None = all)
            cutoff_types: Which cutoff dates to use: "day", "week", "month"
        """
        # Load JSONL data
        with open(data_path, 'r') as f:
            questions = [json.loads(line) for line in f]

        # Subsample if requested
        if num_examples is not None:
            questions = questions[:num_examples]

        # Expand to multiple examples per cutoff type
        self.examples = []
        for question in questions:
            for cutoff_type in cutoff_types:
                example = {
                    "question": question["question"],
                    "description": question["description"],
                    "lastTradePrice": question["lastTradePrice"],
                    "startDate": question["startDate"],
                    "endDate": question["endDate"],
                    "cutoff_type": cutoff_type,
                }

                # Add cutoff-specific data
                if cutoff_type == "day":
                    example["cutoff_price"] = question["price_day_ago"]
                    example["cutoff_date"] = question["date_day_ago"]
                elif cutoff_type == "week":
                    example["cutoff_price"] = question["price_week_ago"]
                    example["cutoff_date"] = question["date_week_ago"]
                elif cutoff_type == "month":
                    example["cutoff_price"] = question["price_month_ago"]
                    example["cutoff_date"] = question["date_month_ago"]
                else:
                    raise ValueError(f"Unknown cutoff_type: {cutoff_type}")

                self.examples.append(example)

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            # Format user message from template
            user_message = f"""Question:
{row['question']}

Description and Resolution Criteria:
{row['description']}
"""

            # Update sampler cutoff date if it's a VLLMSampler
            if hasattr(sampler, 'cutoff_date'):
                # Parse cutoff_date and format it
                cutoff_dt = datetime.fromisoformat(row['cutoff_date'].replace('+00:00', ''))
                sampler.cutoff_date = cutoff_dt.strftime("%Y-%m-%d")

            # Call sampler
            sampler_response = sampler([
                sampler._pack_message(content=user_message, role="user")
            ])
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list

            # Extract probability estimate
            point_estimate, lower_bound, upper_bound = extract_probability(response_text)

            # Calculate metrics
            metrics = {}

            # Extraction success
            metrics["extraction_success"] = 1.0 if point_estimate is not None else 0.0

            # If we extracted a probability, calculate all metrics
            if point_estimate is not None:
                # Brier score vs latest price (ground truth proxy)
                metrics["brier_score_vs_latest"] = calculate_brier_score(
                    point_estimate, row["lastTradePrice"]
                )

                # Brier score vs cutoff price
                metrics["brier_score_vs_cutoff"] = calculate_brier_score(
                    point_estimate, row["cutoff_price"]
                )

                # Absolute difference from latest price
                metrics["abs_diff_from_latest"] = abs(point_estimate - row["lastTradePrice"])

                # Directional accuracy
                metrics["directional_accuracy"] = calculate_directional_accuracy(
                    point_estimate, row["cutoff_price"], row["lastTradePrice"]
                )

                # Uncertainty width (if available)
                if lower_bound is not None and upper_bound is not None:
                    metrics["uncertainty_width"] = upper_bound - lower_bound

                # Overall score (use negative Brier score so higher is better)
                score = -metrics["brier_score_vs_latest"]
            else:
                # Failed to extract, assign worst scores
                metrics["brier_score_vs_latest"] = 1.0  # Worst possible Brier score
                metrics["brier_score_vs_cutoff"] = 1.0
                metrics["abs_diff_from_latest"] = 1.0
                metrics["directional_accuracy"] = 0.0
                score = -1.0

            # Generate HTML report
            html = self._generate_html_report(
                row=row,
                response_text=response_text,
                point_estimate=point_estimate,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                metrics=metrics,
                actual_queried_prompt_messages=actual_queried_prompt_messages,
            )

            # Construct conversation
            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role="assistant")
            ]

            # Store metadata
            example_level_metadata = {
                "question": row["question"],
                "cutoff_type": row["cutoff_type"],
                "cutoff_date": row["cutoff_date"],
                "cutoff_price": row["cutoff_price"],
                "latest_price": row["lastTradePrice"],
                "prediction": point_estimate,
                "uncertainty_band": f"{lower_bound}-{upper_bound}" if lower_bound and upper_bound else None,
            }

            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
                example_level_metadata=example_level_metadata,
            )

        # Run evaluation in parallel
        results = report.map_with_progress(fn, self.examples, num_threads=32)
        return report.aggregate_results(results)

    def _generate_html_report(
        self,
        row: dict,
        response_text: str,
        point_estimate: float | None,
        lower_bound: float | None,
        upper_bound: float | None,
        metrics: dict,
        actual_queried_prompt_messages: list,
    ) -> str:
        """Generate HTML report for a single evaluation."""

        # Calculate price change
        price_change = row["lastTradePrice"] - row["cutoff_price"]
        price_change_pct = (price_change / row["cutoff_price"] * 100) if row["cutoff_price"] > 0 else 0

        # Color code based on performance
        brier_color = "green" if metrics.get("brier_score_vs_latest", 1.0) < 0.1 else "orange" if metrics.get("brier_score_vs_latest", 1.0) < 0.25 else "red"
        direction_color = "green" if metrics.get("directional_accuracy", 0.0) == 1.0 else "red"

        html_template = """
<div style="border: 1px solid #ccc; padding: 16px; margin-bottom: 16px;">
    <h2>{{ question }}</h2>

    <h3>Market Information</h3>
    <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Cutoff Type</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{{ cutoff_type }}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Cutoff Date</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{{ cutoff_date }}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Price at Cutoff</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{{ "%.3f"|format(cutoff_price) }}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Latest Price</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{{ "%.3f"|format(latest_price) }}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Market Movement</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">
                {{ "%+.3f"|format(price_change) }} ({{ "%+.1f"|format(price_change_pct) }}%)
            </td>
        </tr>
    </table>

    <h3>Agent Prediction</h3>
    <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Point Estimate</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">
                {% if point_estimate is not none %}
                    {{ "%.3f"|format(point_estimate) }}
                {% else %}
                    <span style="color: red;">FAILED TO EXTRACT</span>
                {% endif %}
            </td>
        </tr>
        {% if lower_bound is not none and upper_bound is not none %}
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Uncertainty Band</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">
                {{ "%.3f"|format(lower_bound) }} - {{ "%.3f"|format(upper_bound) }}
            </td>
        </tr>
        {% endif %}
    </table>

    <h3>Metrics</h3>
    <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Brier Score vs Latest</b></td>
            <td style="padding: 8px; border: 1px solid #ddd; color: {{ brier_color }};">
                {{ "%.4f"|format(metrics.get('brier_score_vs_latest', 1.0)) }}
            </td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Brier Score vs Cutoff</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">
                {{ "%.4f"|format(metrics.get('brier_score_vs_cutoff', 1.0)) }}
            </td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Absolute Diff from Latest</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">
                {{ "%.4f"|format(metrics.get('abs_diff_from_latest', 1.0)) }}
            </td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><b>Directional Accuracy</b></td>
            <td style="padding: 8px; border: 1px solid #ddd; color: {{ direction_color }};">
                {{ "Correct" if metrics.get('directional_accuracy', 0.0) == 1.0 else "Incorrect" }}
            </td>
        </tr>
    </table>

    <h3>Prompt Conversation</h3>
    {% for message in prompt_messages %}
    {{ message_to_html(message) | safe }}
    {% endfor %}

    <h3>Agent Response</h3>
    {{ message_to_html(next_message) | safe }}
</div>
"""

        return report.jinja_env.from_string(html_template).render(
            question=row["question"],
            cutoff_type=row["cutoff_type"],
            cutoff_date=row["cutoff_date"],
            cutoff_price=row["cutoff_price"],
            latest_price=row["lastTradePrice"],
            price_change=price_change,
            price_change_pct=price_change_pct,
            point_estimate=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metrics=metrics,
            brier_color=brier_color,
            direction_color=direction_color,
            prompt_messages=actual_queried_prompt_messages,
            next_message=dict(content=response_text, role="assistant"),
        )
