"""
Utility functions for working with Metaculus numeric predictions.

These functions are taken directly from the Metaculus guide on continuous forecasts:
https://www.metaculus.com/help/continuous-forecasts/

They ensure that CDFs are properly formatted and meet all Metaculus requirements.
"""

import datetime
import numpy as np
from typing import Dict, Any
from types import SimpleNamespace

import weave
import art
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function


def scenario_to_metaculus_format(scenario: Any) -> Dict[str, Any]:
    """
    Convert our ForecastScenario format to Metaculus question_data format.

    This ensures compatibility with the official Metaculus utility functions.
    """
    return {
        "type": "numeric",  # Could be "date" for date questions
        "scaling": {
            "range_min": scenario.lower_bound,
            "range_max": scenario.upper_bound,
            "zero_point": scenario.zero_point,
        },
        "open_lower_bound": scenario.open_lower_bound,
        "open_upper_bound": scenario.open_upper_bound,
    }


def nominal_location_to_cdf_location(
    nominal_location: str | float,
    question_data: dict,
) -> float:
    """Takes a location in nominal format (e.g. 123, "123",
    or datetime in iso format) and scales it to metaculus's
    "internal representation" range [0,1] incorporating question scaling"""
    if question_data["type"] == "date":
        scaled_location = datetime.fromisoformat(nominal_location).timestamp()
    else:
        scaled_location = float(nominal_location)
    # Unscale the value to put it into the range [0,1]
    scaling = question_data["scaling"]
    range_min = scaling.get("range_min")
    range_max = scaling.get("range_max")
    zero_point = scaling.get("zero_point")
    if zero_point is not None:
        # logarithmically scaled question
        deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
        unscaled_location = (
            np.log(
                (scaled_location - range_min) * (deriv_ratio - 1)
                + (range_max - range_min)
            )
            - np.log(range_max - range_min)
        ) / np.log(deriv_ratio)
    else:
        # linearly scaled question
        unscaled_location = (scaled_location - range_min) / (range_max - range_min)
    return unscaled_location


def generate_continuous_cdf(
    percentiles: dict,
    question_data: dict,
    below_lower_bound: float = None,
    above_upper_bound: float = None,
) -> list[float]:
    """
    Takes a set of percentiles and returns a corresponding cdf with 201 values

    Param: percentiles
    dict[str, float | str]
    keys must terminate in a number interpretable as a float in range (0, 100)
      optionally preceded by an underscore "_"
    values must be a nominal value in the scale of the question, either
      interpretable as a float (for "numeric" type questions) or a datetime in
      ISO format (for "date" type questions)
    example percentiles:
    percentiles = {
      "percentile_01": 25,
      "precentile_25.123": 500,
      "50": 650,
      "percentile_75": "700",
      "percentile_99": 990,
    }
    optionally, include `below_lower_bound` and `above_upper_bound`
    to indicate the amount of probability mass assigned to those locations
    percentiles = {
      "percentile_25": 500,
      "percentile_50": 650,
      "percentile_75": 700,
    }
    below_lower_bound = 0.0025,
    above_upper_bound = 0.009,

    If the percentile locations don't encompass
      [scaling["range_min"], scaling["range_max"]]
    and "below_lower_bound"/"above_upper_bound" aren't provided,
    then the prediction can't be interpreted as a cdf properly.
    Note that range_min/range_max for date questions are unix timestamps.
    """

    # This will be the set of (x, y) points that are the set points
    # of the cdf
    percentile_locations = []

    # take the given boundary values
    if below_lower_bound is not None:
        percentile_locations.append((0.0, below_lower_bound))
    if above_upper_bound is not None:
        percentile_locations.append((1.0, 1 - above_upper_bound))

    # generate the remaining set of points
    for percentile, nominal_location in percentiles.items():
        height = float(str(percentile).split("_")[-1]) / 100
        location = nominal_location_to_cdf_location(nominal_location, question_data)
        percentile_locations.append((location, height))

    # sort to ensure lookup works
    percentile_locations.sort()

    # check validity
    first_point, last_point = percentile_locations[0], percentile_locations[-1]
    if (first_point[0] > 0.0) or (last_point[0] < 1.0):
        raise ValueError("Percentiles must encompass bounds of the question")

    def get_cdf_at(location):
        # helper function that takes a location and returns
        # the height of the cdf at that location, linearly
        # interpolating between values
        previous = percentile_locations[0]
        for i in range(1, len(percentile_locations)):
            current = percentile_locations[i]
            if previous[0] <= location <= current[0]:
                return previous[1] + (current[1] - previous[1]) * (
                    location - previous[0]
                ) / (current[0] - previous[0])
            previous = current

    # generate that cdf
    continuous_cdf = [get_cdf_at(i / 200) for i in range(201)]
    return continuous_cdf


def standardize_cdf(cdf: list[float], question_data: dict) -> list[float]:
    """
    Takes a cdf and returns a standardized version of it

    - assigns no mass outside of closed bounds (scales accordingly)
    - assigns at least a minimum amount of mass outside of open bounds
    - increasing by at least the minimum amount (0.01 / 200 = 0.0005)

    TODO: add smoothing over cdfs that spike too heavily (exceed a change of 0.59)
    """
    lower_open = question_data["open_lower_bound"]
    upper_open = question_data["open_upper_bound"]

    scale_lower_to = 0 if lower_open else cdf[0]
    scale_upper_to = 1.0 if upper_open else cdf[-1]
    rescaled_inbound_mass = scale_upper_to - scale_lower_to

    def standardize(F: float, location: float) -> float:
        # `F` is the height of the cdf at `location` (in range [0, 1])
        # rescale
        rescaled_F = (F - scale_lower_to) / rescaled_inbound_mass
        # offset
        if lower_open and upper_open:
            return 0.988 * rescaled_F + 0.01 * location + 0.001
        elif lower_open:
            return 0.989 * rescaled_F + 0.01 * location + 0.001
        elif upper_open:
            return 0.989 * rescaled_F + 0.01 * location
        return 0.99 * rescaled_F + 0.01 * location

    standardized_cdf = []
    for i, F in enumerate(cdf):
        standardized_F = standardize(F, i / (len(cdf) - 1))
        # round to avoid floating point errors
        standardized_cdf.append(round(standardized_F, 10))

    return standardized_cdf


def cdf_location_to_nominal_location(cdf_location: float, question_data: dict) -> float:
    """
    Convert internal CDF location [0,1] back to nominal value.
    Inverse of nominal_location_to_cdf_location.
    """
    scaling = question_data["scaling"]
    range_min = scaling.get("range_min")
    range_max = scaling.get("range_max")
    zero_point = scaling.get("zero_point")

    if zero_point is not None:
        # logarithmically scaled question
        deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
        scaled_location = range_min + (range_max - range_min) * (
            deriv_ratio**cdf_location - 1
        ) / (deriv_ratio - 1)
    else:
        # linearly scaled question
        scaled_location = range_min + (range_max - range_min) * cdf_location

    return scaled_location


def generate_cdf_x_axis(question_data: dict, num_points: int = 201) -> list[float]:
    """
    Generate the x-axis values for a CDF based on question scaling.
    Returns nominal values corresponding to each CDF location.
    """
    return [cdf_location_to_nominal_location(i / (num_points - 1), question_data)
            for i in range(num_points)]


def convert_agent_percentiles_to_metaculus_format(percentiles_raw: list[dict]) -> dict:
    """
    Convert agent's percentile format to Metaculus format.

    Input: [{"percentile": 10, "value": 5.2}, {"percentile": 50, "value": 10.0}, ...]
    Output: {"percentile_10": 5.2, "percentile_50": 10.0, ...}
    """
    return {f"percentile_{int(p['percentile'])}": p['value'] for p in percentiles_raw}


def validate_and_get_boundary_probabilities(
    percentiles_raw: list[dict],
    lower_bound: float,
    upper_bound: float,
) -> tuple[float | None, float | None]:
    """
    Validate percentiles are within bounds and calculate boundary probabilities.

    Rules:
    - If lowest percentile value > lower_bound AND percentile > 0 → below_lower_bound = 0.0
    - If lowest percentile value == lower_bound → below_lower_bound = None
    - If lowest percentile value < lower_bound → raise ValueError

    - If highest percentile value < upper_bound AND percentile < 100 → above_upper_bound = 0.0
    - If highest percentile value == upper_bound → above_upper_bound = None
    - If highest percentile value > upper_bound → raise ValueError

    Args:
        percentiles_raw: List of agent percentiles [{"percentile": 10, "value": 5.2}, ...]
        lower_bound: Question lower bound
        upper_bound: Question upper bound

    Returns:
        tuple[float | None, float | None]: (below_lower_bound, above_upper_bound)

    Raises:
        ValueError: If percentiles contain values outside the valid range
    """
    if not percentiles_raw:
        raise ValueError("No percentiles provided")

    # Sort by value to find min and max
    sorted_by_value = sorted(percentiles_raw, key=lambda p: p['value'])
    lowest = sorted_by_value[0]
    highest = sorted_by_value[-1]

    # Validate and set lower boundary
    below_lower_bound = None
    if lowest['value'] < lower_bound:
        raise ValueError(
            f"Invalid percentile: P{lowest['percentile']} has value {lowest['value']} "
            f"which is below the lower bound {lower_bound}. "
            f"All percentile values must be >= {lower_bound}."
        )
    elif lowest['value'] > lower_bound and lowest['percentile'] > 0:
        # Agent's distribution starts above the lower bound
        below_lower_bound = 0.0
    # else: lowest['value'] == lower_bound or lowest['percentile'] == 0 → below_lower_bound = None

    # Validate and set upper boundary
    above_upper_bound = None
    if highest['value'] > upper_bound:
        raise ValueError(
            f"Invalid percentile: P{highest['percentile']} has value {highest['value']} "
            f"which is above the upper bound {upper_bound}. "
            f"All percentile values must be <= {upper_bound}."
        )
    elif highest['value'] < upper_bound and highest['percentile'] < 100:
        # Agent's distribution ends below the upper bound
        above_upper_bound = 0.0
    # else: highest['value'] == upper_bound or highest['percentile'] == 100 → above_upper_bound = None

    return below_lower_bound, above_upper_bound


def calculate_log_score_from_cdf(
    cdf_y_values: list[float],
    cdf_x_values: list[float],
    true_value: float,
    question_data: dict,
) -> tuple[float, float]:
    """
    Calculate Metaculus log score from a CDF using PMF in internal coordinates.

    According to Metaculus:
    - The PDF (technically PMF) is derived as differences between consecutive CDF points
    - Scoring happens in the standardized [0,1] internal coordinate space
    - The standardization function already adds the uniform component

    This approach ensures scoring is invariant to question scaling (linear vs logarithmic).

    Args:
        cdf_y_values: The 201-point CDF values (probabilities)
        cdf_x_values: The 201-point CDF x-axis values (nominal coordinates)
        true_value: The true outcome value (in nominal coordinates)
        question_data: Question metadata with scaling information

    Returns:
        tuple[float, float]: (log_score, pdf_internal)
    """
    if len(cdf_x_values) != len(cdf_y_values):
        raise ValueError("CDF x and y values must have same length")

    if len(cdf_x_values) <= 1:
        # Fallback for degenerate case
        pdf_internal = 1.0 / len(cdf_x_values)
        return np.log(pdf_internal), pdf_internal

    # Convert true value to internal [0, 1] coordinates
    internal_location = nominal_location_to_cdf_location(true_value, question_data)
    internal_location = np.clip(internal_location, 0.0, 1.0)

    # Find which bin in the 201-point CDF (200 bins of equal width 1/200 in internal space)
    bin_index = int(internal_location * 200)
    bin_index = min(bin_index, 199)  # Clamp to valid range [0, 199]

    # PMF = probability mass in that bin
    pmf = cdf_y_values[bin_index + 1] - cdf_y_values[bin_index]

    # Convert to density in internal [0,1] space
    # Each bin has width 1/200 in internal coordinates
    pdf_internal = pmf / (1.0 / 200.0)  # = 200 * pmf

    # The standardization already added the uniform component,
    # so we score directly without adding anything
    log_score = np.log(pdf_internal)

    return log_score, pdf_internal

async def get_llm_completion(
    model: art.Model,
    messages: list[dict],
    tools: list[dict],
    temperature: float = 1.0,
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = None,
    previous_response_id: str | None = None,
):
    """Get LLM completion using responses API (GPT) or chat completions API (non-GPT)."""
    client = AsyncOpenAI(api_key=model.inference_api_key, base_url=model.inference_base_url)
    model_name = model.inference_model_name or model.name

    if "gpt" not in model_name.lower():
        return await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_completion_tokens=max_completion_tokens,
        )

    # GPT models use responses API
    api_params = {
        "model": model_name,
        "input": _convert_messages_to_responses_input(messages),
        "temperature": temperature,
        "tools": _convert_tools_to_responses_format(tools),
        "max_output_tokens": max_completion_tokens,
        "tool_choice": "auto",
        "store": True,
    }
    if previous_response_id:
        api_params["previous_response_id"] = previous_response_id
    if reasoning_effort:
        api_params["reasoning"] = {"effort": reasoning_effort}

    return _convert_responses_to_chat_completion(await client.responses.create(**api_params))


def _convert_messages_to_responses_input(messages: list[dict]) -> list:
    """Convert chat completions messages to responses API format."""
    responses_input = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            responses_input.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id"),
                "output": msg.get("content", "")
            })
        elif role == "assistant" and msg.get("tool_calls"):
            if msg.get("content"):
                responses_input.append({"role": "assistant", "content": msg["content"]})

            responses_input.extend([
                {
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"]
                }
                for tc in msg["tool_calls"]
            ])
        else:
            responses_input.append(msg)
    return responses_input


def _convert_tools_to_responses_format(tools: list[dict]) -> list[dict]:
    """Convert tool schemas from chat completions to responses API format."""
    return [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "parameters": tool["function"].get("parameters", {}),
        }
        if tool.get("type") == "function" and "function" in tool
        else tool
        for tool in tools
    ]


def _convert_responses_to_chat_completion(response):
    """Convert responses API format to chat completions format."""
    output = getattr(response, 'output', [])

    content_parts = []
    tool_calls = []

    for item in output:
        if not hasattr(item, 'type'):
            continue

        if item.type == "function_call":
            tool_calls.append(ChatCompletionMessageToolCall(
                id=item.call_id,
                type="function",
                function=Function(name=item.name, arguments=item.arguments)
            ))
        elif item.type == "message" and hasattr(item, 'content'):
            # Handle both list and single content formats
            if isinstance(item.content, list):
                content_parts.extend([
                    c.text for c in item.content
                    if hasattr(c, 'text') and hasattr(c, 'type') and c.type == "output_text"
                ])
            elif hasattr(item.content, 'text'):
                # Single content item (not a list)
                content_parts.append(item.content.text)

    content = "\n".join(content_parts) if content_parts else None

    message = ChatCompletionMessage(
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        role="assistant"
    )

    choice = Choice(
        message=message,
        finish_reason="stop" if response.status == "completed" else response.status,
        index=0
    )

    return SimpleNamespace(
        choices=[choice],
        id=response.id,
        created=getattr(response, 'created_at', None),
        model=getattr(response, 'model', None)
    )