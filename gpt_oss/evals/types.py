from dataclasses import dataclass, field
from typing import Any, Literal, overload

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]



@dataclass
class SamplerResponse:
    """
    Response from a sampler.
    """
    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]
    spans: list[dict[str, Any]] | None = None

class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """
    
    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def __call__(
        self, 
        message_list: MessageList,
    ) -> SamplerResponse:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations
    metadata: dict[str, Any] | None  # Extra data such as rubric scores or sollen
    single_eval_results: list["SingleEvalResult"] | None = None  # Individual evaluation results


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation
    spans: list[dict[str, Any]] | None = None  # spans from the trace
    example_level_metadata: dict[str, Any] | None = (
        None  # Extra data such as rubric scores or sollen
    )


@dataclass
class Checkpoint:
    """
    Checkpoint state for resuming evaluations.
    Maps example index -> serialized SingleEvalResult.
    """
    results: dict[int, dict[str, Any]]  # index -> serialized result


class Eval:
    """
    Base class for defining an evaluation.
    """
    
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError

