"""
고수준 추론 로그(Deep Reasoning Log) 표준 포맷.

Agent 노드가 PLANNING / EVALUATING(ToT) / LLM CALL / MCP CALL / SELF-CORRECTION 5단계를
동일한 포맷으로 로거와 프론트엔드 스트림(SSE)에 동시에 내보낼 수 있도록 돕는다.

LLM CALL은 실제로 어떤 모델이, 어떤 목적으로, 어떤 입력을 가지고 호출되었고
그 결과 무엇을 만들어냈는지를 구체적으로 남겨 "AI가 실제로 추론하고 있음"을
사용자가 로그만 보고도 확인할 수 있게 하는 데 목적이 있다.

이벤트 공통 shape: {"type": "log", "agent": str, "phase": str, "message": str, "level"?: "error"}
"""
import json
from typing import Any, Callable, Optional

# 조달청 비축공고서 1건을 사람이 직접 검토/입력할 때 걸리는 평균 시간(초) 추정치.
# "단축 시간" 성과 지표 계산의 기준선으로 사용한다.
MANUAL_BASELINE_SEC = 600


def _truncate(text: Optional[str], limit: int = 160) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def planning_event(agent: str, steps: list[dict]) -> tuple[dict, str]:
    lines = [
        "[🧠 PLANNING] Initializing Deep Reasoning Engine...",
        f"[🧠 PLANNING] Generated a {len(steps)}-step execution plan:",
    ]
    for i, step in enumerate(steps):
        connector = "└──" if i == len(steps) - 1 else "├──"
        tool_part = f" (Tool: {step['tool']})" if step.get("tool") else ""
        lines.append(f" {connector} Step {i + 1}: {step['label']}{tool_part}")
    text = "\n".join(lines)
    return {"type": "log", "agent": agent, "phase": "planning", "message": text}, text


def evaluating_event(agent: str, paths: list[dict], selected: str) -> tuple[dict, str]:
    lines = ["[🧐 EVALUATING] Running Tree of Thoughts (ToT) simulation..."]
    for i, path in enumerate(paths):
        connector = "└──" if i == len(paths) - 1 else "├──"
        label = chr(65 + i)
        lines.append(
            f' {connector} 💡 Thought Path {label}: "{path["name"]}" -> [Score: {path["score"]}%] ({path["reason"]})'
        )
    lines.append(f"[🧐 EVALUATING] Selected Best Path: {selected}")
    text = "\n".join(lines)
    return {"type": "log", "agent": agent, "phase": "evaluating", "message": text}, text


def llm_call_event(
    agent: str,
    model: str,
    purpose: str,
    prompt_preview: Optional[str] = None,
    params: Optional[dict] = None,
) -> tuple[dict, str]:
    """실제 LLM 호출 직전에 남기는 로그. 어떤 모델이 무슨 목적으로, 어떤 입력을 가지고
    호출되는지를 구체적으로 보여줘 "지금 AI가 추론 중"임을 드러낸다."""
    params_part = f" params={json.dumps(params, ensure_ascii=False)}" if params else ""
    lines = [f"[🤖 LLM CALL] Invoking model '{model}' — {purpose}{params_part}"]
    if prompt_preview:
        lines.append(f' [🤖 LLM CALL] Prompt/Input: "{_truncate(prompt_preview, 220)}"')
    text = "\n".join(lines)
    return {"type": "log", "agent": agent, "phase": "llm_call", "message": text}, text


def llm_response_event(agent: str, model: str, summary: str, output_preview: Optional[str] = None) -> tuple[dict, str]:
    """LLM 호출 직후 실제 응답(또는 응답에서 뽑아낸 핵심 결과)을 구체적으로 남기는 로그."""
    lines = [f"[✅ LLM RESPONSE] ({model}) {summary}"]
    if output_preview:
        lines.append(f' [✅ LLM RESPONSE] Output: "{_truncate(output_preview, 220)}"')
    text = "\n".join(lines)
    return {"type": "log", "agent": agent, "phase": "llm_call", "message": text}, text


def mcp_call_events(agent: str, context_name: str, tool_name: str, args: dict) -> list[tuple[dict, str]]:
    text_connect = f"[🔌 MCP CALL] Connecting to [{context_name}]"
    text_call = f"[🔌 MCP CALL] Calling Tool: '{tool_name}' with args: {json.dumps(args, ensure_ascii=False)}"
    return [
        ({"type": "log", "agent": agent, "phase": "mcp_call", "message": text_connect}, text_connect),
        ({"type": "log", "agent": agent, "phase": "mcp_call", "message": text_call}, text_call),
    ]


def correction_events(agent: str, error: str, cause: str, plan: str) -> list[tuple[dict, str, str]]:
    items = [
        (f"[⚠️ ERROR DETECTED] {error}", "error"),
        ("[🔄 SELF-CORRECTION] Critical error captured by Agent Monitor.", "info"),
        (f"[🔄 SELF-CORRECTION] Analysing error cause... {cause}", "info"),
        (f"[🔄 SELF-CORRECTION] Formulating correction plan: {plan}", "info"),
    ]
    return [
        ({"type": "log", "agent": agent, "phase": "correction", "level": level, "message": text}, text, level)
        for text, level in items
    ]


def retrying_event(agent: str) -> tuple[dict, str]:
    text = "[🛠️ EXECUTION] Re-executing corrected logic..."
    return {"type": "log", "agent": agent, "phase": "execution", "message": text}, text


class ReasoningLog:
    """LangGraph 노드 실행 컨텍스트 안에서 사용하는 표준 로그 래퍼.

    writer는 `langgraph.config.get_stream_writer()`의 반환값을 전달한다.
    노드 실행 컨텍스트 밖(그래프 진입 전)에서는 writer 없이 위 빌더 함수를
    직접 호출해 yield하는 방식을 사용해야 한다 (get_stream_writer는 그래프
    실행 컨텍스트 밖에서는 이벤트를 전달하지 못한다).
    """

    def __init__(self, logger, writer: Optional[Callable[[Any], None]], agent: str):
        self.logger = logger
        self.writer = writer
        self.agent = agent

    def _emit(self, event: dict, text: str, level: str = "info") -> None:
        getattr(self.logger, level)(f"[{self.agent}] {text}")
        if self.writer:
            self.writer(event)

    def planning(self, steps: list[dict]) -> None:
        event, text = planning_event(self.agent, steps)
        self._emit(event, text)

    def evaluating(self, paths: list[dict], selected: str) -> None:
        event, text = evaluating_event(self.agent, paths, selected)
        self._emit(event, text)

    def llm_call(self, model: str, purpose: str, prompt_preview: Optional[str] = None, params: Optional[dict] = None) -> None:
        event, text = llm_call_event(self.agent, model, purpose, prompt_preview, params)
        self._emit(event, text)

    def llm_response(self, model: str, summary: str, output_preview: Optional[str] = None) -> None:
        event, text = llm_response_event(self.agent, model, summary, output_preview)
        self._emit(event, text)

    def mcp_call(self, context_name: str, tool_name: str, args: dict) -> None:
        for event, text in mcp_call_events(self.agent, context_name, tool_name, args):
            self._emit(event, text)

    def correction(self, error: str, cause: str, plan: str) -> None:
        for event, text, level in correction_events(self.agent, error, cause, plan):
            self._emit(event, text, level)

    def retrying(self) -> None:
        event, text = retrying_event(self.agent)
        self._emit(event, text)


def build_notice_summary(final_state: dict, elapsed_sec: float) -> dict:
    """NoticeScanAgent 실행 결과로부터 성과 요약(성공률/처리건수/단축시간)을 집계한다."""
    extracted = final_state.get("extracted_data") or {}
    general = extracted.get("general") or {}
    execution = extracted.get("execution") or {}
    items = extracted.get("items") or []

    field_values = list(general.values()) + list(execution.values())
    total_fields = len(field_values)
    ok_count = sum(1 for v in field_values if v not in (None, ""))
    success_rate = round((ok_count / total_fields) * 100, 1) if total_fields else 0.0

    processed_count = total_fields + len(items)
    time_saved_sec = max(0.0, round(MANUAL_BASELINE_SEC - elapsed_sec, 1))

    return {
        "successRate": success_rate,
        "processedCount": processed_count,
        "elapsedSec": round(elapsed_sec, 1),
        "timeSavedSec": time_saved_sec,
    }
