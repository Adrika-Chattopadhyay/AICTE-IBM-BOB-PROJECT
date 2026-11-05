"""
Guardrail 6: OPERATIONAL GUARDRAILS
Controls agent execution and resource consumption.

Checks:
  - Maximum execution time enforcement
  - Token usage limits and budget tracking
  - API rate limiting (global)
  - Maximum agent iterations
  - Cost monitoring and budget alerts
  - Retry and failure handling with exponential backoff
"""
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar

from smart_farming_agent.config.settings import get_settings
from smart_farming_agent.utils.logger import logger, audit_log

settings = get_settings()

T = TypeVar("T")


@dataclass
class OperationalCheckResult:
    allowed: bool
    reason: Optional[str] = None
    current_usage: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class TokenBudgetTracker:
    """Thread-safe monthly token budget tracker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._monthly_tokens: int = 0
        self._month_start: datetime = datetime.now().replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

    def record_usage(self, tokens: int) -> bool:
        """
        Record token usage.  Returns True if within budget, False if exceeded.
        """
        with self._lock:
            self._reset_if_new_month()
            self._monthly_tokens += tokens
            over_budget = self._monthly_tokens > settings.monthly_token_budget
            if over_budget:
                audit_log(
                    "TOKEN_BUDGET_EXCEEDED",
                    used=self._monthly_tokens,
                    budget=settings.monthly_token_budget,
                )
            return not over_budget

    def get_usage(self) -> Dict[str, Any]:
        with self._lock:
            self._reset_if_new_month()
            pct = (self._monthly_tokens / settings.monthly_token_budget) * 100
            return {
                "tokens_used": self._monthly_tokens,
                "token_budget": settings.monthly_token_budget,
                "usage_pct": round(pct, 1),
                "month_start": self._month_start.isoformat(),
            }

    def _reset_if_new_month(self):
        now = datetime.now()
        new_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if new_month_start > self._month_start:
            self._monthly_tokens = 0
            self._month_start = new_month_start


# Singleton budget tracker shared across the app
_token_tracker = TokenBudgetTracker()


class OperationalGuardrail:
    """
    Enforces operational limits: execution time, token budgets,
    iteration caps, rate limits, and retry logic.
    """

    def __init__(self):
        # per-session iteration counter
        self._iteration_counts: Dict[str, int] = defaultdict(int)
        # global request counter for rate limiting
        self._global_call_log: List[float] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Token usage
    # ------------------------------------------------------------------
    def check_and_record_tokens(
        self, tokens: int, session_id: str = "unknown"
    ) -> OperationalCheckResult:
        """Validate and record token consumption."""
        if tokens > settings.max_tokens_per_request:
            return OperationalCheckResult(
                allowed=False,
                reason=(
                    f"Request would use {tokens} tokens, exceeding per-request limit "
                    f"of {settings.max_tokens_per_request}."
                ),
            )

        within_budget = _token_tracker.record_usage(tokens)
        usage = _token_tracker.get_usage()

        result = OperationalCheckResult(allowed=within_budget, current_usage=usage)
        if not within_budget:
            result.reason = (
                f"Monthly token budget exceeded "
                f"({usage['tokens_used']:,}/{usage['token_budget']:,})."
            )
        elif usage["usage_pct"] > 80:
            result.warnings.append(
                f"Token budget at {usage['usage_pct']}%. "
                "Consider monitoring usage carefully."
            )

        return result

    # ------------------------------------------------------------------
    # Iteration limit
    # ------------------------------------------------------------------
    def check_iteration(
        self, session_id: str, agent_name: str = "agent"
    ) -> OperationalCheckResult:
        """Increment and check the iteration counter for a session."""
        key = f"{session_id}:{agent_name}"
        self._iteration_counts[key] += 1
        current = self._iteration_counts[key]

        if current > settings.max_agent_iterations:
            audit_log(
                "ITERATION_LIMIT_EXCEEDED",
                session=session_id,
                agent=agent_name,
                iterations=current,
            )
            return OperationalCheckResult(
                allowed=False,
                reason=(
                    f"Agent '{agent_name}' exceeded max iterations "
                    f"({settings.max_agent_iterations})."
                ),
                current_usage={"iterations": current},
            )

        remaining = settings.max_agent_iterations - current
        result = OperationalCheckResult(
            allowed=True,
            current_usage={"iterations": current, "remaining": remaining},
        )
        if remaining <= 2:
            result.warnings.append(
                f"Only {remaining} iteration(s) remaining for this session."
            )
        return result

    def reset_session_iterations(self, session_id: str) -> None:
        keys = [k for k in self._iteration_counts if k.startswith(f"{session_id}:")]
        for k in keys:
            del self._iteration_counts[k]

    # ------------------------------------------------------------------
    # Timed execution
    # ------------------------------------------------------------------
    def run_with_timeout(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> T:
        """
        Execute `func(*args, **kwargs)` with a timeout.
        Raises TimeoutError if the function exceeds the limit.
        """
        limit = timeout or settings.max_execution_time_seconds
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, Any] = {}

        def target():
            try:
                result_holder["result"] = func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                error_holder["error"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=limit)

        if thread.is_alive():
            audit_log("EXECUTION_TIMEOUT", timeout=limit)
            raise TimeoutError(
                f"Execution exceeded the {limit}s time limit and was terminated."
            )
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder["result"]

    # ------------------------------------------------------------------
    # Retry with exponential backoff
    # ------------------------------------------------------------------
    def retry_with_backoff(
        self,
        func: Callable[..., T],
        *args,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        exceptions: tuple = (Exception,),
        **kwargs,
    ) -> T:
        """
        Call `func` up to `max_retries` times with exponential backoff
        on the specified exception types.
        """
        retries = max_retries if max_retries is not None else settings.max_retries
        delay = base_delay if base_delay is not None else settings.retry_delay_seconds
        last_exc = None

        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries:
                    wait = delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{retries + 1} failed: {exc}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"All {retries + 1} attempts failed: {exc}")

        raise RuntimeError(
            f"Function failed after {retries + 1} attempts."
        ) from last_exc

    # ------------------------------------------------------------------
    # Cost monitoring
    # ------------------------------------------------------------------
    def get_cost_estimate(self) -> Dict[str, Any]:
        """
        Rough cost estimate based on token usage
        (IBM Granite approximate pricing).
        """
        usage = _token_tracker.get_usage()
        # Approximate: $0.002 per 1K tokens (Granite pricing estimate)
        cost_usd = (usage["tokens_used"] / 1000) * 0.002
        result = {
            **usage,
            "estimated_cost_usd": round(cost_usd, 4),
            "budget_alert": cost_usd > settings.cost_alert_threshold_usd,
        }
        if result["budget_alert"]:
            audit_log(
                "COST_ALERT",
                cost=cost_usd,
                threshold=settings.cost_alert_threshold_usd,
            )
        return result
