"""
Configuration Settings - Centralized environment variable management.
Uses pydantic-settings for type-safe config loading.
"""
import os
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables / .env file."""

    # -- IBM Cloud / Granite --------------------------------------------------
    ibm_api_key: str = Field(default="", env="IBM_API_KEY")
    ibm_project_id: str = Field(default="", env="IBM_PROJECT_ID")
    ibm_watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com", env="IBM_WATSONX_URL"
    )
    ibm_granite_model_id: str = Field(
        default="ibm/granite-13b-instruct-v2", env="IBM_GRANITE_MODEL_ID"
    )
    ibm_granite_chat_model_id: str = Field(
        default="ibm/granite-3-8b-instruct", env="IBM_GRANITE_CHAT_MODEL_ID"
    )
    ibm_granite_embedding_model_id: str = Field(
        default="ibm/slate-125m-english-rtrvr", env="IBM_GRANITE_EMBEDDING_MODEL_ID"
    )

    # -- App ------------------------------------------------------------------
    app_name: str = Field(default="Smart Farming AI Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_env: str = Field(default="development", env="APP_ENV")
    app_secret_key: str = Field(default="fallback_secret", env="APP_SECRET_KEY")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # -- Weather API ----------------------------------------------------------
    openweather_api_key: str = Field(default="", env="OPENWEATHER_API_KEY")
    openweather_base_url: str = Field(
        default="https://api.openweathermap.org/data/2.5", env="OPENWEATHER_BASE_URL"
    )
    default_country_code: str = Field(default="IN", env="DEFAULT_COUNTRY_CODE")

    # -- Market Data API ------------------------------------------------------
    agmarknet_api_url: str = Field(
        default="https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
        env="AGMARKNET_API_URL",
    )
    data_gov_api_key: str = Field(default="", env="DATA_GOV_API_KEY")

    # -- Soil API -------------------------------------------------------------
    soilgrids_api_url: str = Field(
        default="https://rest.isric.org/soilgrids/v2.0", env="SOILGRIDS_API_URL"
    )

    # -- Input Guardrails -----------------------------------------------------
    max_input_length: int = Field(default=2000, env="MAX_INPUT_LENGTH")
    min_input_length: int = Field(default=3, env="MIN_INPUT_LENGTH")
    prompt_injection_threshold: float = Field(
        default=0.85, env="PROMPT_INJECTION_THRESHOLD"
    )
    pii_detection_enabled: bool = Field(default=True, env="PII_DETECTION_ENABLED")
    inappropriate_content_threshold: float = Field(
        default=0.80, env="INAPPROPRIATE_CONTENT_THRESHOLD"
    )
    malicious_input_threshold: float = Field(
        default=0.85, env="MALICIOUS_INPUT_THRESHOLD"
    )

    # -- Output Guardrails ----------------------------------------------------
    max_output_length: int = Field(default=5000, env="MAX_OUTPUT_LENGTH")
    hallucination_threshold: float = Field(default=0.75, env="HALLUCINATION_THRESHOLD")
    toxicity_threshold: float = Field(default=0.80, env="TOXICITY_THRESHOLD")
    factual_accuracy_min_score: float = Field(
        default=0.60, env="FACTUAL_ACCURACY_MIN_SCORE"
    )
    sensitive_info_filter_enabled: bool = Field(
        default=True, env="SENSITIVE_INFO_FILTER_ENABLED"
    )

    # -- Tool Execution Guardrails --------------------------------------------
    allowed_tools: str = Field(
        default="weather,soil,crop,market_price,pest_control,scheme_info",
        env="ALLOWED_TOOLS",
    )
    max_tool_calls_per_session: int = Field(
        default=50, env="MAX_TOOL_CALLS_PER_SESSION"
    )
    tool_timeout_seconds: int = Field(default=15, env="TOOL_TIMEOUT_SECONDS")
    api_rate_limit_per_minute: int = Field(default=30, env="API_RATE_LIMIT_PER_MINUTE")

    # -- Security Guardrails --------------------------------------------------
    jailbreak_detection_enabled: bool = Field(
        default=True, env="JAILBREAK_DETECTION_ENABLED"
    )
    sql_injection_prevention: bool = Field(
        default=True, env="SQL_INJECTION_PREVENTION"
    )
    max_failed_attempts: int = Field(default=5, env="MAX_FAILED_ATTEMPTS")
    lockout_duration_minutes: int = Field(default=15, env="LOCKOUT_DURATION_MINUTES")

    # -- Operational Guardrails -----------------------------------------------
    max_execution_time_seconds: int = Field(
        default=60, env="MAX_EXECUTION_TIME_SECONDS"
    )
    max_tokens_per_request: int = Field(default=2048, env="MAX_TOKENS_PER_REQUEST")
    max_agent_iterations: int = Field(default=10, env="MAX_AGENT_ITERATIONS")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay_seconds: float = Field(default=2.0, env="RETRY_DELAY_SECONDS")
    monthly_token_budget: int = Field(default=1_000_000, env="MONTHLY_TOKEN_BUDGET")
    cost_alert_threshold_usd: float = Field(
        default=10.0, env="COST_ALERT_THRESHOLD_USD"
    )

    # -- HITL Guardrails ------------------------------------------------------
    hitl_enabled: bool = Field(default=True, env="HITL_ENABLED")
    financial_transaction_approval: bool = Field(
        default=True, env="FINANCIAL_TRANSACTION_APPROVAL"
    )
    human_escalation_threshold: float = Field(
        default=0.50, env="HUMAN_ESCALATION_THRESHOLD"
    )
    approval_timeout_seconds: int = Field(default=300, env="APPROVAL_TIMEOUT_SECONDS")

    # -- Multi-Agent Guardrails -----------------------------------------------
    max_agent_hops: int = Field(default=5, env="MAX_AGENT_HOPS")
    cross_agent_verification: bool = Field(
        default=True, env="CROSS_AGENT_VERIFICATION"
    )
    max_loop_detection_count: int = Field(default=3, env="MAX_LOOP_DETECTION_COUNT")

    # -- Ethical Guardrails ---------------------------------------------------
    bias_detection_enabled: bool = Field(default=True, env="BIAS_DETECTION_ENABLED")
    audit_trail_enabled: bool = Field(default=True, env="AUDIT_TRAIL_ENABLED")
    explainability_enabled: bool = Field(default=True, env="EXPLAINABILITY_ENABLED")

    # -- Database -------------------------------------------------------------
    database_url: str = Field(
        default="sqlite:///data/smart_farming_audit.db", env="DATABASE_URL"
    )
    audit_log_retention_days: int = Field(
        default=90, env="AUDIT_LOG_RETENTION_DAYS"
    )

    # -- Supported Languages --------------------------------------------------
    supported_languages: str = Field(
        default="en,hi,mr,ta,te,kn,gu,pa,bn,ml", env="SUPPORTED_LANGUAGES"
    )
    default_user_language: str = Field(default="en", env="DEFAULT_USER_LANGUAGE")

    @property
    def allowed_tools_list(self) -> List[str]:
        return [t.strip() for t in self.allowed_tools.split(",")]

    @property
    def supported_languages_list(self) -> List[str]:
        return [lang.strip() for lang in self.supported_languages.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
