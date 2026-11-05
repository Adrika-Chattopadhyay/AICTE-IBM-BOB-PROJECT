"""
Centralized structured logging using Loguru.
All modules import `logger` from here.
"""
import sys
import os
from loguru import logger

# Remove default handler
logger.remove()

# Console handler - INFO and above
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File handler - DEBUG and above (rotating)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/smart_farming.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)

# Security/Audit log - separate file
logger.add(
    "logs/audit.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | AUDIT | {name}:{line} - {message}",
    level="INFO",
    filter=lambda record: "AUDIT" in record["extra"],
    rotation="50 MB",
    retention="90 days",
)


def audit_log(message: str, **kwargs) -> None:
    """Log an audit event with structured metadata."""
    logger.bind(AUDIT=True).info(f"{message} | {kwargs}")


__all__ = ["logger", "audit_log"]
