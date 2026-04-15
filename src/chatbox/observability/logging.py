from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Configure process-wide logging once."""
    if logging.getLogger().handlers:
        return
    level = os.getenv("CHATBOX_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s trace_id=%(trace_id)s %(message)s",
    )


def get_logger(name: str) -> logging.LoggerAdapter:
    base_logger = logging.getLogger(name)
    return logging.LoggerAdapter(base_logger, {"trace_id": "-"})
