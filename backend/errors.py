"""
Production-grade error handling and logging.
All errors logged securely (no customer data) and returned with appropriate HTTP status.
"""

import logging
import traceback
from enum import Enum
from typing import Optional

from fastapi import HTTPException

logger = logging.getLogger("deepsentinel.errors")


class ErrorCode(str, Enum):
    """Standard error codes for API responses."""

    INVALID_REQUEST = "INVALID_REQUEST"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    LLM_ERROR = "LLM_ERROR"
    EMAIL_ERROR = "EMAIL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class APIError(HTTPException):
    """Base API error with logging and safe response."""

    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        user_message: str,
        internal_details: Optional[str] = None,
        exception: Optional[Exception] = None,
    ):
        self.error_code = error_code
        self.user_message = user_message
        self.internal_details = internal_details
        self.exception = exception

        # Log internal details (not sent to user)
        log_message = f"[{error_code}] {user_message}"
        if internal_details:
            log_message += f" | Details: {internal_details}"
        if exception:
            log_message += f" | Exception: {type(exception).__name__}: {str(exception)}"

        if status_code >= 500:
            logger.error(log_message, exc_info=exception)
        else:
            logger.warning(log_message)

        # Safe response sent to user (no internal details)
        detail = {
            "error_code": error_code,
            "message": user_message,
            "type": "API_ERROR",
        }

        super().__init__(status_code=status_code, detail=detail)


def handle_upstream_error(
    modality: str,
    status_code: Optional[int],
    exception: Exception,
    url: Optional[str] = None,
) -> dict:
    """Handle errors from upstream model APIs (M1/M2/M3)."""
    logger.warning(
        f"Upstream API error [{modality}] status={status_code} url={url} error={type(exception).__name__}"
    )

    # Return safe score when upstream fails
    return {
        "score": 0.5,  # Neutral score
        "available": False,
        "error": str(type(exception).__name__),
    }


def validate_input(
    field: str, value, expected_type, min_val=None, max_val=None
) -> None:
    """Validate input and raise APIError if invalid."""
    if not isinstance(value, expected_type):
        raise APIError(
            status_code=422,
            error_code=ErrorCode.INVALID_REQUEST,
            user_message=f"Field '{field}' must be {expected_type.__name__}",
            internal_details=f"Got {type(value).__name__}: {value}",
        )

    if isinstance(value, (int, float)):
        if min_val is not None and value < min_val:
            raise APIError(
                status_code=422,
                error_code=ErrorCode.INVALID_REQUEST,
                user_message=f"Field '{field}' must be >= {min_val}",
                internal_details=f"Got {value}",
            )
        if max_val is not None and value > max_val:
            raise APIError(
                status_code=422,
                error_code=ErrorCode.INVALID_REQUEST,
                user_message=f"Field '{field}' must be <= {max_val}",
                internal_details=f"Got {value}",
            )

    if isinstance(value, str):
        if not value or not value.strip():
            raise APIError(
                status_code=422,
                error_code=ErrorCode.INVALID_REQUEST,
                user_message=f"Field '{field}' cannot be empty",
            )


def safe_log_transaction(transaction_id: str, **kwargs) -> str:
    """Log transaction safely (no sensitive data)."""
    safe_fields = {
        "transaction_id": transaction_id,
        "step": kwargs.get("step"),
        "type": kwargs.get("type"),
        "amount": kwargs.get("amount"),
    }
    return f"Transaction: {safe_fields}"


class ProductionMiddleware:
    """Middleware for production-grade request/response logging."""

    @staticmethod
    async def log_request(request):
        """Log incoming request safely."""
        path = request.url.path
        method = request.method
        client = request.client.host if request.client else "unknown"
        logger.info(f"Request: {method} {path} from {client}")

    @staticmethod
    def format_error_response(error_code: ErrorCode, message: str) -> dict:
        """Format error response safely."""
        return {
            "error_code": error_code,
            "message": message,
            "type": "API_ERROR",
            "timestamp": None,  # Will be added by response middleware
        }
