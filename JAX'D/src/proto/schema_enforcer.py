from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

F = TypeVar("F", bound=Callable[..., Any])


def enforce_schema(schema: type[BaseModel]) -> Callable[[F], F]:
    """Validate a function's return payload against a pydantic schema.

    The wrapped function still returns a plain ``dict`` so existing call-sites
    remain backward compatible while all outputs are guaranteed schema-valid.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            payload = func(*args, **kwargs)
            model = schema.model_validate(payload)
            return model.model_dump()

        return wrapped  # type: ignore[return-value]

    return decorator
