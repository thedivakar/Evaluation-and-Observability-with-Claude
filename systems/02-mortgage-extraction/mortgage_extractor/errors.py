"""Domain exceptions for the mortgage extraction pipeline.

Network and SDK errors bubble up unchanged from the ``anthropic`` package.
Domain errors — exceptions raised because the *content* of an input or response
cannot be handled by this pipeline — are defined here.
"""
from __future__ import annotations


class MortgageExtractorError(Exception):
    """Base class for all domain errors raised by this package."""


class UnsupportedDocumentTypeError(MortgageExtractorError):
    """Raised when the classifier returns ``other`` (or an unknown type).

    The classifier's stated reason is preserved on the exception so the caller
    can surface it (in logs, in human-review queues, etc.) without having to
    re-parse the underlying response.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Document type is unsupported: {reason}")
        self.reason = reason


class ExtractionError(MortgageExtractorError):
    """Raised when the extractor's response cannot be parsed into a record.

    Covers cases like: response contains no ``tool_use`` block, an unexpected
    tool was called, or the tool input fails dataclass coercion.
    """


class FlaggedForReviewError(MortgageExtractorError):
    """Raised when the extractor chooses to flag the document for review.

    The model may decline to extract by calling the ``flag_for_review`` tool —
    e.g., when the document is too damaged to read confidently. The reason it
    gave is preserved on the exception.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Extractor flagged document for human review: {reason}")
        self.reason = reason
