"""Two-pass extraction pipeline: classify, then extract.

The classification pass uses ``tool_choice={"type":"tool","name":...}`` — the
model is forced to call the classifier and cannot return free text. The
extraction pass uses ``tool_choice={"type":"any"}`` against a small set of
doc-type-specific tools (a primary extractor plus a ``flag_for_review`` escape
hatch). ``"any"`` rather than ``"auto"`` because ``"auto"`` would permit a
conversational text fallback; ``"any"`` guarantees a ``tool_use`` block.

This pattern mirrors the Architect's Playbook "broad-then-pinpoint" exploration
recipe: classify the document first, then drill into the doc-type-specific
extractor with a fallback for unrecoverable cases.
"""
from __future__ import annotations

import logging

from anthropic.types import Message, ToolUseBlock

from mortgage_extractor import prompts
from mortgage_extractor.client import RecordingClient
from mortgage_extractor.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from mortgage_extractor.errors import (
    ExtractionError,
    FlaggedForReviewError,
    UnsupportedDocumentTypeError,
)
from mortgage_extractor.models import (
    Classification,
    DocumentType,
    MortgageExtraction,
)
from mortgage_extractor.tools import (
    ToolDefinition,
    classify_document,
    doc_type_extractor,
    flag_for_review,
)

log = logging.getLogger(__name__)


class Pipeline:
    """Two-pass classifier + extractor over the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        client: RecordingClient | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.client = client or RecordingClient()
        self.max_tokens = max_tokens

    def run(self, document_text: str) -> MortgageExtraction:
        classification = self.classify_document(document_text)
        if classification.document_type is DocumentType.OTHER:
            raise UnsupportedDocumentTypeError(classification.reason)
        return self.extract(document_text, classification.document_type)

    def classify_document(self, document_text: str) -> Classification:
        classifier = classify_document()
        response = self.client.call(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prompts.classifier_system_prompt(),
            tools=[classifier],
            tool_choice={"type": "tool", "name": classifier["name"]},
            messages=[{"role": "user", "content": document_text}],
        )
        block = _single_tool_use_block(response, expected_name=classifier["name"])
        log.info(
            "classify: model=%s in=%d out=%d type=%s",
            self.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            block.input.get("document_type") if isinstance(block.input, dict) else "?",
        )
        return Classification.model_validate(block.input)

    def extract(
        self,
        document_text: str,
        doc_type: DocumentType,
    ) -> MortgageExtraction:
        if doc_type is DocumentType.OTHER:
            raise UnsupportedDocumentTypeError(
                "extract() called with DocumentType.OTHER; classifier should "
                "have short-circuited earlier."
            )

        tools: list[ToolDefinition] = [
            doc_type_extractor(doc_type),
            flag_for_review(),
        ]
        response = self.client.call(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prompts.extractor_system_prompt(doc_type),
            tools=tools,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": document_text}],
        )

        block = _single_tool_use_block(response)
        log.info(
            "extract: model=%s in=%d out=%d tool=%s",
            self.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            block.name,
        )

        if block.name == "flag_for_review":
            reason = ""
            if isinstance(block.input, dict):
                reason = str(block.input.get("reason", ""))
            raise FlaggedForReviewError(reason)

        expected_extractor = doc_type_extractor(doc_type)["name"]
        if block.name != expected_extractor:
            raise ExtractionError(
                f"Unexpected tool call: {block.name!r} (expected "
                f"{expected_extractor!r} or 'flag_for_review')"
            )

        return MortgageExtraction.model_validate(block.input)


def _single_tool_use_block(
    response: Message,
    *,
    expected_name: str | None = None,
) -> ToolUseBlock:
    tool_uses: list[ToolUseBlock] = [
        block for block in response.content if isinstance(block, ToolUseBlock)
    ]
    if not tool_uses:
        raise ExtractionError(
            "Response contained no tool_use block; got "
            f"{[type(b).__name__ for b in response.content]!r}"
        )
    if expected_name is not None:
        named = [b for b in tool_uses if b.name == expected_name]
        if not named:
            raise ExtractionError(
                f"Expected tool_use for {expected_name!r}, got "
                f"{[b.name for b in tool_uses]!r}"
            )
        return named[0]
    return tool_uses[0]
