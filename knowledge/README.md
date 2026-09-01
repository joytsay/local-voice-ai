# Semiconductor speech-normalization wiki

This directory replaces the monolithic `tsmc.csv` prompt with a wiki-shaped
knowledge base. `system-prompt.md` is always sent to the LLM. The remaining
pages are uploaded to RAGFlow and retrieved according to each transcript.

## Rules

- [System prompt](system-prompt.md)
- [Complete example](rules/complete-example.md)

## Terminology

- [Equipment and lithography](terminology/equipment-and-lithography.md)
- [Wafer handling and factory automation](terminology/material-handling-and-automation.md)
- [Processes and inspection](terminology/process-and-inspection.md)
- [Software](terminology/software.md)
- [Common terms and locations](terminology/common-terms.md)
- [Alarms and yield](terminology/alarms-and-yield.md)
- [Standard keywords](terminology/standard-keywords.md)

Each terminology page is intentionally small enough to form a focused RAGFlow
document while retaining headings that describe its place in the hierarchy.
