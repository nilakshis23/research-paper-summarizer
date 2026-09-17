"""
schema.py
---------
Defines the structured-output shape we want the LLM to return
for every research paper, using TypedDict + Annotated
(exactly the pattern from the GenAI notes: section 9-10).
"""

from typing import TypedDict, Annotated, List


class PaperSummary(TypedDict):
    """Structured summary of a research paper."""

    title: Annotated[
        str,
        "The title of the paper. Use 'Unknown' if it cannot be found in the text."
    ]

    summary: Annotated[
        str,
        "A concise 4-6 sentence plain-language summary of what the paper is about, "
        "its motivation, and its overall contribution."
    ]

    key_findings: Annotated[
        List[str],
        "A list of 3-6 bullet points describing the main findings, results, "
        "or contributions reported in the paper."
    ]

    methodology: Annotated[
        str,
        "A short paragraph describing the methodology, approach, dataset, "
        "or experimental setup used by the authors."
    ]

    limitations: Annotated[
        List[str],
        "A list of 2-5 limitations of the paper, either explicitly stated by the "
        "authors or reasonably inferred (e.g. small dataset, narrow scope, "
        "lack of baselines). If none are stated, infer plausible ones and say so."
    ]

    keywords: Annotated[
        List[str],
        "A list of 5-10 important keywords or technical terms that best represent "
        "the paper's topic, for indexing/search purposes."
    ]
