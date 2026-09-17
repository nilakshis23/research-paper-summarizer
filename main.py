"""
main.py
-------
CLI GenAI app: Research Paper Summarizer.

Flow:
  1. User gives a path to a PDF research paper.
  2. We extract the text (pdf_utils.py).
  3. We call a structured-output LLM (schema.py -> PaperSummary) to get
     summary / key findings / methodology / limitations / keywords.
  4. We pretty-print the result and optionally save it.
  5. We drop into a chat loop (using chat history, like the notes' example)
     so the user can ask follow-up questions about the paper.

Built on the same LangChain + Ollama patterns as the course notes:
  - HumanMessage / AIMessage / SystemMessage
  - chat_history for memory
  - TypedDict + with_structured_output() for structured output
"""

import os
import sys
import json
import argparse

from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from schema import PaperSummary
from pdf_utils import prepare_paper_text

console = Console()

DEFAULT_MODEL = "qwen2.5:3b"

SUMMARIZER_SYSTEM_PROMPT = """
You are an expert research assistant that reads academic papers and
extracts an accurate, faithful structured summary.

Rules:
- Base everything ONLY on the paper text provided. Do not invent facts
  that are not supported by the text.
- If a field genuinely cannot be determined from the text, say so
  explicitly (e.g. "Not stated in the provided text") rather than
  guessing wildly.
- Be concise and technical where appropriate; this is for a researcher,
  not a general audience.
"""

CHAT_SYSTEM_PROMPT_TEMPLATE = """
You are a research assistant helping the user understand a specific paper.
Answer questions ONLY using the paper text below. If the answer isn't in
the text, say you can't find it in the paper rather than guessing.

--- PAPER TEXT START ---
{paper_text}
--- PAPER TEXT END ---
"""


def build_model(model_name: str) -> ChatOllama:
    return ChatOllama(model=model_name)


def summarize_paper(model: ChatOllama, paper_text: str) -> PaperSummary:
    """Run the paper text through a structured-output model."""
    structured_model = model.with_structured_output(PaperSummary)

    messages = [
        SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Here is the research paper text:\n\n{paper_text}"
        ),
    ]

    return structured_model.invoke(messages)


def print_summary(summary: PaperSummary) -> None:
    console.rule("[bold cyan]Paper Summary")

    console.print(Panel(summary.get("title", "Unknown"), title="Title", style="bold"))

    console.print(Panel(summary.get("summary", ""), title="Summary"))

    findings_md = "\n".join(f"- {f}" for f in summary.get("key_findings", []))
    console.print(Panel(Markdown(findings_md or "None"), title="Key Findings"))

    console.print(Panel(summary.get("methodology", ""), title="Methodology"))

    limitations_md = "\n".join(f"- {l}" for l in summary.get("limitations", []))
    console.print(Panel(Markdown(limitations_md or "None"), title="Limitations"))

    keywords = ", ".join(summary.get("keywords", []))
    console.print(Panel(keywords or "None", title="Keywords", style="italic"))


def save_summary(summary: PaperSummary, pdf_path: str) -> str:
    """Save the summary next to the PDF as a .json and a .md file."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.dirname(os.path.abspath(pdf_path)) or "."

    json_path = os.path.join(out_dir, f"{base}_summary.json")
    md_path = os.path.join(out_dir, f"{base}_summary.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_lines = [
        f"# {summary.get('title', 'Unknown')}",
        "",
        "## Summary",
        summary.get("summary", ""),
        "",
        "## Key Findings",
        *[f"- {item}" for item in summary.get("key_findings", [])],
        "",
        "## Methodology",
        summary.get("methodology", ""),
        "",
        "## Limitations",
        *[f"- {item}" for item in summary.get("limitations", [])],
        "",
        "## Keywords",
        ", ".join(summary.get("keywords", [])),
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return md_path


def chat_about_paper(model: ChatOllama, paper_text: str) -> None:
    """Follow-up Q&A loop with chat history, same pattern as the notes."""
    console.rule("[bold cyan]Ask follow-up questions about the paper")
    console.print("[dim]Type 'exit' to quit.[/dim]\n")

    system_message = SystemMessage(
        content=CHAT_SYSTEM_PROMPT_TEMPLATE.format(paper_text=paper_text)
    )
    chat_history = [system_message]

    while True:
        query = console.input("[bold green]You:[/bold green] ")

        if query.strip().lower() in ("exit", "quit"):
            break

        chat_history.append(HumanMessage(content=query))

        console.print("[bold blue]AI:[/bold blue] ", end="")
        ai_message = ""
        for chunk in model.stream(chat_history):
            ai_message += chunk.content
            console.print(chunk.content, end="")
        console.print()

        chat_history.append(AIMessage(content=ai_message))


def run(pdf_path: str, model_name: str) -> None:
    if not os.path.isfile(pdf_path):
        console.print(f"[red]File not found:[/red] {pdf_path}")
        sys.exit(1)

    console.print(f"[dim]Reading {pdf_path} ...[/dim]")
    paper_text, was_truncated = prepare_paper_text(pdf_path)

    if not paper_text:
        console.print("[red]Could not extract any text from this PDF "
                       "(it may be a scanned/image-only PDF).[/red]")
        sys.exit(1)

    if was_truncated:
        console.print(
            "[yellow]Note:[/yellow] paper is long; text was truncated to fit "
            "the model's context window. Summary is based on the truncated text."
        )

    console.print(f"[dim]Using model: {model_name}[/dim]")
    model = build_model(model_name)

    console.print("[dim]Generating structured summary...[/dim]")
    summary = summarize_paper(model, paper_text)

    print_summary(summary)

    save = console.input("\n[bold]Save summary to .md/.json? (y/n): [/bold]")
    if save.strip().lower() == "y":
        path = save_summary(summary, pdf_path)
        console.print(f"[green]Saved to {path}[/green]")

    chat = console.input("\n[bold]Ask follow-up questions about this paper? (y/n): [/bold]")
    if chat.strip().lower() == "y":
        chat_about_paper(model, paper_text)

    console.print("\n[bold cyan]Done. Goodbye![/bold cyan]")


def main():
    parser = argparse.ArgumentParser(
        description="GenAI CLI: Research Paper Summarizer (LangChain + Ollama)"
    )
    parser.add_argument(
        "pdf_path", nargs="?", help="Path to the research paper PDF"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name to use (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not pdf_path:
        pdf_path = console.input("[bold]Enter path to research paper PDF: [/bold]").strip()

    run(pdf_path, args.model)


if __name__ == "__main__":
    main()
