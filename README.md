# Research Paper Summarizer (GenAI CLI)

A command-line GenAI app that reads a research paper PDF and generates a
structured summary: **summary, key findings, methodology, limitations,
and keywords** — then lets you ask follow-up questions about the paper.

Built with LangChain + a local Ollama model, using the same patterns from
the course notes: `SystemMessage`/`HumanMessage`/`AIMessage`, chat history
for memory, and `TypedDict` + `with_structured_output()` for structured
output.

## 1. Setup

**Install Ollama** (if you haven't already): https://ollama.com

Pull a model (the code defaults to `qwen2.5:3b`, same as your notes):

```bash
ollama pull qwen2.5:3b
```

**Install Python dependencies:**

```bash
pip install -r requirements.txt
```

## 2. Usage

```bash
python main.py path/to/paper.pdf
```

Or run without arguments and it will prompt you for a path:

```bash
python main.py
```

Use a different Ollama model:

```bash
python main.py path/to/paper.pdf --model llama3.1:8b
```

## 3. What it does

1. Extracts text from the PDF (`pdf_utils.py`). Very long papers are
   truncated to stay within the model's context window (you'll see a
   warning if this happens).
2. Sends the text to a structured-output model built from the
   `PaperSummary` `TypedDict` (`schema.py`) to extract:
   - `title`
   - `summary`
   - `key_findings`
   - `methodology`
   - `limitations`
   - `keywords`
3. Pretty-prints the result in the terminal.
4. Optionally saves the summary as `<paper>_summary.json` and
   `<paper>_summary.md` next to the PDF.
5. Optionally drops you into a chat loop (with full chat history) so you
   can ask follow-up questions like *"What dataset did they use?"* or
   *"How does this compare to prior work?"* — answered strictly from the
   paper text.

## 4. Project structure

```
paper_summarizer/
├── main.py         # CLI entry point: orchestrates everything
├── schema.py        # TypedDict schema for the structured summary
├── pdf_utils.py      # PDF text extraction + truncation
├── requirements.txt
└── README.md
```

## 5. Notes / possible extensions

- **Scanned PDFs**: this uses `pypdf` text extraction, which won't work on
  image-only/scanned PDFs. Add OCR (e.g. `pytesseract`) if you need that.
- **Very long papers**: currently truncated to ~60k characters. A better
  approach for long papers is a *map-reduce* summary: chunk the text,
  summarize each chunk, then summarize the summaries — a natural next step
  once you're comfortable with the current single-pass version.
- **Multiple papers / batch mode**: loop `run()` over a folder of PDFs to
  summarize a whole collection at once.
- **Tool use**: you could add a `@tool` (as in section 15 of your notes) to
  fetch a paper directly from an arXiv URL instead of requiring a local PDF.
