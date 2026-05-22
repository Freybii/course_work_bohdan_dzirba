"""
BidScop PDF Analysis Demo

This file is intentionally simplified for a public/demo repository.
It demonstrates the basic idea of document analysis without exposing
the production matching logic, AI pipeline, semantic models, training code,
OCR pipeline, internal requirement rules, or service integrations.

How to use:
1. Put one or more PDF files into the "input" folder.
2. Install dependency: pip install pymupdf
3. Run: python main.py
4. Check JSON results in the "output" folder.

Optional:
    python main.py --input input --output output --limit 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# =========================
# DEMO SETTINGS
# =========================

MAX_CONTEXT_CHARS = 260
MAX_MATCHES_PER_REQUIREMENT = 5


# =========================
# DATA MODELS
# =========================

@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class DemoRequirement:
    key: str
    name: str
    keywords: List[str]
    patterns: List[str]


@dataclass
class DemoMatch:
    page_number: int
    method: str
    matched_text: str


@dataclass
class DemoResult:
    key: str
    name: str
    status: str
    confidence: float
    pages: List[int] = field(default_factory=list)
    matches: List[DemoMatch] = field(default_factory=list)


# =========================
# PUBLIC DEMO REQUIREMENTS
# =========================
# These are generic examples only.
# The real project can contain a larger and more precise set of rules.

DEMO_REQUIREMENTS: List[DemoRequirement] = [
    DemoRequirement(
        key="BID_SUBMISSION",
        name="Bid Submission Information",
        keywords=[
            "bid submission",
            "sealed bids",
            "bids will be received",
            "submit bids",
            "proposal submission",
        ],
        patterns=[
            r"\b(?:bids?|proposals?)\s+(?:will\s+be\s+)?(?:received|submitted)\b",
            r"\bsealed\s+bids?\b",
        ],
    ),
    DemoRequirement(
        key="PRE_BID_MEETING",
        name="Pre-Bid Meeting / Job Walk",
        keywords=[
            "pre-bid meeting",
            "prebid meeting",
            "pre bid meeting",
            "job walk",
            "site walk",
            "mandatory pre-bid",
            "non-mandatory pre-bid",
        ],
        patterns=[
            r"\b(?:mandatory|non[-\s]?mandatory)?\s*(?:pre[-\s]?bid|job\s+walk|site\s+walk)\b",
        ],
    ),
    DemoRequirement(
        key="ESTIMATED_COST",
        name="Estimated Project Cost",
        keywords=[
            "engineer's estimate",
            "estimated cost",
            "construction estimate",
            "project estimate",
            "estimated construction cost",
        ],
        patterns=[
            r"\b(?:estimated|engineer'?s)\s+(?:construction\s+)?(?:cost|estimate)\b.{0,120}?\$?\s?\d[\d,]*(?:\.\d{2})?",
            r"\$\s?\d[\d,]*(?:\.\d{2})?",
        ],
    ),
    DemoRequirement(
        key="COMPLETION_TIME",
        name="Completion Time / Project Duration",
        keywords=[
            "completion date",
            "time for completion",
            "calendar days",
            "working days",
            "substantial completion",
            "final completion",
        ],
        patterns=[
            r"\b\d{1,4}\s+(?:calendar|working|business)?\s*days?\b",
            r"\b(?:substantial|final)\s+completion\b",
        ],
    ),
    DemoRequirement(
        key="LIQUIDATED_DAMAGES",
        name="Liquidated Damages",
        keywords=[
            "liquidated damages",
            "damages per day",
        ],
        patterns=[
            r"\bliquidated\s+damages\b.{0,120}?\$?\s?\d[\d,]*(?:\.\d{2})?",
        ],
    ),
]


# =========================
# TEXT EXTRACTION
# =========================

def normalize_text(text: str) -> str:
    """Normalize PDF text for simple demo matching."""
    text = text.replace("\u00A0", " ").replace("\xa0", " ")
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u2032", "'")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> List[PageText]:
    """Extract text from a PDF file page by page."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed. Install it with: pip install pymupdf"
        ) from exc

    pages: List[PageText] = []

    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            raw_text = page.get_text("text") or ""
            pages.append(PageText(page_number=index, text=normalize_text(raw_text)))

    return pages


# =========================
# MATCHING
# =========================

def context_snippet(text: str, start: int, end: int, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Return a readable context snippet around a match."""
    half = max_chars // 2
    left = max(0, start - half)
    right = min(len(text), end + half)

    snippet = text[left:right].strip()

    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."

    return snippet


def find_keyword_matches(requirement: DemoRequirement, pages: List[PageText]) -> List[DemoMatch]:
    """Find simple keyword matches."""
    matches: List[DemoMatch] = []

    for page in pages:
        page_lower = page.text.lower()

        for keyword in requirement.keywords:
            keyword_lower = keyword.lower()
            index = page_lower.find(keyword_lower)

            if index != -1:
                matches.append(
                    DemoMatch(
                        page_number=page.page_number,
                        method="keyword",
                        matched_text=context_snippet(page.text, index, index + len(keyword)),
                    )
                )
                break

        if len(matches) >= MAX_MATCHES_PER_REQUIREMENT:
            break

    return matches


def find_regex_matches(requirement: DemoRequirement, pages: List[PageText]) -> List[DemoMatch]:
    """Find basic regex matches."""
    matches: List[DemoMatch] = []
    seen_pages: set[int] = set()

    for pattern in requirement.patterns:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            logger.warning("Invalid demo regex skipped for %s", requirement.key)
            continue

        for page in pages:
            if page.page_number in seen_pages:
                continue

            match = regex.search(page.text)
            if match:
                matches.append(
                    DemoMatch(
                        page_number=page.page_number,
                        method="regex",
                        matched_text=context_snippet(page.text, match.start(), match.end()),
                    )
                )
                seen_pages.add(page.page_number)

            if len(matches) >= MAX_MATCHES_PER_REQUIREMENT:
                return matches

    return matches


def analyze_pages(pages: List[PageText]) -> List[DemoResult]:
    """Analyze pages using only public demo keyword and regex rules."""
    results: List[DemoResult] = []

    for requirement in DEMO_REQUIREMENTS:
        keyword_matches = find_keyword_matches(requirement, pages)
        regex_matches = find_regex_matches(requirement, pages)

        combined: List[DemoMatch] = []
        seen = set()

        for match in keyword_matches + regex_matches:
            key = (match.page_number, match.method, match.matched_text)
            if key not in seen:
                combined.append(match)
                seen.add(key)

        pages_found = sorted({match.page_number for match in combined})

        if combined:
            result = DemoResult(
                key=requirement.key,
                name=requirement.name,
                status="found",
                confidence=0.75 if any(m.method == "regex" for m in combined) else 0.60,
                pages=pages_found,
                matches=combined,
            )
        else:
            result = DemoResult(
                key=requirement.key,
                name=requirement.name,
                status="not_found",
                confidence=0.0,
            )

        results.append(result)

    return results


# =========================
# OUTPUT
# =========================

def build_output(pdf_path: Path, pages: List[PageText], results: List[DemoResult], elapsed_sec: float) -> Dict:
    """Build JSON-ready output."""
    found_count = sum(1 for result in results if result.status == "found")

    return {
        "demo_notice": (
            "This is a simplified public demo. Production AI analysis, semantic search, "
            "OCR, internal rules, scoring, training data collection, and integrations "
            "are intentionally not included."
        ),
        "source_file": pdf_path.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "elapsed_sec": round(elapsed_sec, 3),
        "summary": {
            "requirements_checked": len(results),
            "found": found_count,
            "not_found": len(results) - found_count,
        },
        "results": [asdict(result) for result in results],
    }


def save_json(data: Dict, output_path: Path) -> None:
    """Save JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    logger.info("Saved result: %s", output_path)


def print_summary(pdf_name: str, results: List[DemoResult], elapsed_sec: float) -> None:
    """Print short console summary."""
    print("\n" + "=" * 80)
    print(f"DEMO ANALYSIS: {pdf_name}")
    print("=" * 80)

    for result in results:
        icon = "✓" if result.status == "found" else "—"
        pages = ", ".join(str(page) for page in result.pages) if result.pages else "-"
        print(f"{icon} {result.key:<20} {result.status:<10} pages: {pages}")

    print("-" * 80)
    print(f"Elapsed: {elapsed_sec:.2f} sec")
    print("=" * 80)


# =========================
# MAIN
# =========================

def analyze_pdf_file(pdf_path: Path, output_folder: Path) -> None:
    """Analyze one PDF file and save a JSON result."""
    start = time.perf_counter()

    pages = extract_text_from_pdf(pdf_path)
    results = analyze_pages(pages)

    elapsed = time.perf_counter() - start

    output_data = build_output(pdf_path, pages, results, elapsed)
    output_path = output_folder / f"{pdf_path.stem}_demo_result.json"

    save_json(output_data, output_path)
    print_summary(pdf_path.name, results, elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BidScop simplified PDF analysis demo")
    parser.add_argument(
        "--input",
        default="input",
        help="Folder with PDF files. Default: input",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder for JSON results. Default: output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of PDF files to analyze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_folder = Path(args.input)
    output_folder = Path(args.output)

    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_folder.glob("*.pdf"))

    if args.limit is not None:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        logger.warning("No PDF files found in '%s'. Add PDFs and run again.", input_folder)
        logger.info("Example: python main.py --input input --output output")
        return

    logger.info("Found %s PDF file(s)", len(pdf_files))

    for pdf_path in pdf_files:
        try:
            analyze_pdf_file(pdf_path, output_folder)
        except Exception as exc:
            logger.error("Failed to analyze %s: %s", pdf_path.name, exc)

    logger.info("Demo analysis complete")


if __name__ == "__main__":
    main()
