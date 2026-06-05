"""
Report Generation Engine — Phase 7

=== THEORY ===

The report generator transforms structured agent outputs (from the
SynthesisAgent) into publication-ready documents in multiple formats.

Formats:
  Markdown  — primary output; human-readable, version-controllable
  HTML      — rendered from Markdown with inline CSS
  JSON      — structured data for programmatic consumption

=== ARCHITECTURE ===

  SynthesisAgent output (dict)
    │
    ├── ReportGenerator.to_markdown()
    ├── ReportGenerator.to_html()
    └── ReportGenerator.to_json()

=== COMPLEXITY ===

  to_markdown:  O(S) where S = total text length
  to_html:      O(S) (regex-based Markdown → HTML)
  to_json:      O(S)

=== PRODUCTION EQUIVALENTS ===

  OpenAI Deep Research: Markdown report with citations
  Perplexity Pages:     formatted pages from research results
  Notion AI:            structured document generation
"""

import json
import logging
import re
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML     = "html"
    JSON     = "json"


class ReportGenerator:
    """
    Converts synthesis agent output into formatted reports.

    Usage:
        gen = ReportGenerator()
        md  = gen.generate(synthesis_output, ReportFormat.MARKDOWN)
        html = gen.generate(synthesis_output, ReportFormat.HTML)
    """

    def generate(self, synthesis_output: dict, fmt: ReportFormat = ReportFormat.MARKDOWN) -> str:
        if fmt == ReportFormat.MARKDOWN:
            return self.to_markdown(synthesis_output)
        elif fmt == ReportFormat.HTML:
            return self.to_html(synthesis_output)
        elif fmt == ReportFormat.JSON:
            return self.to_json(synthesis_output)
        raise ValueError(f"Unknown format: {fmt}")

    def to_markdown(self, data: dict) -> str:
        """Generate Markdown report from synthesis output."""
        if "full_report" in data and data["full_report"]:
            return data["full_report"]

        lines: list[str] = []
        goal = data.get("goal", "Research Report")
        lines.append(f"# {goal}")
        lines.append("")

        summary = data.get("summary", "")
        if summary:
            lines.append(f"**Summary:** {summary}")
            lines.append("")

        for section in data.get("sections", []):
            lines.append(f"## {section.get('title', 'Section')}")
            lines.append("")
            lines.append(section.get("content", ""))
            lines.append("")

        quality = data.get("quality_metrics", {})
        if quality:
            lines.append("## Quality Metrics")
            lines.append("")
            for k, v in quality.items():
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    lines.append(f"- **{label}:** {v:.2%}")
                else:
                    lines.append(f"- **{label}:** {v}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Evidence items used: {data.get('evidence_used', 0)}*")
        lines.append(f"*Strategy: {data.get('strategy', 'unknown')}*")

        return "\n".join(lines)

    def to_html(self, data: dict) -> str:
        """Convert to HTML with inline CSS."""
        md = self.to_markdown(data)
        html_body = self._markdown_to_html(md)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{data.get('goal', 'Research Report')}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.6; }}
  h1 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }}
  h2 {{ color: #374151; margin-top: 2rem; }}
  ul {{ padding-left: 1.5rem; }}
  hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 2rem 0; }}
  em {{ color: #6b7280; }}
  strong {{ color: #111827; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    def to_json(self, data: dict) -> str:
        """Return structured JSON report."""
        report = {
            "report_id":      data.get("report_id", ""),
            "goal":           data.get("goal", ""),
            "strategy":       data.get("strategy", ""),
            "summary":        data.get("summary", ""),
            "sections":       data.get("sections", []),
            "evidence_used":  data.get("evidence_used", 0),
            "quality_metrics": data.get("quality_metrics", {}),
        }
        return json.dumps(report, indent=2, default=str)

    @staticmethod
    def _markdown_to_html(md: str) -> str:
        """Lightweight Markdown → HTML conversion."""
        html = md
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'((?:<li>.+</li>\n?)+)', r'<ul>\1</ul>', html)
        html = re.sub(r'^---$', r'<hr>', html, flags=re.MULTILINE)
        html = re.sub(r'\n{2,}', '\n<br>\n', html)
        return html
