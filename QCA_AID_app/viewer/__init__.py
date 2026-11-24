"""
Viewer Package
==============
Vorschau-Komponenten fuer QCA-AID (TextViewer, Annotation-Overlay, CodingPanel).

Module:
  - pdf_viewer_component: PDF.js-Viewer (HTML-Generator + Streamlit-Wrapper)
"""

from .pdf_viewer_component import (
    ViewerConfig,
    build_viewer_html,
    render_pdf_viewer,
    pdf_to_data_uri,
)
from .viewer_bridge import (
    ViewerEvent,
    ViewerState,
    apply_event,
    apply_events,
    build_command,
    goto_page_command,
)
from .coding_panel import (
    CodingRow,
    build_coding_rows,
    panel_summary,
    panel_regions,
    panel_header,
    apply_action,
    render_coding_panel,
)
from .code_options import (
    CodeOption,
    build_code_options,
    code_label,
    find_option,
    option_index,
    parse_code_text,
    recode,
)

__all__ = [
    "ViewerConfig",
    "build_viewer_html",
    "render_pdf_viewer",
    "pdf_to_data_uri",
    "ViewerEvent",
    "ViewerState",
    "apply_event",
    "apply_events",
    "build_command",
    "goto_page_command",
    "CodingRow",
    "build_coding_rows",
    "panel_summary",
    "panel_regions",
    "panel_header",
    "apply_action",
    "render_coding_panel",
    "CodeOption",
    "build_code_options",
    "code_label",
    "find_option",
    "option_index",
    "parse_code_text",
    "recode",
]