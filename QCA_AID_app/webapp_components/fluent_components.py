#!/usr/bin/env python3
"""
Fluent UI Komponenten für QCA-AID Webapp
========================================
Wiederverwendbare UI-Komponenten im Microsoft Fluent Design Style.
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from .fluent_styles import FluentColors, FluentSpacing, FluentBorders


def fluent_card(
    title: Optional[str] = None,
    content: Optional[str] = None,
    icon: Optional[str] = None,
    elevated: bool = False
) -> None:
    """
    Rendert eine Fluent UI Card.
    
    Args:
        title: Titel der Card
        content: Inhalt der Card
        icon: Optional Icon (Emoji)
        elevated: Ob die Card einen Schatten haben soll
    """
    shadow = "box-shadow: 0 2px 4px rgba(0,0,0,0.1);" if elevated else ""
    
    card_html = f"""
    <div style="
        background-color: {FluentColors.NEUTRAL_BACKGROUND};
        border: 1px solid {FluentColors.NEUTRAL_STROKE};
        border-radius: {FluentBorders.RADIUS_MEDIUM};
        padding: {FluentSpacing.M};
        margin: {FluentSpacing.M} 0;
        {shadow}
    ">
    """
    
    if title:
        icon_html = f"<span style='margin-right: {FluentSpacing.XS};'>{icon}</span>" if icon else ""
        card_html += f"""
        <h3 style="
            margin: 0 0 {FluentSpacing.S} 0;
            color: {FluentColors.NEUTRAL_FOREGROUND};
            font-weight: 600;
        ">
            {icon_html}{title}
        </h3>
        """
    
    if content:
        card_html += f"""
        <div style="
            color: {FluentColors.NEUTRAL_FOREGROUND_SECONDARY};
            font-size: 14px;
            line-height: 20px;
        ">
            {content}
        </div>
        """
    
    card_html += "</div>"
    
    st.markdown(card_html, unsafe_allow_html=True)


def fluent_section_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None
) -> None:
    """
    Rendert einen Fluent UI Section Header.
    
    Args:
        title: Titel der Section
        subtitle: Optional Untertitel
        icon: Optional Icon (Emoji)
    """
    icon_html = f"<span style='margin-right: {FluentSpacing.S};'>{icon}</span>" if icon else ""
    
    header_html = f"""
    <div style="margin: {FluentSpacing.XL} 0 {FluentSpacing.M} 0;">
        <h2 style="
            margin: 0;
            color: {FluentColors.NEUTRAL_FOREGROUND};
            font-size: 24px;
            font-weight: 600;
            line-height: 32px;
        ">
            {icon_html}{title}
        </h2>
    """
    
    if subtitle:
        header_html += f"""
        <p style="
            margin: {FluentSpacing.XXS} 0 0 0;
            color: {FluentColors.NEUTRAL_FOREGROUND_SECONDARY};
            font-size: 14px;
            line-height: 20px;
        ">
            {subtitle}
        </p>
        """
    
    header_html += "</div>"
    
    st.markdown(header_html, unsafe_allow_html=True)


def fluent_status_badge(
    text: str,
    status: str = "neutral"  # neutral, success, warning, error, info
) -> str:
    """
    Erstellt ein Fluent UI Status Badge (als HTML String).
    
    Args:
        text: Badge Text
        status: Status-Typ (neutral, success, warning, error, info)
    
    Returns:
        HTML String für das Badge
    """
    color_map = {
        "neutral": (FluentColors.NEUTRAL_FOREGROUND_SECONDARY, FluentColors.NEUTRAL_BACKGROUND_TERTIARY),
        "success": (FluentColors.SUCCESS, "#F1FAF1"),
        "warning": (FluentColors.WARNING, "#FFF9F5"),
        "error": (FluentColors.ERROR, "#FDF3F4"),
        "info": (FluentColors.INFO, "#F3F9FD"),
    }
    
    text_color, bg_color = color_map.get(status, color_map["neutral"])
    
    return f"""
    <span style="
        display: inline-block;
        padding: {FluentSpacing.XXS} {FluentSpacing.XS};
        background-color: {bg_color};
        color: {text_color};
        border-radius: {FluentBorders.RADIUS_MEDIUM};
        font-size: 12px;
        font-weight: 600;
        line-height: 16px;
    ">
        {text}
    </span>
    """


def fluent_divider(spacing: str = "M") -> None:
    """
    Rendert einen Fluent UI Divider.
    
    Args:
        spacing: Spacing-Größe (S, M, L, XL)
    """
    spacing_map = {
        "S": FluentSpacing.S,
        "M": FluentSpacing.M,
        "L": FluentSpacing.L,
        "XL": FluentSpacing.XL,
    }
    
    margin = spacing_map.get(spacing, FluentSpacing.M)
    
    st.markdown(f"""
    <hr style="
        border: none;
        border-top: 1px solid {FluentColors.NEUTRAL_STROKE};
        margin: {margin} 0;
    ">
    """, unsafe_allow_html=True)


def fluent_info_box(
    message: str,
    box_type: str = "info",  # info, success, warning, error
    icon: Optional[str] = None
) -> None:
    """
    Rendert eine Fluent UI Info Box.
    
    Args:
        message: Nachricht
        box_type: Typ der Box (info, success, warning, error)
        icon: Optional Icon (Emoji)
    """
    color_map = {
        "info": (FluentColors.INFO, "#F3F9FD"),
        "success": (FluentColors.SUCCESS, "#F1FAF1"),
        "warning": (FluentColors.WARNING, "#FFF9F5"),
        "error": (FluentColors.ERROR, "#FDF3F4"),
    }
    
    icon_map = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }
    
    border_color, bg_color = color_map.get(box_type, color_map["info"])
    default_icon = icon_map.get(box_type, "ℹ️")
    display_icon = icon if icon else default_icon
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        border-radius: {FluentBorders.RADIUS_MEDIUM};
        padding: {FluentSpacing.M};
        margin: {FluentSpacing.M} 0;
        display: flex;
        align-items: flex-start;
        gap: {FluentSpacing.S};
    ">
        <span style="font-size: 20px; line-height: 20px;">{display_icon}</span>
        <div style="
            color: {FluentColors.NEUTRAL_FOREGROUND};
            font-size: 14px;
            line-height: 20px;
            flex: 1;
        ">
            {message}
        </div>
    </div>
    """, unsafe_allow_html=True)


def fluent_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    icon: Optional[str] = None
) -> None:
    """
    Rendert eine Fluent UI Metric Card.
    
    Args:
        label: Label der Metrik
        value: Wert der Metrik
        delta: Optional Delta-Wert
        delta_positive: Ob Delta positiv ist
        icon: Optional Icon (Emoji)
    """
    delta_color = FluentColors.SUCCESS if delta_positive else FluentColors.ERROR
    delta_html = ""
    
    if delta:
        arrow = "↑" if delta_positive else "↓"
        delta_html = f"""
        <div style="
            color: {delta_color};
            font-size: 14px;
            font-weight: 600;
            margin-top: {FluentSpacing.XXS};
        ">
            {arrow} {delta}
        </div>
        """
    
    icon_html = f"<div style='font-size: 32px; margin-bottom: {FluentSpacing.XS};'>{icon}</div>" if icon else ""
    
    st.markdown(f"""
    <div style="
        background-color: {FluentColors.NEUTRAL_BACKGROUND};
        border: 1px solid {FluentColors.NEUTRAL_STROKE};
        border-radius: {FluentBorders.RADIUS_MEDIUM};
        padding: {FluentSpacing.M};
        margin: {FluentSpacing.M} 0;
    ">
        {icon_html}
        <div style="
            color: {FluentColors.NEUTRAL_FOREGROUND_SECONDARY};
            font-size: 14px;
            line-height: 20px;
            margin-bottom: {FluentSpacing.XXS};
        ">
            {label}
        </div>
        <div style="
            color: {FluentColors.NEUTRAL_FOREGROUND};
            font-size: 28px;
            font-weight: 600;
            line-height: 36px;
        ">
            {value}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def fluent_button_group(buttons: List[Dict[str, Any]]) -> None:
    """
    Rendert eine Gruppe von Buttons im Fluent UI Style.
    
    Args:
        buttons: Liste von Button-Definitionen mit 'label', 'key', 'type' (primary/secondary)
    """
    cols = st.columns(len(buttons))
    
    for col, button in zip(cols, buttons):
        with col:
            button_type = button.get("type", "secondary")
            if button_type == "primary":
                st.button(
                    button["label"],
                    key=button["key"],
                    type="primary",
                    use_container_width=True
                )
            else:
                st.button(
                    button["label"],
                    key=button["key"],
                    use_container_width=True
                )
