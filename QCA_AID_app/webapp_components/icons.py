"""Zentrale Icon-Registry der QCA-AID Webapp.

Alle UI-Icons werden als Streamlit-Shorthand ``:material/<name>:`` referenziert.
Streamlit buendelt den Font "Material Symbols Outlined" lokal (offline-faehig),
die Glyphen erben Textfarbe und -groesse (currentColor) und rendern in allen
Labels (Buttons, Expander, Callouts, Markdown, Captions, Subheader ...).

CLI-/Log-Output behalten bewusst Emojis -- diese Registry gilt nur fuer die UI.

Regel: Pro Bedeutung genau EIN Icon. Neue Icons nur hier definieren.
"""
from __future__ import annotations

# ---------------------------------------------------------------- Konstanten --
ICON_SETTINGS = ":material/settings:"
ICON_CODEBOOK = ":material/menu_book:"
ICON_ANALYSIS = ":material/biotech:"
ICON_EXPLORER = ":material/bar_chart:"
ICON_SEARCH = ":material/search:"
ICON_SUCCESS = ":material/check_circle:"
ICON_CHECK = ":material/check:"
ICON_WARNING = ":material/warning:"
ICON_ERROR = ":material/error:"
ICON_CANCEL = ":material/cancel:"
ICON_INFO = ":material/info:"
ICON_SAVE = ":material/save:"
ICON_DELETE = ":material/delete:"
ICON_ADD = ":material/add:"
ICON_FOLDER = ":material/folder:"
ICON_FOLDER_OPEN = ":material/folder_open:"
ICON_FILE = ":material/description:"
ICON_DOWNLOAD = ":material/download:"
ICON_PLAY = ":material/play_arrow:"
ICON_PAUSE = ":material/pause:"
ICON_STOP = ":material/stop_circle:"
ICON_SKIP_NEXT = ":material/skip_next:"
ICON_REFRESH = ":material/autorenew:"
ICON_SYNC = ":material/sync:"
ICON_EDIT = ":material/edit:"
ICON_EDIT_NOTE = ":material/edit_note:"
ICON_LOGS = ":material/receipt_long:"
ICON_MERGE = ":material/shuffle:"
ICON_OPTIONS = ":material/build:"
ICON_COSTS = ":material/payments:"
ICON_COSTS_FREE = ":material/savings:"
ICON_MODEL = ":material/smart_toy:"
ICON_TIMER = ":material/timer:"
ICON_DATE = ":material/calendar_month:"
ICON_TIP = ":material/lightbulb:"
ICON_ROCKET = ":material/rocket_launch:"
ICON_PALETTE = ":material/palette:"
ICON_HEATMAP = ":material/local_fire_department:"
ICON_TREND_UP = ":material/trending_up:"
ICON_LABEL = ":material/sell:"
ICON_PREVIEW = ":material/visibility:"
ICON_KEY = ":material/key:"
ICON_CHEVRON_LEFT = ":material/chevron_left:"
ICON_CHEVRON_RIGHT = ":material/chevron_right:"
ICON_ARROW_FORWARD = ":material/arrow_forward:"
ICON_ARROW_DOWN = ":material/arrow_downward:"
ICON_PRIORITY = ":material/priority_high:"
ICON_ALARM = ":material/alarm:"
ICON_ALERT = ":material/notification_important:"
ICON_FINISH = ":material/sports_score:"
ICON_BOLT = ":material/bolt:"
ICON_LIGHT_MODE = ":material/light_mode:"
ICON_HANDSHAKE = ":material/handshake:"
ICON_GROUP = ":material/group:"
ICON_SIZE = ":material/straighten:"
ICON_TARGET = ":material/my_location:"
ICON_MAP = ":material/map:"
ICON_EMPTY_INBOX = ":material/inbox:"
ICON_PSYCHOLOGY = ":material/psychology:"
ICON_SMILE = ":material/sentiment_satisfied:"
ICON_CELEBRATE = ":material/celebration:"
ICON_CONSTRUCTION = ":material/construction:"
ICON_HUB = ":material/hub:"
ICON_HOURGLASS = ":material/hourglass_top:"
ICON_BACK_HAND = ":material/back_hand:"
ICON_INVENTORY = ":material/inventory_2:"
ICON_NEW = ":material/new_releases:"
ICON_EXTENSION = ":material/extension:"

# --- Status-Icons mit farblicher Abhebung ---------------------------------
# Technik: Streamlit-Farb-Markdown um die Shorthand, z. B.
# ":green[:material/check_circle:]". Funktioniert in ALLEN Markdown-
# Kontexten (st.markdown, Button-/Radio-/Expander-Labels, Captions).
# NICHT verwenden in: help=-Texten, plain Strings ohne Markdown-Rendering.
# Farben sind Streamlit-Theme-Farben (Dark-Mode-sicher).
ICON_STATUS_OK = ":green[:material/check_circle:]"        # erledigt/erfolgreich
ICON_STATUS_WARN = ":orange[:material/warning:]"          # Aufmerksamkeit noetig
ICON_STATUS_ERROR = ":red[:material/error:]"              # Blocker/kritisch
ICON_STATUS_INFO = ":blue[:material/info:]"               # Hinweis/neutrale Aenderung
# Neutral (offen/pausiert): unfaerbige Shorthand nutzen -- erbt Textfarbe.

# ----------------------------------------------------------------- Mapping --
# Emoji -> Material-Symbol. Wird von scripts/_migrate_emojis.py und als
# Nachschlage-Legende verwendet. Sequenzen mit U+FE0F werden zuerst ersetzt.
EMOJI_TO_MATERIAL: dict[str, str] = {
    "✅": ICON_SUCCESS,
    "✓": ICON_CHECK,
    "⚠": ICON_WARNING,
    "❗": ICON_ERROR,
    "❌": ICON_CANCEL,
    "ℹ": ICON_INFO,
    "📊": ICON_EXPLORER,
    "📁": ICON_FOLDER,
    "📂": ICON_FOLDER_OPEN,
    "🔬": ICON_ANALYSIS,
    "💡": ICON_TIP,
    "📋": ICON_LOGS,
    "💾": ICON_SAVE,
    "🔄": ICON_REFRESH,
    "➕": ICON_ADD,
    "▶": ICON_PLAY,
    "🔍": ICON_SEARCH,
    "📄": ICON_FILE,
    "⚙": ICON_SETTINGS,
    "🗑": ICON_DELETE,
    "💰": ICON_COSTS,
    "💸": ICON_COSTS_FREE,
    "📚": ICON_CODEBOOK,
    "◀": ICON_CHEVRON_LEFT,
    "🤖": ICON_MODEL,
    "📥": ICON_DOWNLOAD,
    "⏱": ICON_TIMER,
    "📅": ICON_DATE,
    "🔧": ICON_OPTIONS,
    "✏": ICON_EDIT,
    "🔥": ICON_HEATMAP,
    "📈": ICON_TREND_UP,
    "👉": ICON_ARROW_FORWARD,
    "📝": ICON_EDIT_NOTE,
    "⏸": ICON_PAUSE,
    "🔑": ICON_KEY,
    "☀": ICON_LIGHT_MODE,
    "🏷": ICON_LABEL,
    "🎨": ICON_PALETTE,
    "🤝": ICON_HANDSHAKE,
    "👁": ICON_PREVIEW,
    "🗺": ICON_MAP,
    "😊": ICON_SMILE,
    "🧠": ICON_PSYCHOLOGY,
    "📭": ICON_EMPTY_INBOX,
    "⏹": ICON_STOP,
    "⏭": ICON_SKIP_NEXT,
    "👥": ICON_GROUP,
    "📏": ICON_SIZE,
    "🎯": ICON_TARGET,
    "🔀": ICON_MERGE,
    "⬇": ICON_ARROW_DOWN,
    "🎉": ICON_CELEBRATE,
    "🖐": ICON_BACK_HAND,
    "📦": ICON_INVENTORY,
    "🚀": ICON_ROCKET,
    "🕸": ICON_HUB,
    "⏳": ICON_HOURGLASS,
    "🚧": ICON_CONSTRUCTION,
    "📖": ICON_CODEBOOK,
    "🆕": ICON_NEW,
    "🧩": ICON_EXTENSION,
}


def material(emoji: str) -> str:
    """Emoji -> Shorthand; unbekannte Zeichen unverÃ¤ndert zurÃ¼ckgeben."""
    return EMOJI_TO_MATERIAL.get(emoji, emoji)
