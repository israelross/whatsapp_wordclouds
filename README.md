# WhatsApp Chat Analyzer

Parses a WhatsApp chat export and produces a visual summary report.

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone or download this repository, then enter the project folder
cd whatsapp_chat_analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Exporting your WhatsApp chat

1. Open the chat in WhatsApp on your phone.
2. Tap ⋮ (Android) or the contact name (iOS) → **More** → **Export chat**.
3. Choose **Without media** and save the resulting `_chat.txt` file somewhere accessible.

## Usage

```
python parse_chat.py <chat_file> [--output-dir <dir>] [--all]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `chat_file` | Yes | Path to the exported `_chat.txt` file |
| `--output-dir` | No | Folder to save output images. Defaults to the same folder as the chat file |
| `--all` | No | Also save each plot as a separate image in addition to the summary report |

### Output files

| File | Always saved | Only with `--all` |
|---|---|---|
| `summary_report.png` | ✓ | |
| `activity_heatmap.png` | | ✓ |
| `messages_per_week.png` | | ✓ |
| `wordclouds.png` | | ✓ |

### Examples

```bash
# Generate only the summary report, saved next to the chat file
python parse_chat.py /path/to/_chat.txt

# Save all plots to a specific output folder
python parse_chat.py /path/to/_chat.txt --output-dir ./reports --all

# Save only the summary to a custom folder
python parse_chat.py /path/to/_chat.txt --output-dir ./reports
```

## Summary report contents

The single-page `summary_report.png` contains five panels:

1. **Messages per week** — stacked bar chart showing each sender's weekly message count.
2. **Activity heatmap** — message density by day of week and hour of day (starts at 05:00).
3. **Message share** — pie chart of total messages per sender.
4. **Average message length** — horizontal bar chart of mean words per message per sender.
5. **Word clouds** — one cloud per sender showing their most distinctively used words (sized by statistical significance via chi-square p-value).
