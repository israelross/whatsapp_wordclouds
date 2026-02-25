import argparse
import logging
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import chi2 as chi2_dist
from wordcloud import WordCloud

logging.basicConfig(format="%(message)s", level=logging.INFO)


# Matches lines like: [26/06/2022, 14:05:52] Sender Name: message
# The leading \u200e (left-to-right mark) is stripped before matching
MSG_PATTERN = re.compile(
    r"^\[(\d{2}/\d{2}/\d{4}), (\d{1,2}:\d{2}:\d{2})\] ([^:]+): (.*)"
)

# Inline meta annotations WhatsApp appends to message content.
# Order matters: the document pattern (with filename preamble) must come first
# so it matches the full "file.pdf • ‎N pages ‎document omitted" string before
# the simpler "document omitted" alternative can match only the suffix.
_META_RE = re.compile(
    r".+\.\w+\s*•\s*\u200e\d+\s+pages?\s*\u200e?document omitted"   # document with filename
    r"|\u200e?(?:image|audio|video|GIF|sticker|document|Contact\s+card) omitted"
    r"|\u200e?<This message was edited>",
    re.IGNORECASE,
)
# Whole messages that carry no user content
_DROP_CONTENT = re.compile(
    r"^\u200e?You deleted this message\.?\s*$"
    r"|^\u200e?Missed (?:voice|video) call[\u200e\w ,]*$",
    re.IGNORECASE,
)


def parse_chat(filepath: str) -> pd.DataFrame:
    records = []
    current = None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").lstrip("\u200e\u200f\u202a\u202c")
            match = MSG_PATTERN.match(line)
            if match:
                if current:
                    records.append(current)
                date_str, time_str, sender, content = match.groups()
                current = {
                    "sender": sender,
                    "date": datetime.strptime(date_str, "%d/%m/%Y").date(),
                    "time_of_day": datetime.strptime(time_str, "%H:%M:%S").time(),
                    "content": content,
                }
            elif current:
                # Continuation of a multi-line message
                current["content"] += "\n" + line

    if current:
        records.append(current)

    df = pd.DataFrame(records, columns=["sender", "date", "time_of_day", "content"])

    # Strip inline meta annotations, then drop rows that are now empty or
    # that contained only a "You deleted this message" notice
    df["content"] = df["content"].apply(lambda t: _META_RE.sub("", t).strip())
    df = df[df["content"] != ""]
    df = df[~df["content"].str.match(_DROP_CONTENT)]

    return df.reset_index(drop=True)


_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
_HEB_PUNCT = str.maketrans("", "", "\u05F3\u05F4")  # ׳ ״ → drop


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip Hebrew geresh/gershayim, then extract words of ≥ 2 letters."""
    cleaned = text.lower().translate(_HEB_PUNCT)
    return [w for w in _TOKEN_RE.findall(cleaned) if len(w) >= 2]


def word_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for word in _tokenize(row["content"]):
            rows.append({"word": word, "sender": row["sender"]})

    if not rows:
        return pd.DataFrame()

    counts = (
        pd.DataFrame(rows)
        .groupby(["word", "sender"])
        .size()
        .unstack(level="sender", fill_value=0)
    )
    counts.columns.name = None
    counts.index.name = "word"
    return counts


def enrich_word_frequencies(wf: pd.DataFrame, top_n: int = 1000) -> pd.DataFrame:
    senders = list(wf.columns)

    # Total word count per sender across the full vocabulary (before filtering)
    total_per_sender = wf.sum()   # Series: sender -> int
    total_words = float(total_per_sender.sum())

    result = wf.copy()
    result["total_usage"] = result[senders].sum(axis=1)

    # Keep only the top_n most-used words
    result = result.nlargest(top_n, "total_usage").copy()

    # Per-sender frequency = word count / total words by that sender
    for sender in senders:
        result[f"freq_{sender}"] = result[sender] / float(total_per_sender[sender])

    # Overall frequency = total_usage / all words in chat
    result["total_frequency"] = result["total_usage"] / total_words

    # Chi-square p-value: how uniquely sender S uses this word vs. all other senders.
    # For each (word, sender) we build a 2×2 contingency table:
    #
    #              word W    not word W
    #   sender S:    a           b
    #   others:      c           d
    #
    # and return the p-value of the chi-square test of independence.
    # Low p-value → word usage rate differs significantly between S and the rest.
    for sender in senders:
        others = [s for s in senders if s != sender]
        total_sender = float(total_per_sender[sender])
        total_others = float(total_per_sender[others].sum()) if others else 0.0

        a = result[sender].values.astype(float)
        c = result[others].sum(axis=1).values.astype(float) if others else np.zeros(len(result))
        b = total_sender - a
        d = total_others - c

        n = a + b + c + d
        denom = (a + b) * (c + d) * (a + c) * (b + d)
        chi2_stat = np.where(denom > 0, n * (a * d - b * c) ** 2 / denom, 0.0)
        result[f"chi2_pval_{sender}"] = chi2_dist.sf(chi2_stat, df=1)

    # Whether the sender uses each word more (+) or less (-) than the chat average
    for sender in senders:
        result[f"above_avg_{sender}"] = result[f"freq_{sender}"] > result["total_frequency"]

    return result


def plot_sender_wordclouds(
    ewf: pd.DataFrame,
    output_path: str = "wordclouds.png",
    max_words: int = 100,
    font_path: str = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
) -> None:
    def fix_rtl(word: str) -> str:
        return word[::-1] if any("\u0590" <= c <= "\u05FF" for c in word) else word

    senders = [c.removeprefix("chi2_pval_") for c in ewf.columns if c.startswith("chi2_pval_")]
    n = len(senders)
    cols = min(n, 2)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 7 * rows), squeeze=False)
    axes = axes.flatten()

    for ax, sender in zip(axes, senders):
        # Keep only words the sender uses more than average
        subset = ewf[ewf[f"above_avg_{sender}"]].copy()

        # Take the top max_words by significance (smallest p-value first)
        subset = subset.nsmallest(max_words, f"chi2_pval_{sender}")

        # Word size ∝ -log10(p): more significant → bigger word
        p_clipped = subset[f"chi2_pval_{sender}"].clip(lower=1e-300)
        scores = -np.log10(p_clipped)
        weights = {fix_rtl(word): score for word, score in scores.items()}

        wc = WordCloud(
            width=1200,
            height=700,
            background_color="white",
            max_words=max_words,
            font_path=font_path,
        ).generate_from_frequencies(weights)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(sender, fontsize=18, fontweight="bold", pad=10)

        # Hide ticks but keep a visible border around each subplot
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_color("#888888")

    # Hide unused subplot slots
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    logging.info(f"Saved: {output_path}")


DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def plot_activity_heatmap(df: pd.DataFrame, output_path: str = "activity_heatmap.png") -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_heatmap(ax, df)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Saved: {output_path}")


def _week_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (week_start × sender) message-count pivot, partial boundary weeks dropped."""
    dates = pd.to_datetime(df["date"])
    df2 = df.copy()
    df2["week_start"] = dates - pd.to_timedelta((dates.dt.dayofweek + 1) % 7, unit="D")
    pivot = df2.groupby(["week_start", "sender"]).size().unstack(fill_value=0)
    if dates.min().dayofweek != 6:
        pivot = pivot.iloc[1:]
    if dates.max().dayofweek != 5:
        pivot = pivot.iloc[:-1]
    return pivot


def _draw_weekly_bars(ax: plt.Axes, pivot: pd.DataFrame, sender_colors: dict) -> None:
    bottom = np.zeros(len(pivot))
    for sender in pivot.columns:
        ax.bar(pivot.index, pivot[sender], bottom=bottom, width=5,
               label=sender, color=sender_colors[sender], alpha=0.85)
        bottom += pivot[sender].values
    ax.set_xlabel("Week (starting Sunday)")
    ax.set_ylabel("Messages")
    ax.set_title("Messages per Week")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _draw_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    data = df.copy()
    data["day_of_week"] = (pd.to_datetime(data["date"]).dt.dayofweek + 1) % 7
    data["hour"] = pd.to_datetime(data["time_of_day"], format="%H:%M:%S").dt.hour
    counts = (data.groupby(["day_of_week", "hour"]).size()
              .unstack(level="hour", fill_value=0))
    hour_order = list(range(5, 24)) + list(range(0, 5))
    counts = counts.reindex(index=range(7), columns=hour_order, fill_value=0)
    counts.index = DAY_NAMES
    sns.heatmap(counts, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Messages"})
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title("Chat Activity Heatmap")
    ax.set_xticklabels([f"{h:02d}:00" for h in hour_order], rotation=45, ha="right")


def _draw_pie(ax: plt.Axes, df: pd.DataFrame, sender_colors: dict) -> None:
    counts = df["sender"].value_counts()
    colors = [sender_colors[s] for s in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
        pctdistance=0.75,
        labeldistance=1.25,
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color("black")
    ax.set_title("Message Share by Sender", fontsize=13, pad=12)


def _draw_avg_words(ax: plt.Axes, df: pd.DataFrame, sender_colors: dict) -> None:
    tmp = df.copy()
    tmp["n"] = tmp["content"].apply(lambda t: len(_tokenize(t)))
    avg = tmp.groupby("sender")["n"].mean().sort_values()
    bars = ax.barh(avg.index, avg.values,
                   color=[sender_colors[s] for s in avg.index],
                   edgecolor="white", linewidth=1.5, height=0.5)
    for bar, val in zip(bars, avg.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Avg words per message", fontsize=11)
    ax.set_title("Average Message Length", fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, avg.max() * 1.2)


def word_frequencies_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Word × year count matrix (all senders aggregated)."""
    tmp = df.copy()
    tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
    tmp["tokens"] = tmp["content"].apply(_tokenize)
    tmp = tmp.explode("tokens").dropna(subset=["tokens"])
    tmp = tmp.rename(columns={"tokens": "word"})
    counts = (
        tmp.groupby(["word", "year"])
        .size()
        .unstack(level="year", fill_value=0)
    )
    counts.columns.name = None
    counts.index.name = "word"
    return counts


def year_over_year_changes(
    wf_year: pd.DataFrame, top_n: int = 10, min_count_per_year: int = 20
) -> dict:
    """For each consecutive year pair (Y-1 → Y), compute a chi-square p-value for
    every word and return the top_n most-changed words per year.

    Only words with at least *min_count_per_year* appearances in **both** the
    previous year and the current year are considered.
    Returns dict: year → list of (word, direction) where direction is '+' (more used
    in Y than Y-1) or '-' (less used).
    """
    years = sorted(wf_year.columns)
    total_per_year = wf_year.sum()

    result = {}
    for i in range(1, len(years)):
        y_curr, y_prev = years[i], years[i - 1]

        # Only include words that appear enough in both years being compared
        mask = (wf_year[y_curr] >= min_count_per_year) & (wf_year[y_prev] >= min_count_per_year)
        wf = wf_year[mask]

        a = wf[y_curr].values.astype(float)   # word count current year
        c = wf[y_prev].values.astype(float)   # word count previous year
        b = float(total_per_year[y_curr]) - a
        d = float(total_per_year[y_prev]) - c

        n = a + b + c + d
        denom = (a + b) * (c + d) * (a + c) * (b + d)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2_stat = np.where(denom > 0, n * (a * d - b * c) ** 2 / denom, 0.0)
        p_vals = chi2_dist.sf(chi2_stat, df=1)

        freq_curr = a / max(float(total_per_year[y_curr]), 1)
        freq_prev = c / max(float(total_per_year[y_prev]), 1)
        directions = np.where(freq_curr >= freq_prev, "+", "-")

        year_df = pd.DataFrame(
            {"word": wf.index, "p_val": p_vals, "direction": directions}
        )
        result[y_curr] = list(
            zip(year_df.nsmallest(top_n, "p_val")["word"],
                year_df.nsmallest(top_n, "p_val")["direction"])
        )

    return result


def _draw_year_word_table(ax: plt.Axes, changes: dict, font_path: str) -> None:
    from matplotlib.font_manager import FontProperties

    def fix_rtl(w: str) -> str:
        return w[::-1] if any("\u0590" <= c <= "\u05FF" for c in w) else w

    ax.axis("off")
    ax.set_title("Most Changed Words Year-over-Year", fontsize=13, pad=12)

    years = sorted(changes.keys())
    if not years:
        ax.text(0.5, 0.5, "Not enough years of data.",
                ha="center", va="center", transform=ax.transAxes)
        return

    n_cols = len(years)
    n_rows = max(len(v) for v in changes.values())
    col_w = 1.0 / n_cols
    top = 0.92
    row_h = top / (n_rows + 1.5)

    fp_header = FontProperties(fname=font_path, size=11, weight="bold")
    fp_cell = FontProperties(fname=font_path, size=10)

    # Column headers
    for j, year in enumerate(years):
        ax.text((j + 0.5) * col_w, top, str(year),
                ha="center", va="top", fontproperties=fp_header,
                transform=ax.transAxes)

    # Separator under header
    y_sep = top - row_h * 0.8
    ax.plot([0.01, 0.99], [y_sep, y_sep], color="#aaaaaa", linewidth=1.2,
            transform=ax.transAxes, clip_on=False)

    # Vertical column separators
    for j in range(1, n_cols):
        ax.plot([j * col_w, j * col_w], [0.02, 0.95], color="#eeeeee",
                linewidth=1, transform=ax.transAxes, clip_on=False)

    # Word cells
    for j, year in enumerate(years):
        for i, (word, direction) in enumerate(changes[year]):
            color = "#27ae60" if direction == "+" else "#c0392b"
            arrow = "↑" if direction == "+" else "↓"
            y_pos = top - (i + 2) * row_h
            ax.text((j + 0.5) * col_w, y_pos,
                    f"{arrow} {fix_rtl(word)}",
                    ha="center", va="center", color=color,
                    fontproperties=fp_cell, transform=ax.transAxes)


def plot_messages_per_week(
    df: pd.DataFrame, output_path: str = "messages_per_week.png"
) -> None:
    senders = sorted(df["sender"].unique())
    palette = sns.color_palette("Set2", len(senders))
    sender_colors = {s: palette[i] for i, s in enumerate(senders)}
    pivot = _week_pivot(df)

    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_weekly_bars(ax, pivot, sender_colors)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_summary_report(
    df: pd.DataFrame,
    ewf: pd.DataFrame,
    output_path: str = "summary_report.png",
    max_wc_words: int = 100,
    font_path: str = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
) -> None:
    def fix_rtl(word: str) -> str:
        return word[::-1] if any("\u0590" <= c <= "\u05FF" for c in word) else word

    senders = sorted(df["sender"].unique())
    palette = sns.color_palette("Set2", len(senders))
    sender_colors = {s: palette[i] for i, s in enumerate(senders)}
    wc_senders = [c.removeprefix("chi2_pval_") for c in ewf.columns if c.startswith("chi2_pval_")]
    n_wc = len(wc_senders)
    wc_rows = math.ceil(n_wc / 2)

    height_ratios = [3, 3, 2, 4] + [5] * wc_rows
    fig = plt.figure(figsize=(22, sum(height_ratios) * 1.3))
    gs = fig.add_gridspec(4 + wc_rows, 2, height_ratios=height_ratios,
                          hspace=0.55, wspace=0.3)

    # 1 — Messages per week (stacked)
    _draw_weekly_bars(fig.add_subplot(gs[0, :]), _week_pivot(df), sender_colors)

    # 2 — Activity heatmap
    _draw_heatmap(fig.add_subplot(gs[1, :]), df)

    # 3 — Pie chart
    _draw_pie(fig.add_subplot(gs[2, 0]), df, sender_colors)

    # 4 — Avg words per message
    _draw_avg_words(fig.add_subplot(gs[2, 1]), df, sender_colors)

    # 5 — Year-over-year word change table
    wf_year = word_frequencies_by_year(df)
    changes = year_over_year_changes(wf_year)
    _draw_year_word_table(fig.add_subplot(gs[3, :]), changes, font_path)

    # 6 — Word clouds
    for i, sender in enumerate(wc_senders):
        ax_wc = fig.add_subplot(gs[4 + i // 2, i % 2])
        subset = ewf[ewf[f"above_avg_{sender}"]].nsmallest(max_wc_words, f"chi2_pval_{sender}")
        p_clipped = subset[f"chi2_pval_{sender}"].clip(lower=1e-300)
        weights = {fix_rtl(w): s for w, s in (-np.log10(p_clipped)).items()}
        wc_img = WordCloud(width=2000, height=1000, background_color="white",
                           max_words=max_wc_words,
                           font_path=font_path).generate_from_frequencies(weights)
        ax_wc.imshow(wc_img, interpolation="bilinear")
        ax_wc.set_title(sender, fontsize=14, fontweight="bold", pad=8)
        ax_wc.set_xticks([])
        ax_wc.set_yticks([])
        for spine in ax_wc.spines.values():
            spine.set_linewidth(2)
            spine.set_color("#888888")

    for i in range(n_wc, wc_rows * 2):
        fig.add_subplot(gs[4 + i // 2, i % 2]).set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    logging.info(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse a WhatsApp chat export and produce visualisations."
    )
    parser.add_argument("chat_file", type=Path,
                        help="Path to the exported WhatsApp chat file (_chat.txt).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save output files. "
                             "Defaults to the same folder as the chat file.")
    parser.add_argument("--all", dest="save_all", action="store_true",
                        help="Also save each plot as a separate image file.")
    parser.add_argument("--start-date", type=str, default=None,
                        metavar="YYYY-MM-DD",
                        help="Ignore all messages before this date (inclusive from this date).")
    args = parser.parse_args()

    if not args.chat_file.is_file():
        logging.error(f"File not found: {args.chat_file}")
        sys.exit(1)

    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            logging.error(f"Invalid --start-date '{args.start_date}'. Expected format: YYYY-MM-DD.")
            sys.exit(1)

    out_dir = args.output_dir if args.output_dir else args.chat_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    def out(filename: str) -> str:
        return str(out_dir / filename)

    logging.info(f"Parsing {args.chat_file} ...")
    df = parse_chat(str(args.chat_file))

    if start_date:
        before = len(df)
        df = df[df["date"] >= start_date].reset_index(drop=True)
        logging.info(f"  Filtered to messages from {start_date} onwards "
                     f"({before - len(df):,} messages removed).")

    logging.info(f"  {len(df):,} messages from {df['sender'].nunique()} senders.")

    logging.info("Computing word frequencies ...")
    wf = word_frequencies(df)
    ewf = enrich_word_frequencies(wf)

    if args.save_all:
        plot_activity_heatmap(df, output_path=out("activity_heatmap.png"))
        plot_messages_per_week(df, output_path=out("messages_per_week.png"))
        plot_sender_wordclouds(ewf, output_path=out("wordclouds.png"))

    logging.info("Generating summary report ...")
    plot_summary_report(df, ewf, output_path=out("summary_report.png"))
