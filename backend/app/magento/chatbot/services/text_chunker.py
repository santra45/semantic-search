"""
Recursive boundary-aware text splitter for CMS-style content.

Used by the sync pipeline before embedding so a long CMS page or block is
broken into ~target_size-char chunks with ~overlap-char overlap, then each
chunk becomes its own Qdrant point. Retrieval then matches the *right
paragraph* of a long policy page instead of having the first paragraph
dominate the whole-document embedding.

Design notes:

  * Splitter is GREEDY (one pass, left to right) — not the recursive
    LangChain-style splitter. The pass walks the text, looking for the
    last good break before the target boundary, and falls through a
    cascade of separators (paragraph → sentence → semicolon → word →
    hard char-split). This keeps the runtime O(n) for any input size,
    which matters: a 50,000-char CMS page would otherwise embed into
    100+ chunks and we don't want quadratic split cost on top.

  * Overlap is applied by *backing the cursor up* by `overlap` chars
    after each chunk close. So chunk N+1 begins `overlap` chars inside
    the tail of chunk N. The overlap is then snapped forward to the
    next word boundary so chunks don't start mid-word. Effective chunk
    size is `target_size`; overlap is the carry-over context, NOT
    additional capacity on top of target.

  * Returns a list of *non-empty* chunks. Empty / whitespace-only input
    yields [""] so the caller always has at least one chunk to embed
    (the rest of the pipeline assumes "at least one point per item").
"""

from __future__ import annotations

import re

# Separator cascade — earlier groups are preferred. Inside each group we
# pick the LAST occurrence in the lookback window, so the chunk closes at
# the most-natural boundary closest to the target size.
#
# Why grouped:
#   - Paragraph breaks ("\n\n") give the cleanest semantic boundary; if
#     present near target, always prefer them.
#   - Single newlines are next-best — they cover bulleted lists, table
#     rows, and short-paragraph layouts where authors don't double-space.
#   - Sentence terminators come next. ". " is the most common; "?" / "!"
#     / "; " are surprisingly common in policy docs (FAQ-style and
#     compound legal sentences).
#   - Space is the last-resort soft boundary — avoids splitting inside
#     a word, which corrupts the embedding (mid-word tokens are rare
#     and high-perplexity).
_SEPARATOR_CASCADE = (
    ("\n\n",),
    ("\n",),
    (". ", "? ", "! "),
    ("; ", ": "),
    (" ",),
)


def chunk_text(text: str, target_size: int = 500, overlap: int = 200) -> list[str]:
    """Split *text* into chunks of ~*target_size* chars with ~*overlap* char overlap.

    Always returns at least one element. For empty / short input the result
    is a single chunk equal to the input (stripped). For long input the
    chunks are >= 1 char each and never exceed `target_size` chars.
    """
    target_size = max(50, int(target_size))
    # Overlap can't equal or exceed target — otherwise the cursor would
    # never advance and we'd loop forever (the failsafe below catches it,
    # but better to normalise the input).
    overlap = max(0, min(int(overlap), target_size - 1))

    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= target_size:
        return [text]

    chunks: list[str] = []
    n = len(text)
    cursor = 0

    while cursor < n:
        target_end = cursor + target_size
        if target_end >= n:
            tail = text[cursor:].strip()
            if tail:
                chunks.append(tail)
            break

        # min_end stops the splitter from producing tiny degenerate chunks
        # when there's no good separator anywhere near target. We accept up
        # to a 50% under-shoot before giving up and hard-splitting at
        # target_end.
        min_end = cursor + max(target_size // 2, 1)
        best_end = _find_best_split(text, cursor, target_end, min_end)

        chunk = text[cursor:best_end].strip()
        if chunk:
            chunks.append(chunk)

        # Walk the cursor forward by (chunk_size - overlap) so chunk N+1
        # begins `overlap` chars inside chunk N's tail. Snap the new
        # cursor forward to the next space so we never start mid-word.
        next_cursor = best_end - overlap
        if next_cursor <= cursor:
            # Defensive: shouldn't happen because target_size > overlap,
            # but if a tiny chunk lands close to cursor, force forward.
            next_cursor = cursor + 1
        cursor = _snap_to_word_start(text, next_cursor, best_end)

    # Final sanity: never return an empty list — caller's "at least one
    # point per item" invariant depends on it.
    return chunks if chunks else [text]


def _find_best_split(text: str, cursor: int, target_end: int, min_end: int) -> int:
    """Walk back from target_end through the separator cascade.

    Returns the position to split at (exclusive end of the current chunk).
    Falls back to target_end when no separator is found in the window
    [min_end, target_end] — that's a hard split at char boundary, which is
    ugly but preferable to an unbounded chunk.
    """
    for group in _SEPARATOR_CASCADE:
        best = -1
        for sep in group:
            sep_len = len(sep)
            # Find the LAST occurrence of `sep` ending at or before target_end
            # and starting at or after min_end - sep_len. rfind from a window
            # is O(window), so the total scan stays linear in chunk size.
            idx = text.rfind(sep, max(min_end - sep_len, cursor), target_end)
            if idx == -1:
                continue
            candidate = idx + sep_len  # cut AFTER the separator
            if candidate > best:
                best = candidate
        if best > cursor:
            return best
    # No natural break — hard split.
    return target_end


def _snap_to_word_start(text: str, position: int, upper_bound: int) -> int:
    """Move *position* forward to the next whitespace boundary (exclusive).

    Caps at *upper_bound* so we don't skip past the chunk boundary we just
    closed. If no whitespace lies between position and upper_bound, return
    position unchanged — accepting a possibly mid-word start over an
    infinite loop.
    """
    if position >= upper_bound:
        return position
    # Already at a word boundary?
    if position == 0 or text[position - 1].isspace():
        return position
    # Find the next space within the bounded window.
    idx = text.find(" ", position, upper_bound)
    if idx == -1:
        return position
    return idx + 1
