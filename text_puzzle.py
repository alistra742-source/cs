#!/usr/bin/env python3
"""text_puzzle.py — answer hCaptcha's text-entry rounds without a model.

hCaptcha increasingly serves plain word problems instead of images:

    "The jar begins with 19 coins. On Sunday, you place 9 coins in the
     jar. How many coins are in the jar now?"

The prompt is already captured as TEXT, so this needs no vision at all —
it is arithmetic and simple language parsing. A local answer is also
instant and free, where a vision round trip is neither.

solve(prompt) -> answer string, or "" when it cannot be answered.
"""

from __future__ import annotations

import re

# number words hCaptcha actually uses
_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "a": 1, "an": 1,
}

_ADD = ("place", "places", "placed", "add", "adds", "added", "put", "puts",
        "gain", "gains", "gained", "receive", "receives", "received",
        "buy", "buys", "bought", "find", "finds", "found", "more",
        "additional", "another", "plus", "insert", "inserts", "drop",
        "drops", "dropped", "deposit", "deposits")

_SUB = ("remove", "removes", "removed", "take", "takes", "took", "taken",
        "lose", "loses", "lost", "spend", "spends", "spent", "give",
        "gives", "gave", "eat", "eats", "ate", "sell", "sells", "sold",
        "minus", "fewer", "less", "withdraw", "withdraws", "subtract")

_MUL = ("times", "twice", "double", "doubles", "doubled", "triple",
        "triples", "tripled", "multiplied")


def _numbers(text: str):
    """Every number in order, digits and words alike."""
    out = []
    for tok in re.findall(r"[a-z]+|\d+", text.lower()):
        if tok.isdigit():
            out.append(int(tok))
        elif tok in _WORDS and tok not in ("a", "an"):
            out.append(_WORDS[tok])
    return out


def _clauses(text: str):
    """Split into clauses so each number keeps its own verb."""
    return [c for c in re.split(r"[.;,]|\band\b|\bthen\b", text.lower())
            if c.strip()]


def solve_arithmetic(prompt: str) -> str:
    """Answer a word-problem prompt, or "" if it is not one."""
    p = " ".join((prompt or "").split()).lower()
    if not p:
        return ""
    if not re.search(r"how many|how much|what is|total|altogether|"
                     r"in total|now\?|result", p):
        return ""

    total = None
    for clause in _clauses(p):
        nums = _numbers(clause)
        if not nums:
            continue
        if total is None:
            # the first number seen is the starting amount
            total = nums[0]
            rest = nums[1:]
        else:
            rest = nums
        for n in rest:
            if any(w in clause for w in _MUL):
                total *= n
            elif any(w in clause for w in _SUB):
                total -= n
            elif any(w in clause for w in _ADD):
                total += n
            else:
                total += n          # default: accumulate
        # a clause that only restates the start must not double-count
    if total is None:
        return ""
    if "double" in p or "doubles" in p or "doubled" in p:
        pass                        # already handled per-clause
    return str(total)


def solve(prompt: str) -> str:
    """Best-effort answer for a text-entry round."""
    p = " ".join((prompt or "").split())
    if not p:
        return ""
    # Direct arithmetic: "what is 7 plus 4"
    m = re.search(r"(\d+)\s*(?:\+|plus)\s*(\d+)", p, re.I)
    if m:
        return str(int(m.group(1)) + int(m.group(2)))
    m = re.search(r"(\d+)\s*(?:-|minus)\s*(\d+)", p, re.I)
    if m:
        return str(int(m.group(1)) - int(m.group(2)))
    m = re.search(r"(\d+)\s*(?:\*|x|times)\s*(\d+)", p, re.I)
    if m:
        return str(int(m.group(1)) * int(m.group(2)))
    return solve_arithmetic(p)


if __name__ == "__main__":  # pragma: no cover
    tests = [
        "The jar begins with 19 coins. On Sunday, you place 9 coins in the "
        "jar. How many coins are in the jar now?",
        "You have 12 apples and you eat 5. How many apples are left?",
        "A box holds seven balls. You add three more. How many now?",
        "What is 7 plus 4?",
        "Please click each image containing a boat",
    ]
    for t in tests:
        print(f"{solve(t)!r:8s} <- {t[:60]}")
