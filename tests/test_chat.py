"""The desk: what it says, when it stays quiet, and who does the arithmetic.

The feed's whole job is to report what changed, so being wrong about the
direction of a change is the one failure that makes it worthless. A 7B asked
to compare two numbers it had read in different places produced "1526
articles, up from 1527" -- a fall reported as a rise -- which is why `changed`
exists and why these tests are mostly about it.

That function takes no model and returns finished sentences, so every case
below is exact. The model's remaining job is to write them up, and there is a
test that it is told not to do the sums.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import chat, context                               # noqa: E402


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "CHAT_DIR", str(tmp_path / "chat"))
    return tmp_path


def opinion(**over):
    base = {
        "reading": [{"kind": "market", "text": "London is busy."}],
        "markets": [{"market": "London", "articles": 40, "tracked": 20,
                     "per_name": 2.0, "usual": None, "times": None}],
        "loudest": [{"symbol": "AAL.L", "market": "London", "articles": 10,
                     "usual": 1.5, "times": 6.6, "move": -0.027,
                     "headlines": ["Miners slide"]}],
        "suggestions": [],
        "unwatched_markets": [],
        "readiness": {"items": 2085, "history_days": 20, "needed_days": 365},
        "without_baseline": 12,
    }
    base.update(over)
    return base


def snap(opinion_dict):
    return chat._snapshot(opinion_dict)


# --- the arithmetic ------------------------------------------------------------

def test_a_fall_is_reported_as_a_fall(desk):
    """The bug this function was written for.

    The model wrote "1526 articles, up from 1527". The direction of a change
    is not something a language model should be deriving.
    """
    before = snap(opinion())
    after = snap(opinion(loudest=[{
        "symbol": "AAL.L", "market": "London", "articles": 6, "usual": 1.5,
        "times": 4.0, "move": 0.0, "headlines": []}]))

    lines = chat.changed(before, after)
    assert any("down from 10 at the last line" in line for line in lines), lines
    assert not any("up from" in line for line in lines), lines


def test_a_rise_is_reported_as_a_rise(desk):
    before = snap(opinion())
    after = snap(opinion(loudest=[{
        "symbol": "AAL.L", "market": "London", "articles": 14, "usual": 1.5,
        "times": 9.3, "move": 0.0, "headlines": []}]))

    lines = chat.changed(before, after)
    assert any("14 articles now, up from 10" in line for line in lines), lines


def test_an_unchanged_archive_produces_no_lines(desk):
    same = snap(opinion())
    assert chat.changed(same, same) == []


def test_the_first_snapshot_has_nothing_to_compare(desk):
    assert chat.changed(None, snap(opinion())) == []


def test_a_new_name_and_a_departed_one_are_both_reported(desk):
    before = snap(opinion())
    after = snap(opinion(loudest=[{
        "symbol": "SHEL.L", "market": "London", "articles": 8, "usual": 2.0,
        "times": 4.0, "move": 0.0, "headlines": []}]))

    lines = chat.changed(before, after)
    assert any("SHEL.L is newly among the loudest" in line for line in lines), lines
    assert any("AAL.L is no longer among the loudest" in line for line in lines), lines


def test_a_suggestion_becoming_corroborated_is_the_interesting_case(desk):
    weak = snap(opinion(suggestions=[{
        "symbol": "BA", "mentions": 3, "corroborated": False}]))
    strong = snap(opinion(suggestions=[{
        "symbol": "BA", "mentions": 3, "corroborated": True}]))

    lines = chat.changed(weak, strong)
    assert any("BA is now corroborated" in line for line in lines), lines


def test_a_new_market_with_nothing_tracked_is_reported(desk):
    before = snap(opinion())
    after = snap(opinion(unwatched_markets=[{
        "market": "Toronto", "distinct_names": 3, "mentions": 9, "days": 10,
        "providers": 1, "names": ["MFC.TO"], "headlines": []}]))

    lines = chat.changed(before, after)
    assert any("Toronto is newly named with nothing tracked" in line
               for line in lines), lines


def test_the_archive_growing_is_context_and_not_news(desk, monkeypatch):
    """It gains items nearly every cycle. Speaking about that every cycle is
    how a feed becomes something you stop opening."""
    before = snap(opinion())
    after = snap(opinion(readiness={"items": 2500, "history_days": 21,
                                    "needed_days": 365}))

    assert chat.changed(before, after) == []
    assert "gained 415 items" in chat.growth(before, after)

    asked = []
    monkeypatch.setattr(context, "_ask",
                        lambda *a, **k: asked.append(1) or "a line")
    chat.notice(opinion())
    assert len(asked) == 1

    # Only the item count moved, so it has nothing to report.
    assert chat.notice(opinion(readiness={"items": 2500, "history_days": 21,
                                          "needed_days": 365})) is None
    assert len(asked) == 1


# --- when it speaks ------------------------------------------------------------

def test_it_says_nothing_when_nothing_moved(desk, monkeypatch):
    """And it does not call the model to find that out.

    Whether anything changed is a question the code answers exactly. Asking a
    model was how a fall got reported as a rise.
    """
    asked = []
    monkeypatch.setattr(context, "_ask",
                        lambda *a, **k: asked.append(k) or "something")

    first = chat.notice(opinion())
    assert first is not None
    assert len(asked) == 1

    again = chat.notice(opinion())
    assert again is None, "it spoke twice about the same archive"
    assert len(asked) == 1, "it called the model to ask whether nothing changed"


def test_the_computed_changes_are_handed_over_not_derived(desk, monkeypatch):
    seen = {}

    def capture(prompt, **kwargs):
        seen["prompt"] = prompt
        return "AAL.L is down to 6 articles from 10."

    monkeypatch.setattr(context, "_ask", capture)

    chat.notice(opinion())
    chat.notice(opinion(loudest=[{
        "symbol": "AAL.L", "market": "London", "articles": 6, "usual": 1.5,
        "times": 4.0, "move": 0.0, "headlines": []}]))

    prompt = seen["prompt"]
    assert "CHANGES since you last spoke" in prompt
    assert "down from 10 at the last line" in prompt
    assert "never recompute it" in chat.NOTICE_SYSTEM
    assert "do not work out any arithmetic of your own" in prompt


def test_the_line_carries_the_numbers_it_was_written_from(desk, monkeypatch):
    """So a claim made last week can be checked against the archive of the day."""
    monkeypatch.setattr(context, "_ask", lambda *a, **k: "a line")

    said = chat.notice(opinion())
    assert said["snapshot"]["loudest"]["AAL.L"]["articles"] == 10
    assert chat.last_snapshot()["loudest"]["AAL.L"]["articles"] == 10


def test_with_no_model_it_writes_nothing(desk, monkeypatch):
    monkeypatch.setattr(context, "_ask", lambda *a, **k: None)
    assert chat.notice(opinion()) is None
    assert chat.feed() == []


def test_a_question_and_its_answer_are_recorded_together(desk, monkeypatch):
    monkeypatch.setattr(context, "answer",
                        lambda *a, **k: {"text": "an answer", "sources": [],
                                         "generated": True})

    chat.ask("what is going on?")
    kinds = [entry["kind"] for entry in chat.feed()]
    assert kinds == ["answer", "question"], kinds


def test_an_empty_question_is_refused(desk):
    with pytest.raises(ValueError, match="question"):
        chat.ask("   ")


def test_the_transcript_is_appended_never_edited(desk, monkeypatch):
    monkeypatch.setattr(context, "_ask", lambda *a, **k: "one")
    chat.notice(opinion())
    before = open(chat._path(), encoding="utf-8").read()

    monkeypatch.setattr(context, "_ask", lambda *a, **k: "two")
    chat.notice(opinion(loudest=[{
        "symbol": "AAL.L", "market": "London", "articles": 20, "usual": 1.5,
        "times": 13.3, "move": 0.0, "headlines": []}]))
    after = open(chat._path(), encoding="utf-8").read()

    assert after.startswith(before)
    assert len(chat.feed()) == 2


# --- the seam -------------------------------------------------------------------

def test_nothing_that_builds_features_can_read_the_desk():
    import ast
    import inspect

    from trader import cross, dataset, evaluate, features, labels, macro, news

    for module in (dataset, features, labels, macro, cross, evaluate, news):
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(alias.name for alias in node.names)
        assert not {n for n in names if "chat" in n}, module.__name__


def test_its_own_previous_lines_are_kept_out_of_the_prompt(desk, monkeypatch):
    """It copied one, verbatim, in exactly the case that matters.

    The transcript used to travel under "what you already said -- do not
    repeat it". CHANGES said "6 articles now, up from 3"; the model wrote
    "decreased to 3 articles from 6", which was its own previous line word for
    word and the opposite of the truth. A small model reproduces whatever is
    in front of it whatever the heading says, and `changed` already guarantees
    there is something new, so the prompt does not need the history.
    """
    prompts = []
    monkeypatch.setattr(context, "_ask",
                        lambda prompt, **k: prompts.append(prompt) or "a line")

    chat.notice(opinion())
    chat.notice(opinion(loudest=[{
        "symbol": "AAL.L", "market": "London", "articles": 3, "usual": 1.5,
        "times": 2.0, "move": 0.0, "headlines": []}]))

    latest = prompts[-1]
    assert "RECENTLY" not in latest
    assert "a line" not in latest, "its own previous output reached the prompt"
    assert "down from 10 at the last line" in latest
