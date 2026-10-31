from logits_lab.autocomplete import grounding_lines
from logits_lab.regeneration import Segment, check_acrostic, regenerate


def test_check_acrostic_prefix_and_completion():
    c = check_acrostic("Brave and strong\nOn the prowl", "bold", finished=False)
    assert c.initials == "bo" and c.fits and not c.complete

    c = check_acrostic("Brave\nOn\nLurking\nDaring to roam\n", "bold", finished=False)
    assert c.complete

    # last line not yet finished: not complete
    c = check_acrostic("Brave\nOn\nLurking\nDaring to", "bold", finished=False)
    assert c.initials == "bold" and not c.complete


def test_check_acrostic_rejects_wrong_letter_and_ignores_markup():
    assert not check_acrostic("Tenacious tiger\nEagerly", "bold", finished=False).fits
    assert check_acrostic("**Brave** tiger\n", "bold", finished=False).fits


def test_regenerate_picks_most_probable_fitting_segment():
    rounds = {
        "": [Segment("Tiger bright\n", -1.0, False), Segment("Brave tiger\n", -2.0, False)],
        "Brave tiger\n": [Segment("Over hills\nLurking\nDaring\n", -1.5, False)],
    }
    result = regenerate(lambda poem: rounds[poem], "bold", max_rounds=4)
    assert result.status == "complete"
    assert result.rounds[0].chosen.segment.text == "Brave tiger\n"
    assert [l[0] for l in result.poem_lines] == list("BOLD")


def test_regenerate_reports_stuck():
    result = regenerate(lambda poem: [Segment("Zebra\n", -1.0, False)], "bold")
    assert result.status == "stuck"


def test_grounding_lines_case_insensitive():
    doc = ["Friends, Romans, countrymen, lend me your ears;", "Et tu, Brute?"]
    assert grounding_lines(doc, "Lend") == [doc[0]]
