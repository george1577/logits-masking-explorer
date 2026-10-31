from logits_lab.scorers import alliteration_score, brand_score, find_phrases


def test_brand_score_counts_desired_minus_banned():
    text = "A keto protein shake that may cure cancer."
    assert brand_score(text, ["keto", "protein shake", "diet"], ["cure cancer"]) == 1


def test_phrases_match_whole_words_only():
    assert find_phrases("Please add your address", ["add"]) == ["add"]
    assert find_phrases("Additional address info", ["add"]) == []


def test_alliteration_score_is_case_insensitive():
    assert alliteration_score("Dainty donkeys dance down dusty roads", "d") == 5
    assert alliteration_score("Dainty donkeys dance down dusty roads", "D") == 5
