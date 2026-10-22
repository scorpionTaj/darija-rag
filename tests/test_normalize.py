from darija_rag.normalize import normalize, tokenize


def test_normalize_folds_alef_variants_diacritics_and_tatweel():
    assert normalize("أَحْمَد إبراهيم آمال") == "احمد ابراهيم امال"
    assert normalize("المـــادة") == "الماده"


def test_normalize_folds_presentation_forms_and_digits():
    # "لا" ligature presentation form, as often produced by PDF extraction.
    assert normalize("ﻻ") == "لا"
    assert normalize("المادة ١٢") == "الماده 12"


def test_tokenize_strips_articles_and_stopwords():
    assert tokenize("في المحكمة والعقد") == ["محكمه", "عقد"]
    assert tokenize("Le contrat de travail") == ["contrat", "travail"]


def test_tokenize_matches_spelling_variants():
    assert tokenize("مدونة الأسرة") == tokenize("مدونه الاسره")


def test_tokenize_keeps_short_words_whole():
    # "الا" must not become a one-letter token after stripping "ال".
    assert tokenize("الا") == ["الا"]
