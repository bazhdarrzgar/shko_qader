"""Heuristic labeling: morphology of drug/disease names, dosage context, function words."""

from __future__ import annotations

import re
from typing import Sequence

from common.schema import OUTSIDE
from step2_preprocessing.stemmer import KurdishStemmer
from step2_preprocessing.tokenizer import is_number, is_punctuation
from step3_weak_supervision.labeling_functions.base import TokenLabelingFunction, TokenVote

DRUG_SUFFIXES_CKB: tuple[str, ...] = (
    "سیلین", "مایسین", "سایکلین", "پیسین", "ازۆل", "پرۆفین", "ستاتین", "پریل", "ۆلۆل",
    "تامۆڵ", "تامۆل", "دیپین", "سولۆسین", "نیازید", "ۆکسامین", "فۆرمین", "سۆلین", "سولین",
    "ازیپام", "سارتان", "پرازۆل", "فلۆکساسین",
)
DRUG_PATTERN_LATIN = re.compile(
    r"[a-z]+(?:mycin|cillin|cycline|azole|profen|statin|pril|olol|sartan|dipine|formin|"
    r"amol|azepam|prazole|floxacin|mab|vir|sulin)"
)

DISEASE_SUFFIXES_CKB: tuple[str, ...] = ("یمیا", "ایتیس", "ایتس", "ۆسیس", "ۆما", "پاسی")
DISEASE_PATTERN_LATIN = re.compile(r"[a-z]+(?:itis|osis|emia|aemia|oma|pathy|algia)")

DOSAGE_UNITS = frozenset({
    "mg", "ml", "g", "mcg", "iu", "ملگم", "ملیگرام", "میلیگرام", "گرام", "مل", "میلیلیتر", "یەکە", "دانە", "حەب",
})


class SuffixMorphologyLF(TokenLabelingFunction):
    """Votes `label` for tokens whose (stripped) form ends with a characteristic suffix."""

    def __init__(
        self,
        name: str,
        label: str,
        kurdish_suffixes: tuple[str, ...],
        latin_pattern: re.Pattern[str],
        stemmer: KurdishStemmer,
        min_length: int = 6,
    ) -> None:
        super().__init__(name)
        self.label = label
        self.kurdish_suffixes = kurdish_suffixes
        self.latin_pattern = latin_pattern
        self.stemmer = stemmer
        self.min_length = min_length

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        votes: list[TokenVote] = [None] * len(tokens)
        for index, token in enumerate(tokens):
            if len(token) < self.min_length:
                continue
            if token.isascii():
                if self.latin_pattern.fullmatch(token):
                    votes[index] = self.label
            elif any(form.endswith(self.kurdish_suffixes) for form in self.stemmer.candidates(token)):
                votes[index] = self.label
        return votes


class DosageContextLF(TokenLabelingFunction):
    """`<drug> <number> <unit>` -> the word before the dose is a TREATMENT."""

    def __init__(self, name: str = "heuristic_dosage_context") -> None:
        super().__init__(name)

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        votes: list[TokenVote] = [None] * len(tokens)
        for index in range(len(tokens) - 2):
            token = tokens[index]
            if is_number(token) or is_punctuation(token):
                continue
            if is_number(tokens[index + 1]) and tokens[index + 2] in DOSAGE_UNITS:
                votes[index] = "TREATMENT"
        return votes


class FunctionWordLF(TokenLabelingFunction):
    """Function words, punctuation and numbers are never entities."""

    def __init__(self, stopwords: frozenset[str], name: str = "heuristic_function_words") -> None:
        super().__init__(name)
        self.stopwords = stopwords

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        return [
            OUTSIDE if token in self.stopwords or is_punctuation(token) or is_number(token) else None
            for token in tokens
        ]


def build_heuristic_lfs(stemmer: KurdishStemmer, stopwords: frozenset[str]) -> list[TokenLabelingFunction]:
    return [
        SuffixMorphologyLF(
            "heuristic_drug_morphology", "TREATMENT", DRUG_SUFFIXES_CKB, DRUG_PATTERN_LATIN, stemmer,
        ),
        SuffixMorphologyLF(
            "heuristic_disease_morphology", "DISEASE", DISEASE_SUFFIXES_CKB, DISEASE_PATTERN_LATIN, stemmer,
        ),
        DosageContextLF(),
        FunctionWordLF(stopwords),
    ]
