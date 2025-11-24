#! /usr/bin/env python
import inspect
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from bs4 import BeautifulSoup
from faker import Faker
import pymorphy2

if not hasattr(inspect, "getargspec"):
    from collections import namedtuple

    ArgSpec = namedtuple(
        "ArgSpec", ["args", "varargs", "keywords", "defaults"]
    )

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec  # type: ignore[attr-defined]

fake = Faker('ru_RU')
morph = pymorphy2.MorphAnalyzer()

BASE_DIR = Path(__file__).resolve().parent
TERMS_MAP_PATH = BASE_DIR / "terms_map.json"
FANDOM_PAGES_PATH = BASE_DIR / "fandom_pages.json"

WORD_TOKEN_RE = re.compile(r'[А-Яа-яЁё-]+')
TOKENIZER_RE = re.compile(r'[А-Яа-яЁё-]+|[^А-Яа-яЁё-]+')
INFLECTION_GRAMMEMES = {
    'nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct',
    'gen2', 'acc2', 'loc2', 'sing', 'plur', 'masc', 'femn', 'neut'
}


def _select_parse(word: str):
    parses = morph.parse(word)
    lower = word.lower()
    for parse in parses:
        if parse.normal_form == lower:
            return parse
    return parses[0]


def _adjust_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.capitalize()
    if source.islower():
        return target.lower()
    return target


class TermReplacement:
    def __init__(self, original: str, replacement: str):
        self.original = original
        self.replacement = replacement
        self.original_words = WORD_TOKEN_RE.findall(original)
        self.replacement_words = WORD_TOKEN_RE.findall(replacement)

        if not self.original_words:
            raise ValueError(f"Term '{original}' does not contain replaceable words")

        if len(self.replacement_words) != len(self.original_words):
            raise ValueError(
                "Replacement must have the same number of words as original term. "
                f"Original '{original}', replacement '{replacement}'"
            )

        self.original_parses = [_select_parse(word) for word in self.original_words]
        self.original_lemmas = [parse.normal_form for parse in self.original_parses]
        self.replacement_parses = [_select_parse(word) for word in self.replacement_words]

    @property
    def first_lemma(self) -> str:
        return self.original_lemmas[0]

    def _match_word(self, idx: int, token: str):
        if not WORD_TOKEN_RE.fullmatch(token):
            return None
        for parse in morph.parse(token):
            if parse.normal_form == self.original_lemmas[idx]:
                return parse
        return None

    def try_match(self, tokens: Sequence[str], start_index: int) -> Optional[Tuple[int, List[str]]]:
        matched_parses = []
        separators: List[str] = []
        j = start_index

        for idx in range(len(self.original_words)):
            if j >= len(tokens):
                return None

            token = tokens[j]
            parse = self._match_word(idx, token)
            if parse is None:
                return None

            matched_parses.append((token, parse))
            j += 1

            if idx < len(self.original_words) - 1:
                sep = ''
                while j < len(tokens) and not WORD_TOKEN_RE.fullmatch(tokens[j]):
                    sep += tokens[j]
                    j += 1
                separators.append(sep)

        replacement_tokens: List[str] = []
        for idx, (original_token, original_parse) in enumerate(matched_parses):
            grammemes = set(original_parse.tag.grammemes) & INFLECTION_GRAMMEMES
            replacement_parse = self.replacement_parses[idx]
            inflected = replacement_parse.inflect(grammemes) if grammemes else replacement_parse
            if inflected is None:
                inflected = replacement_parse
            replacement_word = _adjust_case(original_token, inflected.word)
            replacement_tokens.append(replacement_word)
            if idx < len(separators):
                replacement_tokens.append(separators[idx])

        return j, replacement_tokens


def replace_terms(text: str, replacements: List[TermReplacement]) -> str:
    tokens = TOKENIZER_RE.findall(text)
    index = 0
    first_word_map: Dict[str, List[TermReplacement]] = {}
    for repl in replacements:
        first_word_map.setdefault(repl.first_lemma, []).append(repl)

    while index < len(tokens):
        token = tokens[index]
        if not WORD_TOKEN_RE.fullmatch(token):
            index += 1
            continue

        possible_terms: List[TermReplacement] = []
        for parse in morph.parse(token):
            possible_terms.extend(first_word_map.get(parse.normal_form, []))
        if not possible_terms:
            index += 1
            continue
        possible_terms = list(dict.fromkeys(possible_terms))
        possible_terms.sort(key=lambda t: len(t.original_words), reverse=True)

        replaced = False
        for term in possible_terms:
            result = term.try_match(tokens, index)
            if result is None:
                continue
            end_index, replacement_tokens = result
            tokens[index:end_index] = replacement_tokens
            index += len(replacement_tokens)
            replaced = True
            break

        if not replaced:
            index += 1

    return ''.join(tokens)

def scrape_and_clean(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.find('div', {'class': 'mw-parser-output'})

        if content:
            for tag in content.find_all(['table', 'sup', 'div']):
                tag.decompose()
            return content.get_text('\n', strip=True)

        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def _generate_replacement_phrase(term: str, used_phrases: set) -> str:
    word_count = max(1, len(WORD_TOKEN_RE.findall(term)))

    def unique_first_name() -> str:
        return fake.unique.first_name()

    phrase: Optional[str] = None
    attempts = 0
    while phrase is None or phrase in used_phrases:
        attempts += 1
        if attempts > 100:
            fake.unique.clear()
            attempts = 0

        if word_count == 1:
            phrase = unique_first_name()
        elif word_count == 2:
            phrase = f"{unique_first_name()} {fake.unique.last_name()}"
        else:
            words = [unique_first_name() for _ in range(word_count)]
            phrase = ' '.join(words)

    used_phrases.add(phrase)
    return phrase

def _load_terms_map(path: Path) -> Dict[str, str]:
    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _load_fandom_pages(path: Path) -> Dict[str, str]:
    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def process_fandom_pages(pages, knowledge_base_dir='knowledge_base'):
    knowledge_base_path = Path(knowledge_base_dir)
    knowledge_base_path.mkdir(exist_ok=True)

    terms_map: Dict[str, str] = _load_terms_map(TERMS_MAP_PATH)
    term_replacements: List[TermReplacement] = []
    used_phrases: set = set(terms_map.values())
    terms_map_updated = False

    for term, replacement in terms_map.items():
        try:
            term_replacements.append(TermReplacement(term, replacement))
        except ValueError as error:
            print(f"Skipping term '{term}': {error}")

    for term in pages.keys():
        if term in terms_map:
            continue

        replacement = _generate_replacement_phrase(term, used_phrases)
        terms_map[term] = replacement
        terms_map_updated = True
        try:
            term_replacements.append(TermReplacement(term, replacement))
        except ValueError as error:
            print(f"Skipping term '{term}': {error}")

    for original_name, url in pages.items():
        print(f"Processing {original_name}...")
        text = scrape_and_clean(url)

        if not text:
            continue

        transformed_text = replace_terms(text, term_replacements)

        file_name = f"{terms_map[original_name].replace(' ', '_')}.txt"
        with open(knowledge_base_path / file_name, 'w', encoding='utf-8') as f:
            f.write(transformed_text)

if __name__ == "__main__":
    fandom_pages = _load_fandom_pages(FANDOM_PAGES_PATH)
    process_fandom_pages(fandom_pages)

