#! /usr/bin/env python
import os
import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from bs4 import BeautifulSoup
from faker import Faker
import pymorphy2

fake = Faker('ru_RU')
morph = pymorphy2.MorphAnalyzer()

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

def process_fandom_pages(pages, knowledge_base_dir='knowledge_base'):
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir)

    terms_map: Dict[str, str] = {}
    term_replacements: List[TermReplacement] = []
    used_phrases: set = set()

    for term in pages.keys():
        replacement = _generate_replacement_phrase(term, used_phrases)
        terms_map[term] = replacement
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
        with open(os.path.join(knowledge_base_dir, file_name), 'w', encoding='utf-8') as f:
            f.write(transformed_text)

    with open('terms_map.json', 'w', encoding='utf-8') as f:
        json.dump(terms_map, f, ensure_ascii=False, indent=4)

fandom_pages = {
    "Алконост": "https://mythological-creations.fandom.com/ru/wiki/%D0%90%D0%BB%D0%BA%D0%BE%D0%BD%D0%BE%D1%81%D1%82",
    "Бабай": "https://mythological-creations.fandom.com/ru/wiki/%D0%91%D0%B0%D0%B1%D0%B0%D0%B9",
    "Банник": "https://mythological-creations.fandom.com/ru/wiki/%D0%91%D0%B0%D0%BD%D0%BD%D0%B8%D0%BA",
    "Берегиня": "https://mythological-creations.fandom.com/ru/wiki/%D0%91%D0%B5%D1%80%D0%B5%D0%B3%D0%B8%D0%BD%D1%8F",
    "Блуд": "https://mythological-creations.fandom.com/ru/wiki/%D0%91%D0%BB%D1%83%D0%B4",
    "Василиск": "https://mythological-creations.fandom.com/ru/wiki/%D0%92%D0%B0%D1%81%D0%B8%D0%BB%D0%B8%D1%81%D0%BA",
    "Вий": "https://mythological-creations.fandom.com/ru/wiki/%D0%92%D0%B8%D0%B9",
    "Водяной": "https://mythological-creations.fandom.com/ru/wiki/%D0%92%D0%BE%D0%B4%D1%8F%D0%BD%D0%BE%D0%B9",
    "Вытьянка": "https://mythological-creations.fandom.com/ru/wiki/%D0%92%D1%8B%D1%82%D1%8C%D1%8F%D0%BD%D0%BA%D0%B0",
    "Гамаюн": "https://mythological-creations.fandom.com/ru/wiki/%D0%93%D0%B0%D0%BC%D0%B0%D1%8E%D0%BD",
    "Домовой": "https://mythological-creations.fandom.com/ru/wiki/%D0%94%D0%BE%D0%BC%D0%BE%D0%B2%D0%BE%D0%B9",
    "Ератники": "https://mythological-creations.fandom.com/ru/wiki/%D0%95%D1%80%D0%B0%D1%82%D0%BD%D0%B8%D0%BA%D0%B8",
    "Жар-птица": "https://mythological-creations.fandom.com/ru/wiki/%D0%96%D0%B0%D1%80-%D0%BF%D1%82%D0%B8%D1%86%D0%B0",
    "Жердяй": "https://mythological-creations.fandom.com/ru/wiki/%D0%96%D0%B5%D1%80%D0%B4%D1%8F%D0%B9",
    "Злыдень": "https://mythological-creations.fandom.com/ru/wiki/%D0%97%D0%BB%D1%8B%D0%B4%D0%B5%D0%BD%D1%8C",
    "Зыбочник": "https://mythological-creations.fandom.com/ru/wiki/%D0%97%D1%8B%D0%B1%D0%BE%D1%87%D0%BD%D0%B8%D0%BA",
    "Кикимора": "https://mythological-creations.fandom.com/ru/wiki/%D0%9A%D0%B8%D0%BA%D0%B8%D0%BC%D0%BE%D1%80%D0%B0",
    "Лебединые девы": "https://mythological-creations.fandom.com/ru/wiki/%D0%9B%D0%B5%D0%B1%D0%B5%D0%B4%D0%B8%D0%BD%D1%8B%D0%B5_%D0%B4%D0%B5%D0%B2%D1%8B",
    "Лесавка": "https://mythological-creations.fandom.com/ru/wiki/%D0%9B%D0%B5%D1%81%D0%B0%D0%B2%D0%BA%D0%B0",
    "Летавица": "https://mythological-creations.fandom.com/ru/wiki/%D0%9B%D0%B5%D1%82%D0%B0%D0%B2%D0%B8%D1%86%D0%B0",
    "Леший": "https://mythological-creations.fandom.com/ru/wiki/%D0%9B%D0%B5%D1%88%D0%B8%D0%B9",
    "Лоскотуха": "https://mythological-creations.fandom.com/ru/wiki/%D0%9B%D0%BE%D1%81%D0%BA%D0%BE%D1%82%D1%83%D1%85%D0%B0",
    "Мавка": "https://mythological-creations.fandom.com/ru/wiki/%D0%9C%D0%B0%D0%B2%D0%BA%D0%B0",
    "Мара": "https://mythological-creations.fandom.com/ru/wiki/%D0%9C%D0%B0%D1%80%D0%B0",
    "Моховик": "https://mythological-creations.fandom.com/ru/wiki/%D0%9C%D0%BE%D1%85%D0%BE%D0%B2%D0%B8%D0%BA",
    "Огненный змей": "https://mythological-creations.fandom.com/ru/wiki/%D0%9E%D0%B3%D0%BD%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B7%D0%BC%D0%B5%D0%B9",
    "Полевик": "https://mythological-creations.fandom.com/ru/wiki/%D0%9F%D0%BE%D0%BB%D0%B5%D0%B2%D0%B8%D0%BA",
    "Полудница": "https://mythological-creations.fandom.com/ru/wiki/%D0%9F%D0%BE%D0%BB%D1%83%D0%B4%D0%BD%D0%B8%D1%86%D0%B0",
    "Разрыв-трава": "https://mythological-creations.fandom.com/ru/wiki/%D0%A0%D0%B0%D0%B7%D1%80%D1%8B%D0%B2-%D1%82%D1%80%D0%B0%D0%B2%D0%B0",
    "Русалка": "https://mythological-creations.fandom.com/ru/wiki/%D0%A0%D1%83%D1%81%D0%B0%D0%BB%D0%BA%D0%B0",
    "Утопец": "https://mythological-creations.fandom.com/ru/wiki/%D0%A3%D1%82%D0%BE%D0%BF%D0%B5%D1%86",
    "Чёрт": "https://mythological-creations.fandom.com/ru/wiki/%D0%A7%D1%91%D1%80%D1%82",
    "Ырка": "https://mythological-creations.fandom.com/ru/wiki/%D0%AB%D1%80%D0%BA%D0%B0"
}

process_fandom_pages(fandom_pages)
