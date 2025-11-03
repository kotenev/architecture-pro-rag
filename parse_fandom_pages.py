#! /usr/bin/env python

import os
import json
import requests
from bs4 import BeautifulSoup
from faker import Faker

fake = Faker()

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

def process_fandom_pages(pages, knowledge_base_dir='knowledge_base'):
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir)

    terms_map = {}

    original_terms = list(pages.keys())
    for term in original_terms:
        replacement = fake.unique.first_name() + " " + fake.unique.last_name()
        terms_map[term] = replacement

    for original_name, url in pages.items():
        print(f"Processing {original_name}...")
        text = scrape_and_clean(url)
        if text:
            for term, replacement in terms_map.items():
                text = text.replace(term, replacement)

            file_name = f"{terms_map[original_name].replace(' ', '_')}.txt"
            with open(os.path.join(knowledge_base_dir, file_name), 'w', encoding='utf-8') as f:
                f.write(text)

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
