#! /usr/bin/env python
import argparse
import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

def _extract_page_title(soup: BeautifulSoup) -> str | None:
    heading = soup.find("h1")
    if heading and heading.get_text(strip=True):
        return heading.get_text(strip=True)

    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    return None

def scrape_and_clean(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.find("div", {"class": "mw-parser-output"})

        if content:
            for tag in content.find_all(["table", "sup", "div"]):
                tag.decompose()
                return content.get_text("\n", strip=True), _extract_page_title(soup)
            return None, None
    except requests.exceptions.RequestException as error:
        print(f"Ошибка скачивания {url}: {error}")
        return None, None


def _derive_filename(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] or "page"
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", slug).strip("_")
    return f"{sanitized or 'page'}.txt"


def parse_fandom_page(url: str, output_path: str | None = None, knowledge_base_dir: str = "knowledge_base") -> str:
    text, page_title = scrape_and_clean(url)
    if not text:
        raise SystemExit(f"Нет содержимого которое может быть распарсено {url}")

    if output_path is None:
        os.makedirs(knowledge_base_dir, exist_ok=True)
        filename_source = page_title or _derive_filename(url).removesuffix(".txt")
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", filename_source).strip("_")
        output_path = os.path.join(knowledge_base_dir, f"{sanitized or 'page'}.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсинг единичной Fandom-страницы в простой текст.")
    parser.add_argument("url", help="Полный URL Fandom-страницы для парсинга")
    parser.add_argument(
        "-o",
        "--output",
        help="Дополнительная опция для файла вывода. По-умолчанию: <knowledge_base>/incoming/<имя>.txt",
    )
    parser.add_argument(
        "-k",
        "--knowledge-base",
        default="knowledge_base/incoming",
        help="Каталог для файла вывода если --output не задано",
    )

    args = parser.parse_args()

    output_file = parse_fandom_page(args.url, args.output, args.knowledge_base)
    print(f"Распарсенное содержимое сохранено в {output_file}")


if __name__ == "__main__":
    main()
