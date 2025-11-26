#! /usr/bin/env python
import os
import argparse
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

class VectorSearchTester:
    def __init__(self, 
                 index_path: str = "faiss_index",
                 model_name: str = "intfloat/Multilingual-E5-large"):
        self.index_path = index_path
        self.model_name = model_name

        print(f"Загрузка модели эмбеддингов: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.load_index()
    
    def load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Индекс не найден: {self.index_path}")
        
        print(f"Загрузка индекса из {self.index_path}")
        self.vectorstore = FAISS.load_local(
            self.index_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Индекс успешно загружен")
    
    def search(self, query: str, k: int = 5, threshold: float = 0.7) -> List:
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        filtered_results = []
        for doc, score in results:
            relevance = 1 - score
            if relevance >= threshold:
                filtered_results.append({
                    'document': doc,
                    'relevance': relevance,
                    'score': score
                })
        
        return filtered_results
    
    def display_results(self, query: str, results: List, verbose: bool = True):
        print(f"\n{'=' * 60}")
        print(f"ЗАПРОС: {query}")
        print(f"{'=' * 60}")
        
        if not results:
            print("Релевантные документы не найдены")
            return
        
        print(f"Найдено релевантных чанков: {len(results)}\n")
        
        for i, result in enumerate(results, 1):
            doc = result['document']
            relevance = result['relevance']
            
            print(f"{i}.Документ: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   Релевантность: {relevance:.2%}")
            print(f"   Чанк ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"   Источник: {doc.metadata.get('source', 'N/A')}")
            
            if verbose:
                text_preview = doc.page_content[:300]
                if len(doc.page_content) > 300:
                    text_preview += "..."
                print(f"   Текст:\n      {text_preview}\n")
            
            print("-" * 60)
    
    def test_golden_queries(self):
        golden_queries = [
            {
                "query": "Кто такой Якуб и как он относится к лошадям хозяев?",
                "expected_doc": None,
                "description": "Вопрос о домовом-защитнике усадьбы"
            },
            {
                "query": "Что происходит с теми, кто нарушает полуденный покой Ярополка?",
                "expected_doc": None,
                "description": "Вопрос о духе полуденного зноя"
            },
            {
                "query": "Почему Авдея считают вещим певцом и предвестником бед?",
                "expected_doc": None,
                "description": "Вопрос о пророческой птице-гамаюне"
            },
            {
                "query": "Как задобрить Самуила перед походом в лес?",
                "expected_doc": None,
                "description": "Общий вопрос о встрече с лесным хозяином"
            },
            {
                "query": "Какие обереги помогают против наваждений Дементия?",
                "expected_doc": None,
                "description": "Вопрос о защите от чар огненного духа"
            }
        ]
        
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ НА ЗОЛОТОМ НАБОРЕ ВОПРОСОВ")
        print("=" * 60)
        
        results_summary = []
        
        for test_case in golden_queries:
            query = test_case["query"]
            expected = test_case["expected_doc"]
            description = test_case["description"]
            
            print(f"\n Тест: {description}")
            print(f"   Запрос: {query}")
            print(f"   Ожидается: {expected if expected else 'Любой релевантный документ'}")
            
            results = self.search(query, k=3, threshold=0.5)
            
            if results:
                top_result = results[0]['document']
                found_doc = top_result.metadata.get('filename', 'Unknown')
                relevance = results[0]['relevance']
                
                if expected:
                    success = found_doc == expected
                else:
                    success = True
                
                status = " УСПЕХ" if success else "  НЕТОЧНО"
                print(f"   Результат: {status}")
                print(f"   Найден: {found_doc} (релевантность: {relevance:.2%})")
                
                results_summary.append({
                    'query': query,
                    'success': success,
                    'found': found_doc,
                    'relevance': relevance
                })
            else:
                print(f"   Результат:  НЕ НАЙДЕНО")
                results_summary.append({
                    'query': query,
                    'success': False,
                    'found': None,
                    'relevance': 0
                })

        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 60)
        
        successful = sum(1 for r in results_summary if r['success'])
        total = len(results_summary)
        accuracy = successful / total * 100
        
        print(f"Успешных тестов: {successful}/{total} ({accuracy:.1f}%)")
        print(f"Средняя релевантность: {sum(r['relevance'] for r in results_summary) / total:.2%}")
        
        return results_summary
    
    def interactive_search(self):
        print("\n" + "=" * 60)
        print("ИНТЕРАКТИВНЫЙ ПОИСК")
        print("=" * 60)
        print("Введите запросы для поиска (или 'выход' для завершения)")
        print("Формат: <запрос> [количество результатов]")
        print("Пример: Как задобрить Самуила? 3")
        
        while True:
            try:
                user_input = input("\n🔍 Запрос: ").strip()
                
                if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("Завершение работы...")
                    break
                
                if not user_input:
                    continue

                parts = user_input.rsplit(' ', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    query = parts[0]
                    k = int(parts[1])
                else:
                    query = user_input
                    k = 3

                results = self.search(query, k=k, threshold=0.3)
                self.display_results(query, results)
                
            except KeyboardInterrupt:
                print("\n\nПрервано пользователем")
                break
            except Exception as e:
                print(f" Ошибка: {e}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Тестирование векторного поиска')
    parser.add_argument('--index', default='faiss_index', help='Путь к индексу')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Модель эмбеддингов')
    parser.add_argument('--query', help='Одиночный запрос для поиска')
    parser.add_argument('--k', type=int, default=3, help='Количество результатов')
    parser.add_argument('--test', action='store_true', 
                        help='Запустить тестирование на золотом наборе')
    parser.add_argument('--interactive', action='store_true', 
                        help='Интерактивный режим')
    
    args = parser.parse_args()

    tester = VectorSearchTester(args.index, args.model)
    
    if args.test:
        tester.test_golden_queries()
    elif args.query:
        results = tester.search(args.query, k=args.k)
        tester.display_results(args.query, results)
    elif args.interactive:
        tester.interactive_search()
    else:
        tester.interactive_search()


if __name__ == "__main__":
    main()