import unittest
from unittest.mock import MagicMock
from article_generator import ArticleGenerator
from gpt_client import GPTClient
from summarizer import Summarizer
from utils import load_prompts, save_article_to_file

class TestArticleGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_gpt = MagicMock(spec=GPTClient)
        self.generator = ArticleGenerator(gpt=self.mock_gpt, language="RU")

    def test_generate_outline(self):
        self.mock_gpt.chat_with_format.return_value = {
            "outline": [
                {"title": "Введение", "subtopics": ["История", "Цель"]},
                {"title": "Основная часть", "subtopics": ["Тема A", "Тема B"]}
            ]
        }
        outline = self.generator.generate_outline("Тестовая тема")
        self.assertIn("outline", outline)

class TestGPTClient(unittest.TestCase):
    def setUp(self):
        self.client = GPTClient()

    def test_chat(self):
        self.client.conversation = []
        self.client.chat = MagicMock(return_value="Ответ")
        response = self.client.chat("Привет")
        self.assertEqual(response, "Ответ")

class TestUtils(unittest.TestCase):
    def test_load_prompts(self):
        content = load_prompts("prompts/outline_prompt_RU.txt")
        self.assertTrue(len(content) > 0)

    def test_save_article_to_file(self):
        save_article_to_file("Тестовый текст", "Тестовая тема", output_dir="test_articles")
        # Проверяем, что файл создан
        import os
        self.assertTrue(os.path.exists("test_articles/Тестовая_тема.md"))
        os.remove("test_articles/Тестовая_тема.md")

if __name__ == "__main__":
    unittest.main()