import json
import os
import re
import logging

from gpt_client import GPTClient
from summarizer import Summarizer
from article_generator import ArticleGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------
# КЛАСС ArticleGenerator
# -------------------------------------------
class ArticleGenerator:
    def __init__(self, gpt: GPTClient):
        """
        Сюда можно передать только advanced_client, если summarizer временно не нужен.
        """
        self.gpt = gpt

    @staticmethod
    def parse_outline_json(outline_text: str):
        # Убираем возможные ```json
        cleaned = re.sub(r"```json\s*", "", outline_text)
        cleaned = re.sub(r"```", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON parsing error: %s", e)
            return []

        outline = data.get("outline", [])
        if not isinstance(outline, list):
            logger.warning("'outline' is not a list.")
            return []

        return outline  # Это список объектов { "title": str, "subtopics": list[str] }

    def generate_outline(self, topic: str) -> str:
        user_prompt = (
            f"""
        You are an assistant that must produce a valid JSON outline for an article on "{topic}".

        Return ONLY a JSON object with a single key "outline", whose value is an array. 
        Each element in this array is an object with:
        - "title": a string identifying a main section
        - "subtopics": an array of strings, each describing a subtopic

        No additional commentary or code fences. Example:

        {{
          "outline": [
            {{
              "title": "Section 1: Introduction",
              "subtopics": ["Background", "Purpose of Classification"]
            }},
            {{
              "title": "Section 2: Key Points",
              "subtopics": ["Point A", "Point B"]
            }}
          ]
        }}

        Now generate that JSON for the topic: {topic}.
        """
        )
        outline_raw = self.gpt.chat(user_prompt)
        logging.info(outline_raw)
        return outline_raw

    def generate_introduction(self, topic: str) -> str:
        user_prompt = (
            f"Topic: '{topic}'\n\n"
            "Write an Introduction that:\n"
            "1. Briefly hooks the reader with the importance or relevance of the topic.\n"
            "2. Transitions smoothly into what will be covered in the article.\n"
            "3. Avoids simply listing the outline sections verbatim.\n"
            "4. Maintains a professional yet accessible tone.\n"
            "Length guide: aim for ~150-200 words."
        )
        return self.gpt.chat(user_prompt)

    def generate_section_with_subtopics(self, topic: str, section_title: str, subtopics: list[str]) -> str:
        """
        Генерирует единый текст для 'section_title' (основной раздел) и
        всех подпунктов (subtopics) внутри.
        """
        # Можно оформить subtopics как список пунктов в prompt:
        bullets = "\n".join([f"- {s}" for s in subtopics])

        user_prompt = f"""
            Topic: {topic}

            Section title: {section_title}

            Please write a cohesive text covering the main section and the following subtopics:
            {bullets}

            Guidelines:
            - Merge all subtopics into one coherent piece of writing (not separate mini-chapters).
            - Aim for 300-500 words total.
            - Maintain clarity, a formal yet approachable tone.
            - Avoid repeating the Introduction verbatim, but do provide context where needed.
            """

        return self.gpt.chat(user_prompt)

    def generate_conclusion(self, topic: str) -> str:
        user_prompt = (
            f"Topic: '{topic}'\n\n"
            "Based on the above content, write a **Conclusion** that:\n"
            "1. Restates the core topic in a succinct way.\n"
            "2. Summarizes the major insights or takeaways.\n"
            "3. Gives a closing thought or call-to-action (if relevant).\n"
            "4. Maintains a professional tone, about ~100 words.\n"
        )
        return self.gpt.chat(user_prompt)

    @staticmethod
    def assemble_article(topic: str, outline: str, introduction: str, body_text: str, conclusion: str) -> str:
        return (
            f"# {topic}\n\n"
            f"## Outline\n{outline}\n\n"
            f"## Introduction\n{introduction}\n\n"
            f"## Main Body\n{body_text}\n\n"
            f"## Conclusion\n{conclusion}\n"
        )

    def generate_article(self, topic: str) -> str:
        # 1) Генерируем outline
        outline_raw = self.generate_outline(topic)
        sections = self.parse_outline_json(outline_raw)

        if not sections:
            logger.warning("No sections found in outline.")
            sections = []

        # 2) (Опционально) Генерируем Introduction
        introduction = self.generate_introduction(topic)  # если надо
        logger.info("Introduction: %s", introduction)

        # 3) Генерируем текст для каждого Section
        body_parts = []
        for sec in sections:
            section_title = sec.get("title", "Untitled Section")
            subtopics = sec.get("subtopics", [])

            # Генерируем одним вызовом
            section_text = self.generate_section_with_subtopics(topic, section_title, subtopics)

            # Можно добавить заголовок
            body_parts.append(f"## {section_title}\n{section_text}\n")

        body_full = "\n".join(body_parts)

        # 4) (Опционально) Генерируем Conclusion
        conclusion = self.generate_conclusion(topic)
        logger.info("Conclusion: %s", conclusion)

        # 5) Собираем всё
        article = (
            f"# {topic}\n\n"
            f"## Introduction\n{introduction}\n\n"
            f"{body_full}\n\n"
            f"## Conclusion\n{conclusion}\n"
        )
        return article


# -------------------------------------------
# ФУНКЦИИ ДЛЯ СОХРАНЕНИЯ И ЧТЕНИЯ ТЕМ
# -------------------------------------------
def save_article_to_file(article_text: str, topic: str, output_dir: str = "articles") -> None:
    if not article_text:
        logger.error("Article text is empty. Cannot save.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error("Error creating directory: %s", e)
        return

    safe_topic = re.sub(r'[\\/*?:"<>|, ]', "_", topic)
    filename = f"{safe_topic}__5.md"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(article_text)
        logger.info("Article on topic '%s' saved to: %s", topic, filepath)
    except Exception as e:
        logger.error("Error writing file: %s", e)


def load_topics_from_file(filepath: str) -> list:
    topics = []
    if not os.path.exists(filepath):
        logger.error("File '%s' not found. Returning empty list of topics.", filepath)
        return topics

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                topics.append(line)
    return topics


# -------------------------------------------
# MAIN
# -------------------------------------------
def main() -> None:
    topics_file = "files/topics.txt"
    topics = load_topics_from_file(topics_file)
    if not topics:
        logger.warning("No topics found in '%s'.", topics_file)
        return

    advanced_client = GPTClient()
    # summarizer = Summarizer(model=MODEL_SUMMARIZER, temperature=TEMPERATURE)
    article_generator = ArticleGenerator(advanced_client)

    for topic in topics:
        logger.info("Generating article for topic: %s", topic)
        article_text = article_generator.generate_article(topic)
        save_article_to_file(article_text, topic)

    logger.info("All articles have been generated successfully!")


if __name__ == "__main__":
    main()
