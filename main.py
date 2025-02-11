import json
import os
import re

from openai import OpenAI
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Инициализация клиента OpenAI
CLIENT = OpenAI(
    api_key="api_key"
)

# Константы для моделей
MODEL_ADVANCED = "gpt-4o"  # Более продвинутая модель (для основной генерации)
MODEL_CHEAPER = "gpt-3.5-turbo"  # Дешевле, используется для summary

# Общие параметры генерации
TEMPERATURE = 0.7
SUMMARY_MAX_SENTENCES = 20


class GPTClient:
    """
    Класс для общения с OpenAI GPT-моделью.
    """
    def __init__(self, model: str = MODEL_ADVANCED, temperature: float = TEMPERATURE):
        self.model = model
        self.temperature = temperature

    def chat(self, context_messages, user_prompt):
        """
        Отправляет запрос к ChatCompletion.
        """
        messages = context_messages + [{"role": "user", "content": user_prompt}]
        response = CLIENT.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()


class Summarizer:
    """
    Класс для генерации кратких summary текста.
    """
    def __init__(self, model: str = MODEL_ADVANCED):
        self.model = model

    def summarize(self, text: str, max_sentences: int = SUMMARY_MAX_SENTENCES) -> str:
        """
        Создаёт краткое резюме (summary) для переданного текста.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant that creates concise summaries of user-provided text. "
                "Your output should be in English, direct to the point, and limited to a maximum number of sentences."
            )
        }

        prompt = (
            f"Please summarize the following text in **no more than {max_sentences} sentences**. "
            f"Focus on the most essential points, omit minor details, and maintain clarity.\n\n"
            f"Text:\n{text}\n\n"
            "Make sure the final summary is coherent and self-contained."
        )
        response = CLIENT.chat.completions.create(
            model=self.model,
            messages=[system_message, {"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content.strip()


class OutlineParser:
    """
    Класс для генерации плана статьи и его парсинга из JSON-формата.
    """
    def __init__(self, gpt_client: GPTClient):
        self.gpt_client = gpt_client

    def generate_outline(self, topic: str) -> str:
        """
        Генерирует подробный план (outline) статьи в JSON-формате.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an experienced English-writing article creator. "
                "You produce structured, comprehensive outlines in JSON format."
            )
        }
        context_messages = [system_message]
        prompt = (
            f"Topic: '{topic}'\n\n"
            "Your task:\n"
            "1. Propose a clear, logical structure of sections and subsections relevant to this topic.\n"
            "2. Return the outline in JSON, with two keys:\n"
            "   - 'outline': A concise textual summary of each main section.\n"
            "   - 'subtopics': An array listing each subtopic (as bullet points) in the order they should appear.\n"
            "3. The final JSON must be valid (parsable) and must reflect a well-organized approach.\n\n"
            "Example:\n"
            "{\n"
            "  \"outline\": \"1. Intro ... 2. Key aspects ... 3. Summary ...\",\n"
            "  \"subtopics\": [\"Definition\", \"Importance\", \"Examples\"]\n"
            "}\n\n"
            "Focus on ensuring each subtopic is relevant and helps develop a well-rounded article."
        )
        return self.gpt_client.chat(context_messages, prompt)

    @staticmethod
    def parse_outline_json(outline_text: str):
        """
        Пытается распарсить строку outline_text как JSON.
        """
        try:
            data = json.loads(outline_text)
            outline = data.get("outline", "")
            subtopics = data.get("subtopics", [])
            return outline, subtopics
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON. Using fallback.")
            return outline_text, []


class ArticleGenerator:
    """
    Генерирует различные части статьи.
    """
    def __init__(self, advanced_client: GPTClient, summarizer: Summarizer):
        self.gpt = advanced_client
        self.summarizer = summarizer

    def generate_introduction(self, topic: str, outline_text: str) -> str:
        """
        Генерирует введение (introduction).
        """
        system_message = {
            "role": "system",
            "content": (
                "You are an advanced English-writing AI specialized in crafting engaging introductions."
            )
        }
        prompt = (
            f"Topic: '{topic}'\n\n"
            f"Outline overview:\n{outline_text}\n\n"
            "Write an Introduction that:\n"
            "1. Briefly hooks the reader with the importance or relevance of the topic.\n"
            "2. Transitions smoothly into what will be covered in the article.\n"
            "3. Avoids simply listing the outline sections verbatim.\n"
            "4. Maintains a professional yet accessible tone.\n"
            "Length guide: aim for ~150-200 words."
        )
        return self.gpt.chat([system_message], prompt)

    def generate_body_part(self, topic: str, subtopic: str, accumulated_text: str) -> str:
        """
        Генерирует часть основного блока статьи по конкретной подпункте.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a proficient English article writer. You expand on specific subtopics in detail, "
                "providing coherent, well-researched paragraphs."
            )
        }
        prompt = (
            f"Article topic: '{topic}'\n\n"
            f"Accumulated text or summary so far:\n{accumulated_text}\n\n"
            f"Subtopic to address: '{subtopic}'\n\n"
            "Please write a detailed section (one or more paragraphs) covering:\n"
            "1. Explanation or definition (if relevant) of the subtopic.\n"
            "2. Key points, facts, or examples.\n"
            "3. Potential challenges, advantages, or nuances.\n\n"
            "Maintain clarity and cohesiveness. Avoid unnecessary repetition of the introduction. "
            "Use a formal yet approachable tone, and assume the reader is not an expert on the topic.\n"
            "Aim for 150-250 words."
        )
        return self.gpt.chat([system_message], prompt)

    def generate_conclusion(self, topic: str, accumulated_text: str) -> str:
        """
        Генерирует заключение (conclusion).
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a skilled English article writer specializing in writing concise, effective conclusions."
            )
        }
        prompt = (
            f"Topic: '{topic}'\n\n"
            f"Current accumulated text or summary:\n{accumulated_text}\n\n"
            "Based on the above content, write a **Conclusion** that:\n"
            "1. Restates the core topic in a succinct way.\n"
            "2. Summarizes the major insights or takeaways.\n"
            "3. Gives a closing thought or call-to-action (if relevant).\n"
            "4. Maintains a professional tone, about ~100 words.\n"
        )
        return self.gpt.chat([system_message], prompt)

    @staticmethod
    def assemble_article(topic: str, outline: str, introduction: str, body_text: str, conclusion: str) -> str:
        """
        Склеивает всю статью в формате Markdown.
        """
        return (
            f"# {topic}\n\n"
            f"## Outline\n{outline}\n\n"
            f"## Introduction\n{introduction}\n\n"
            f"## Main Body\n{body_text}\n\n"
            f"## Conclusion\n{conclusion}\n"
        )

    def generate_article(self, topic: str) -> str:
        """
        Последовательная генерация полной статьи:
          1. Outline
          2. Introduction
          3. Body (каждый subtopic)
          4. Conclusion
        """
        # 1) Генерация outline
        outline_parser = OutlineParser(self.gpt)
        outline_raw = outline_parser.generate_outline(topic)
        parsed_outline, subtopics = outline_parser.parse_outline_json(outline_raw)
        if not subtopics:
            subtopics = ["Part 1", "Part 2"]
            logger.warning("Could not parse subtopics. Using fallback.")

        # 2) Введение
        introduction = self.generate_introduction(topic, parsed_outline)
        logger.info("Introduction: %s", introduction)

        # Поддержание "накопленного" текста для контекста
        accumulated_text = f"Outline:\n{parsed_outline}\n\nIntroduction:\n{introduction}"

        # 3) Основная часть по подпунктам
        body_parts = []
        for i, subtopic in enumerate(subtopics, start=1):
            if i % 2 == 0:
                summary = self.summarizer.summarize(accumulated_text)
                accumulated_text = summary
            part_text = self.generate_body_part(topic, subtopic, accumulated_text)
            body_parts.append(part_text)
            accumulated_text += f"\n\n=== Subtopic '{subtopic}' ===\n{part_text}"
        body_full = "\n\n".join(body_parts)

        # 4) Заключение
        final_summary = self.summarizer.summarize(accumulated_text)
        conclusion = self.generate_conclusion(topic, final_summary)
        logger.info("Conclusion: %s", conclusion)

        # 5) Сборка
        article = self.assemble_article(
            topic=topic,
            outline=parsed_outline,
            introduction=introduction,
            body_text=body_full,
            conclusion=conclusion
        )
        return article


def save_article_to_file(
    article_text: str,
    topic: str,
    output_dir: str = "articles"
) -> None:
    """
    Сохраняет статью в .md-файл для дальнейшего использования.
    Args:
        article_text (str): Полный текст статьи.
        topic (str): Тема статьи (используется в названии файла).
        output_dir (str): Папка для сохранения.
    """
    if not article_text:
        logger.error("Article text is empty. Cannot save.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error("Error creating directory: %s", e)
        return

    # Очистка названия файла от недопустимых символов
    safe_topic = re.sub(r'[\\/*?:"<>|]', "_", topic)  # Заменяем запрещённые символы на "_"
    filename = f"{safe_topic}.md"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(article_text)
        logger.info("Article on topic '%s' saved to: %s", topic, filepath)
    except Exception as e:
        logger.error("Error writing file: %s", e)


def load_topics_from_file(filepath: str) -> list:
    """
    Загружает темы статей из указанного файла.
    """
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


def main() -> None:
    """
    Точка входа: генерирует статьи по списку тем и сохраняет их в файлы .md.
    """
    topics_file = "files/topics.txt"
    topics = load_topics_from_file(topics_file)
    if not topics:
        logger.warning("No topics found. Please ensure the file has at least one topic.")
        return

    advanced_client = GPTClient(model=MODEL_ADVANCED, temperature=TEMPERATURE)
    summarizer = Summarizer(model=MODEL_ADVANCED)
    article_generator = ArticleGenerator(advanced_client, summarizer)

    for topic in topics:
        logger.info("Generating article for topic: %s", topic)
        article_text = article_generator.generate_article(topic)
        save_article_to_file(article_text, topic)

    logger.info("All articles have been generated successfully!")


if __name__ == "__main__":
    main()
