import re
import json
import logging
from pydantic import BaseModel, ValidationError

from utils import load_prompts
from gpt_client import GPTClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OutlineItem(BaseModel):
    title: str
    subtopics: list[str]

class OutlineResponse(BaseModel):
    outline: list[OutlineItem]


class ArticleGenerator:
    def __init__(self, gpt: GPTClient, language: str):
        """
        Инициализирует генератор статей с клиентом GPT и языком.
        """
        self.gpt = gpt
        self.language = language

    def generate_system_prompt(self, topic: str) -> str:
        """
        Генерирует специализированный системный промпт для конкретной темы статьи.
        """
        # Используем отдельный клиент GPT для генерации системного промпта
        # чтобы не мешать основному клиенту с его контекстом
        from gpt_client import GPTClient, MODEL_ADVANCED, TEMPERATURE
        
        prompt_template = load_prompts(f"prompts/system_prompt_generator_{self.language}.txt")
        user_prompt = prompt_template.format(topic=topic)
        
        try:
            # Создаем временный клиент для генерации системного промпта
            temp_client = GPTClient(model=MODEL_ADVANCED, temperature=TEMPERATURE)
            system_prompt = temp_client.chat(user_prompt)
            logger.info(f"Generated custom system prompt for topic: {topic}")
            return system_prompt
        except Exception as e:
            logger.error(f"Failed to generate system prompt: {e}")
            # В случае ошибки используем стандартный системный промпт
            return load_prompts(f"prompts/system_prompt_{self.language}.txt")

    @staticmethod
    def parse_outline_json(outline_text: str) -> list:
        """
        Парсит JSON-ответ с outline и возвращает список разделов.
        """
        # Убираем возможные ```json
        cleaned = re.sub(r"```json\s*", "", outline_text)
        cleaned = re.sub(r"```", "", cleaned).strip()

        try:
            # Проверяем, является ли строка валидным JSON
            data = json.loads(cleaned)
            
            # Пытаемся создать структурированный объект для валидации
            try:
                parsed = OutlineResponse.parse_obj(data)
                return [{"title": item.title, "subtopics": item.subtopics} for item in parsed.outline]
            except ValidationError as e:
                logger.warning("Validation error: %s", e)
                
            # Если валидация не удалась, пробуем получить данные напрямую
            outline = data.get("outline", [])
            if not isinstance(outline, list):
                logger.warning("'outline' is not a list.")
                return []
                
            return outline
        except json.JSONDecodeError as e:
            logger.warning("JSON parsing error: %s", e)
            return []

    def generate_outline(self, topic: str) -> str:
        """
        Generates a structured JSON outline for the given topic using OpenAI's Structured Outputs approach.
        """
        # Reading the prompt template
        template = load_prompts(f"prompts/outline_prompt_{self.language}.txt")
        user_prompt = template.format(topic=topic)

        try:
            # Using OpenAI's Structured Outputs to ensure JSON format
            response = self.gpt.chat_with_format(
                user_prompt,
                response_format=OutlineResponse
            )

            # Convert the structured response to a dictionary and then to JSON
            return json.dumps(response.dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to generate outline: %s", e)
            # В случае ошибки возвращаем JSON с пустым outline
            return json.dumps({"outline": []}, indent=2, ensure_ascii=False)

    def generate_introduction(self, topic: str) -> str:
        """
        Генерирует введение для статьи по заданной теме.
        """
        template = load_prompts(f"prompts/introduction_prompt_{self.language}.txt")
        user_prompt = template.format(topic=topic)
        try:
            intro_text = self.gpt.chat(user_prompt)
            logger.info(f"Introduction generated for topic: {topic}")
            return intro_text
        except Exception as e:
            logger.error(f"Failed to generate introduction: {e}")
            return "Не удалось сгенерировать введение."

    def generate_section_with_subtopics(self, topic: str, section_title: str, subtopics: list[str]) -> str:
        """
        Генерирует единый текст для 'section_title' (основной раздел) и
        всех подпунктов (subtopics) внутри.
        """
        # Можно оформить subtopics как список пунктов в prompt:
        bullets = "\n".join([f"- {s}" for s in subtopics])
        template = load_prompts(f"prompts/subtopics_prompt_{self.language}.txt")
        user_prompt = template.format(
            topic=topic,
            section_title=section_title,
            bullets=bullets
        )

        try:
            return self.gpt.chat(user_prompt)
        except Exception as e:
            logger.error("Failed to generate section with subtopics: %s", e)
            return f"Не удалось сгенерировать раздел '{section_title}'."

    def generate_conclusion(self, topic: str) -> str:
        """
        Генерирует заключение для статьи по заданной теме.
        """
        template = load_prompts(f"prompts/conclusion_prompt_{self.language}.txt")
        user_prompt = template.format(topic=topic)
        try:
            conclusion_text = self.gpt.chat(user_prompt)
            logger.info(f"Conclusion generated for topic: {topic}")
            return conclusion_text
        except Exception as e:
            logger.error(f"Failed to generate conclusion: {e}")
            return "Не удалось сгенерировать заключение."

    def generate_article(self, topic: str) -> str:
        """
        Генерирует полную статью, включая структуру, введение, основной текст 
        и заключение.
        """
        # 0) Генерируем специализированный системный промпт для темы
        system_prompt = self.generate_system_prompt(topic)
        # Обновляем системный промпт в GPT клиенте
        self.gpt.update_system_prompt(system_prompt)
        logger.info(f"Updated system prompt for topic: {topic}")
        
        # 1) Генерируем outline
        outline_raw = self.generate_outline(topic)
        sections = self.parse_outline_json(outline_raw)

        if not sections:
            logger.warning("No sections found in outline.")
            return f"# {topic}\n\nНе удалось сгенерировать статью по теме."

        # Флаги для определения, нужно ли генерировать введение и заключение
        with_introduction = True
        with_conclusion = True

        body_parts = []
        
        # 2) Генерируем введение
        introduction = ""
        if with_introduction:
            introduction = self.generate_introduction(topic)
        
        # 3) Фильтруем разделы: исключаем разделы введения и заключения из основного содержания
        main_sections = []
        for sec in sections:
            section_title = sec.get("title", "").lower()
            # Исключаем разделы с заголовками, похожими на "введение" и "заключение"
            if not any(keyword in section_title for keyword in ["введение", "вступление", "обзор", "заключение", "вывод", "итог"]):
                main_sections.append(sec)
            elif "введение" in section_title or "вступление" in section_title or "обзор" in section_title:
                # Используем подтемы из раздела введения для улучшения нашего введения
                if with_introduction:
                    logger.info(f"Found introduction section in outline: {section_title}")
            elif "заключение" in section_title or "вывод" in section_title or "итог" in section_title:
                # Используем подтемы из раздела заключения для улучшения нашего заключения
                if with_conclusion:
                    logger.info(f"Found conclusion section in outline: {section_title}")

        # 4) Генерируем текст для каждого основного раздела
        for sec in main_sections:
            section_title = sec.get("title", "Untitled Section")
            subtopics = sec.get("subtopics", [])

            # Генерируем одним вызовом
            section_text = self.generate_section_with_subtopics(topic, section_title, subtopics)

            # Добавляем заголовок
            body_parts.append(f"## {section_title}\n{section_text}\n")

        body_full = "\n".join(body_parts)

        # 5) Генерируем заключение
        conclusion = ""
        if with_conclusion:
            conclusion = self.generate_conclusion(topic)
            
        # 6) Собираем статью
        article_parts = [f"# {topic}\n\n"]
        
        if with_introduction:
            article_parts.append(f"## Введение\n{introduction}\n\n")
            
        article_parts.append(f"{body_full}\n\n")
        
        if with_conclusion:
            article_parts.append(f"## Заключение\n{conclusion}\n")
            
        return "".join(article_parts)
