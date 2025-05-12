from dotenv import dotenv_values
from openai import OpenAI
from utils import load_prompts
from pydantic import BaseModel
import json
import logging


logger = logging.getLogger(__name__)
config = dotenv_values(".env")


MODEL_ADVANCED = config["MODEL_ADVANCED"]
TEMPERATURE = float(config["TEMPERATURE"])
OPENAI_API_KEY = config["OPENAI_API_KEY"]
SYSTEM_PROMPT = load_prompts("prompts/system_prompt_EN.txt")

CLIENT = OpenAI(
    api_key=OPENAI_API_KEY
)


class GPTClient:
    """
    Хранит и аккумулирует всю переписку в self.conversation,
    чтобы модель имела контекст на каждом шаге.
    """

    def __init__(self, model: str = MODEL_ADVANCED, temperature: float = TEMPERATURE):
        self.model = model
        self.temperature = temperature
        # Начинаем разговор с некоего system_message, описывающего стиль и цели
        self.conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        
    def update_system_prompt(self, new_system_prompt: str) -> None:
        """
        Обновляет системный промпт в контексте разговора.
        Если у нас уже есть сообщения в разговоре, заменяем только первое системное сообщение.
        """
        if not new_system_prompt:
            logger.warning("Attempted to update system prompt with empty text. Using default.")
            return
            
        if self.conversation and self.conversation[0]["role"] == "system":
            # Заменяем существующий системный промпт
            self.conversation[0]["content"] = new_system_prompt
            logger.info("Updated existing system prompt")
        else:
            # Добавляем новый системный промпт в начало разговора
            self.conversation.insert(0, {"role": "system", "content": new_system_prompt})
            logger.info("Added new system prompt at the beginning of conversation")
        
        # Опциональная очистка контекста при обновлении промпта
        # Сохраняем только системное сообщение
        self.conversation = [self.conversation[0]]

    def chat(self, user_prompt: str) -> str:
        """
        Добавляет новое user-сообщение в conversation, делает запрос к OpenAI,
        затем добавляет ответ assistant в conversation и возвращает его.
        """
        # Добавляем сообщение пользователя
        self.conversation.append({"role": "user", "content": user_prompt})

        try:
            # Вызываем API
            response = CLIENT.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                temperature=self.temperature
            )

            # Текст ответа
            assistant_message = response.choices[0].message.content.strip()
            # Сохраняем в истории
            self.conversation.append({"role": "assistant", "content": assistant_message})

            return assistant_message

        except Exception as e:
            error_msg = f"Неожиданная ошибка при обращении к OpenAI API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def chat_with_format(self, user_prompt: str, response_format: BaseModel) -> BaseModel:
        """
        Similar to `chat`, but ensures the response adheres to a specific format using Pydantic models.
        """
        # Add user message to the conversation
        self.conversation.append({"role": "user", "content": user_prompt})

        try:
            # Call the OpenAI API
            response = CLIENT.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                temperature=self.temperature
            )

            # Extract the assistant's message
            assistant_message = response.choices[0].message.content.strip()

            # Save the assistant's response in the conversation
            self.conversation.append({"role": "assistant", "content": assistant_message})

            # Удаляем маркеры JSON, если они есть
            cleaned_message = assistant_message
            if "```json" in assistant_message:
                cleaned_message = assistant_message.replace("```json", "").replace("```", "").strip()

            # Parse the response into the specified format
            try:
                parsed_response = response_format.parse_raw(cleaned_message)
                return parsed_response
            except Exception as e:
                # Если не удалось распарсить напрямую, пробуем загрузить как JSON
                try:
                    data = json.loads(cleaned_message)
                    parsed_response = response_format.parse_obj(data)
                    return parsed_response
                except Exception as json_e:
                    logger.error(f"Failed to parse response: {e}, JSON error: {json_e}")
                    raise ValueError(f"Failed to parse response into the specified format: {e}")
                
        except RateLimitError:
            error_msg = "Превышен лимит запросов к OpenAI API. Пожалуйста, повторите попытку позже."
            logger.error(error_msg)
            raise Exception(error_msg)
        except APIError as e:
            error_msg = f"Ошибка OpenAI API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Неожиданная ошибка при обращении к OpenAI API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
