from dotenv import dotenv_values
from openai import OpenAI


config = dotenv_values(".env")


MODEL_ADVANCED = config["MODEL_ADVANCED"]
TEMPERATURE = config["TEMPERATURE"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]


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
                "content": (
                    "You are a professional English-language article writer. "
                    "You create well-structured, detailed, and coherent articles. "
                    "Use a clear, formal (but approachable) tone, and expand on key ideas when asked."
                )
            }
        ]

    def chat(self, user_prompt: str) -> str:
        """
        Добавляет новое user-сообщение в conversation, делает запрос к OpenAI,
        затем добавляет ответ assistant в conversation и возвращает его.
        """
        # Добавляем сообщение пользователя
        self.conversation.append({"role": "user", "content": user_prompt})

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
