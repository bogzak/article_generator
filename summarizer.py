from dotenv import dotenv_values
from openai import OpenAI


config = dotenv_values(".env")


MODEL_SUMMARIZER = config["MODEL_SUMMARIZER"]
TEMPERATURE = config["TEMPERATURE"]
SUMMARY_MAX_SENTENCES = config["SUMMARY_MAX_SENTENCES"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]


CLIENT = OpenAI(
    api_key=OPENAI_API_KEY
)


class Summarizer:
    """
    Класс для генерации краткого summary (поддерживает собственное хранение контекста при желании).
    """

    def __init__(self, model: str = MODEL_SUMMARIZER, temperature: float = TEMPERATURE):
        self.model = model
        self.temperature = temperature
        # Отдельный контекст; можно сделать иначе, но, как правило, Summarizer —
        # отдельный, более простой сценарий
        self.conversation = [
            {
                "role": "system",
                "content": (
                    "You are a concise summarizer. Your task is to read long texts "
                    "and provide short, clear summaries in English."
                )
            }
        ]

    def summarize(self, text: str, max_sentences: int = SUMMARY_MAX_SENTENCES) -> str:
        """
        Генерирует краткое summary исходного текста (до max_sentences предложений).
        """
        prompt = (
            f"Please summarize the following text in no more than {max_sentences} sentences:\n\n{text}"
        )

        self.conversation.append({"role": "user", "content": prompt})

        response = CLIENT.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=self.temperature
        )

        summary = response.choices[0].message.content.strip()
        # Сохраняем ответ
        self.conversation.append({"role": "assistant", "content": summary})
        return summary
