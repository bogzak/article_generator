import os
import re
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
