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
    except PermissionError:
        logger.error(f"Permission denied when creating directory: {output_dir}")
        return
    except Exception as e:
        logger.error(f"Error creating directory {output_dir}: {e}")
        return

    safe_topic = re.sub(r'[\\/*?:"<>|, ]', "_", topic)
    filename = f"{safe_topic}.md"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(article_text)
        logger.info(f"Article on topic '{topic}' saved to: {filepath}")
    except PermissionError:
        logger.error(f"Permission denied when writing to file: {filepath}")
    except Exception as e:
        logger.error(f"Error writing to file {filepath}: {e}")


def load_topics_from_file(filepath: str) -> list:
    topics = []
    if not os.path.exists(filepath):
        logger.error(f"File '{filepath}' not found. Returning empty list of topics.")
        return topics

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    topics.append(line)
        
        logger.info(f"Successfully loaded {len(topics)} topics from {filepath}")
        return topics
    except UnicodeDecodeError:
        logger.error(f"Error decoding file {filepath}. Try with a different encoding.")
        return topics
    except PermissionError:
        logger.error(f"Permission denied when reading file: {filepath}")
        return topics
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return topics


def load_prompts(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {filepath}")
        return ""
    except UnicodeDecodeError:
        logger.error(f"Error decoding prompt file {filepath}. Try with a different encoding.")
        return ""
    except Exception as e:
        logger.error(f"Error reading prompt file {filepath}: {e}")
        return ""
