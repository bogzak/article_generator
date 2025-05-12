import logging
import traceback

from gpt_client import GPTClient
from summarizer import Summarizer
from article_generator import ArticleGenerator
from utils import load_topics_from_file, save_article_to_file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


TOPICS_FILE = "files/topics.txt"
LANGUAGE = "RU"


def main() -> None:
    topics = load_topics_from_file(TOPICS_FILE)
    if not topics:
        logger.warning("No topics found in '%s'.", TOPICS_FILE)
        return

    logger.info(f"Loaded {len(topics)} topics for article generation")

    try:
        advanced_client = GPTClient()
        # Создаем summarizer только если он понадобится
        # summarizer = Summarizer()
        article_generator = ArticleGenerator(gpt=advanced_client, language=LANGUAGE)

        for i, topic in enumerate(topics, 1):
            logger.info(f"[{i}/{len(topics)}] Starting article generation for topic: {topic}")
            logger.info(f"Step 1: Generating custom system prompt for topic: {topic}")
            
            try:
                article_text = article_generator.generate_article(topic)
                save_article_to_file(article_text, topic)
                logger.info(f"Article successfully generated for topic: {topic}")
            except Exception as e:
                logger.error(f"Failed to generate article for topic '{topic}': {e}")
                logger.debug(traceback.format_exc())
                # Продолжаем со следующей темой
                continue

        logger.info("Article generation completed!")
    except Exception as e:
        logger.error(f"Critical error in article generation process: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
