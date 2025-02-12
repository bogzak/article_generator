import logging

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


def main() -> None:
    topics = load_topics_from_file(TOPICS_FILE)
    if not topics:
        logger.warning("No topics found in '%s'.", TOPICS_FILE)
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
