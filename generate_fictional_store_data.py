import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("store-data-generator")

# File paths
KB_JSON_FILE = Path(__file__).parent / "orion_store.json"
OUTPUT_FILE = Path(__file__).parent / "data/raw_data.txt"

def load_kb_from_json():
    try:
        with open(KB_JSON_FILE, "r", encoding="utf-8") as f:
            kb_entries = json.load(f)
        logger.info(f"Loaded {len(kb_entries)} entries from {KB_JSON_FILE}")
        return kb_entries
    except Exception as e:
        logger.error(f"Failed to load KB JSON: {e}")
        return []

def save_kb_to_text(kb_entries):
    formatted_entries = [
        f"Content from {entry['source']}:\n\n{entry['content']}"
        for entry in kb_entries
    ]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(formatted_entries))
    logger.info(f"Saved fictional store content to {OUTPUT_FILE}")

if __name__ == "__main__":
    kb_data = load_kb_from_json()
    if kb_data:
        save_kb_to_text(kb_data)
