"""Project paths. Override with environment variables if your layout differs."""
import os

# repository root (folder that contains src/, runs/, cache/, paper/)
ROOT = os.environ.get("FF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/").rstrip("/") + "/"
# AIForge-Doc v2 full package (must contain TrainingSet/, TestingSet/ with authentic/, metadata.jsonl, source_mapping.csv)
DATA = os.environ.get("AIFORGE_DIR", ROOT + "data/AIForge-Doc-v2.0").replace("\\", "/").rstrip("/") + "/"
