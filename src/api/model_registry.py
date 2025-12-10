import joblib
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, model_paths: Dict[str, str]):
        self.model_paths = model_paths
        self.registry: Dict[str, Dict] = {}

    def load_models(self):
        for name, path in self.model_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Model path not found: {path}")
                continue
            try:
                loaded = joblib.load(path)
                entry = self._normalize_loaded(loaded)
                entry.setdefault("metadata", {})
                entry["metadata"].setdefault("source_path", path)
                entry["metadata"].setdefault("model_name", name)
                # version fallback
                entry["version"] = entry["metadata"].get("version", entry.get("version", "1.0"))
                self.registry[name] = entry
                logger.info(f"Loaded model '{name}' from {path}")
            except Exception as e:
                logger.exception(f"Failed to load model {name} from {path}: {e}")

    def _normalize_loaded(self, loaded: Any) -> Dict[str, Any]:
        # If the saved object is a dict containing 'model'
        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded["model"]
            metadata = {k: v for k, v in loaded.items() if k != "model"}
            return {"model": model, "metadata": metadata}
        
        return {"model": loaded, "metadata": {}}

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self.registry.get(name)

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "version": entry.get("version", entry["metadata"].get("version", "1.0")),
                "metadata": entry.get("metadata", {}),
            }
            for name, entry in self.registry.items()
        }

    def default_model(self) -> Optional[str]:
        # Prefer xgboost if present, else first loaded
        if "xgboost" in self.registry:
            return "xgboost"
        return next(iter(self.registry.keys()), None)
