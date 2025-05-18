from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    def __init__(self, provider: str, model_name: str, temperature: float = 0.7, max_tokens: int = 2000):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = self._get_api_key()

    def _get_api_key(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "google":
            return os.getenv("GOOGLE_API_KEY", "")
        return ""

    def get_config(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key
        }

# Model configurations for each agent
DRAFTING_MODELS = {
    "drafting1": ModelConfig("openai", "gpt-4-0125-preview", temperature=0.7),  # GPT-4.1 Mini
    "drafting2": ModelConfig("google", "gemini-pro-flash-2.5", temperature=0.7),  # Flash 2.5 Non-thinking
    "drafting3": ModelConfig("anthropic", "claude-3-haiku-20240307", temperature=0.7)  # Claude 3.5 Haiku
}

VERIFICATION_MODEL = ModelConfig("google", "gemini-pro-flash-2.5-thinking", temperature=0.3)  # Flash 2.5 Thinking
CORRECTIONS_MODEL = ModelConfig("google", "gemini-pro-flash-2.5", temperature=0.7)  # Flash 2.5 Non-thinking
CONSOLIDATION_MODEL = ModelConfig("google", "gemini-pro-flash-2.5", temperature=0.5)  # Flash 2.5 Non-thinking 