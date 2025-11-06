"""Configuration management for the Podcast Agent application."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM API Keys
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    llama_api_key: str = Field(..., alias="LLAMA_API_KEY")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")  # Optional - deprecated

    # Search Tool API Keys (optional - at least one required)
    tavily_api_key: str | None = Field(None, alias="TAVILY_API_KEY")
    serper_api_key: str | None = Field(None, alias="SERPER_API_KEY")
    brave_api_key: str | None = Field(None, alias="BRAVE_API_KEY")

    # LangSmith Tracing (optional)
    langsmith_tracing: bool = Field(True, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        "https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    langsmith_api_key: str | None = Field(None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        "pr-healthy-sustainment-69", alias="LANGSMITH_PROJECT"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class Config:
    """Configuration manager that combines YAML config and environment settings."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file and environment variables."""
        self.settings = Settings()

        # Load YAML configuration
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r") as f:
            self.yaml_config: Dict[str, Any] = yaml.safe_load(f)

    @property
    def models(self) -> Dict[str, Any]:
        """Get model configurations."""
        return self.yaml_config.get("models", {})

    @property
    def search_tools(self) -> Dict[str, Any]:
        """Get search tool configurations."""
        return self.yaml_config.get("search_tools", {})

    @property
    def app_settings(self) -> Dict[str, Any]:
        """Get application settings."""
        return self.yaml_config.get("app", {})

    def get_model_config(self, model_key: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        model_config = self.models.get(model_key)
        if not model_config:
            raise ValueError(f"Model configuration not found: {model_key}")
        return model_config

    def get_api_key(self, provider: str) -> str:
        """Get API key for a specific provider."""
        if provider == "anthropic":
            return self.settings.anthropic_api_key
        elif provider == "llama":
            return self.settings.llama_api_key
        elif provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return self.settings.openai_api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_search_tool_key(self, tool: str) -> str | None:
        """Get API key for a specific search tool."""
        if tool == "tavily":
            return self.settings.tavily_api_key
        elif tool == "serper":
            return self.settings.serper_api_key
        elif tool == "brave":
            return self.settings.brave_api_key
        else:
            raise ValueError(f"Unknown search tool: {tool}")

    def setup_langsmith(self) -> None:
        """Set up LangSmith tracing if enabled."""
        if self.settings.langsmith_tracing and self.settings.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.settings.langsmith_endpoint
            os.environ["LANGCHAIN_API_KEY"] = self.settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.settings.langsmith_project


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
