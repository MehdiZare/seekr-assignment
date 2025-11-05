"""Search tool implementations for fact-checking."""

from typing import Any

from langchain_tavily import TavilySearch
from langchain_core.tools import Tool

from app.config import get_config


def create_search_tools() -> list[Tool]:
    """Create and return available search tools based on API key availability.

    Returns:
        List of configured search tools.
    """
    config = get_config()
    tools = []

    # Tavily Search (preferred for comprehensive results)
    if config.settings.tavily_api_key:
        tavily_config = config.search_tools.get("tavily", {})
        tavily_tool = TavilySearch(
            api_key=config.settings.tavily_api_key,
            max_results=tavily_config.get("max_results", 10),
            search_depth=tavily_config.get("search_depth", "advanced"),
        )
        tools.append(tavily_tool)

    # Serper (Google Search API)
    if config.settings.serper_api_key:
        from langchain_community.utilities import GoogleSerperAPIWrapper

        serper_config = config.search_tools.get("serper", {})
        serper = GoogleSerperAPIWrapper(
            serper_api_key=config.settings.serper_api_key,
            k=serper_config.get("num_results", 10),
        )
        serper_tool = Tool(
            name="google_search",
            description="Search Google for recent results. Useful for fact-checking and finding current information.",
            func=serper.run,
        )
        tools.append(serper_tool)

    # Brave Search
    if config.settings.brave_api_key:
        from langchain_community.tools import BraveSearch

        brave_config = config.search_tools.get("brave", {})
        brave_tool = BraveSearch.from_api_key(
            api_key=config.settings.brave_api_key,
            search_kwargs={"count": brave_config.get("count", 10)},
        )
        tools.append(brave_tool)

    if not tools:
        raise ValueError(
            "No search tools available. Please configure at least one search API key "
            "(TAVILY_API_KEY, SERPER_API_KEY, or BRAVE_API_KEY)"
        )

    return tools


def get_tool_descriptions() -> str:
    """Get descriptions of available search tools for agent prompts.

    Returns:
        String describing available tools.
    """
    config = get_config()
    descriptions = []

    if config.settings.tavily_api_key:
        descriptions.append("- Tavily Search: Comprehensive search with advanced filtering")

    if config.settings.serper_api_key:
        descriptions.append("- Google Search: Current web search results")

    if config.settings.brave_api_key:
        descriptions.append("- Brave Search: Privacy-focused search engine")

    return "\n".join(descriptions) if descriptions else "No search tools available"
