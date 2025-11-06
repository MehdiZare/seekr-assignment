"""Search tool implementations for fact-checking."""

import logging
from typing import Any

from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain_tavily._utilities import TavilySearchAPIWrapper

from app.config import get_config

logger = logging.getLogger(__name__)


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
        tavliy_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=config.settings.tavily_api_key)

        tavily_tool = TavilySearch(
            max_results=tavily_config.get("max_results", 10),
            search_depth=tavily_config.get("search_depth", "advanced"),
            api_wrapper=tavliy_api_wrapper
        )
        tools.append(tavily_tool)

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

    if config.settings.brave_api_key:
        descriptions.append("- Brave Search: Privacy-focused search engine")

    return "\n".join(descriptions) if descriptions else "No search tools available"


def validate_and_filter_search_results(tool_result: Any, timeout: int = 3) -> Any:
    """
    Validate URLs in search results and remove inaccessible ones (404s, timeouts).

    This function checks each URL in search results to ensure it's accessible
    before passing the results to the LLM. This prevents broken links from
    being included in fact-checking sources.

    Args:
        tool_result: Raw result from search tool
        timeout: Timeout in seconds for URL validation (default: 3)

    Returns:
        Filtered search results with only valid URLs
    """
    import requests
    from requests.exceptions import RequestException

    def is_url_accessible(url: str) -> bool:
        """Check if URL is accessible (not 404 or error)."""
        try:
            # Use HEAD request for speed (doesn't download content)
            response = requests.head(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; PodcastFactChecker/1.0)'}
            )
            # Accept anything under 400 (200s, 300s are fine)
            is_valid = response.status_code < 400
            if not is_valid:
                logger.debug(f"URL validation failed for {url}: Status {response.status_code}")
            return is_valid
        except RequestException as e:
            # Network errors, timeouts, DNS failures, etc.
            logger.debug(f"URL validation failed for {url}: {type(e).__name__}")
            return False

    # Handle Tavily format and similar structured results
    if isinstance(tool_result, dict) and "results" in tool_result:
        original_count = len(tool_result["results"])
        tool_result["results"] = [
            result for result in tool_result["results"]
            if "url" in result and is_url_accessible(result["url"])
        ]
        filtered_count = original_count - len(tool_result["results"])
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} inaccessible URLs from {original_count} search results")

    # Handle string format (some tools return plain text)
    # Just return as-is since we can't validate embedded URLs in text easily

    return tool_result
