"""
Search Tools using Serper API

Provides web search and URL fetching capabilities for AG2 agents.
"""

import os
import requests
from typing import Annotated, Optional
import json


def google_search(
    query: Annotated[str, "The search query to execute"],
    filter_year: Annotated[Optional[int], "Optional: Filter results by year (YYYY format)"] = None,
) -> str:
    """
    Search the web using Google via Serper API.

    This function performs a Google search and returns formatted results
    including titles, URLs, and snippets.

    Args:
        query: The search query string
        filter_year: Optional year to filter results (e.g., 2024)

    Returns:
        Formatted search results as a string

    Example:
        >>> results = google_search("Python programming", filter_year=2024)
        >>> print(results)
    """
    api_key = os.getenv("SERPER_KEY")
    if not api_key:
        return "Error: SERPER_KEY environment variable not set"

    # Prepare request payload
    payload = {
        "q": query,
        "num": 10  # Number of results
    }

    # Add year filter if specified
    if filter_year:
        # Serper API date filter format
        payload["tbs"] = f"cdr:1,cd_min:1/1/{filter_year},cd_max:12/31/{filter_year}"

    try:
        # Make request to Serper API
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=10
        )
        response.raise_for_status()

        results = response.json()

        # Format results for LLM consumption
        if "organic" not in results or not results["organic"]:
            return f"No results found for query: {query}"

        formatted_results = []
        formatted_results.append(f"Google search results for '{query}':\n")

        for i, result in enumerate(results["organic"][:10], 1):
            title = result.get("title", "No title")
            link = result.get("link", "")
            snippet = result.get("snippet", "No description")
            date = result.get("date", "")

            formatted_results.append(f"{i}. [{title}]({link})")
            if date:
                formatted_results.append(f"   Date: {date}")
            formatted_results.append(f"   {snippet}\n")

        return "\n".join(formatted_results)

    except requests.exceptions.RequestException as e:
        return f"Error performing search: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def fetch_url(
    url: Annotated[str, "The URL to fetch content from"],
    query: Annotated[str, "The query to guide content summarization"]
) -> str:
    """
    Fetch content from a URL and summarize it based on a query using LLM.

    This function retrieves content from the given URL, then uses an LLM
    to extract and summarize information relevant to the provided query.

    Args:
        url: The URL to fetch
        query: Query string to guide the summarization

    Returns:
        LLM-summarized content relevant to the query

    Example:
        >>> content = fetch_url("https://example.com/article", "Python features")
        >>> print(content)
    """
    try:
        # Fetch the webpage content
        response = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        # Check if it's text content
        if "text/" not in content_type.lower() and "html" not in content_type.lower():
            return f"Unsupported content type: {content_type}"

        # Get the raw content (truncate if too long to avoid token limits)
        raw_content = response.text[:8000]
        if len(response.text) > 8000:
            raw_content += "\n\n[Content truncated due to length...]"

        # Use LLM to summarize the content based on the query
        return _summarize_with_llm(raw_content, query, url)

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def _summarize_with_llm(content: str, query: str, url: str) -> str:
    """Use LLM to summarize web content based on query

    Args:
        content: Raw webpage content
        query: User's query
        url: Source URL

    Returns:
        Summarized content
    """
    model = os.getenv("CHAT_MODEL", "deepseek-chat")
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_BASE", "https://api.deepseek.com/v1")

    if not api_key:
        return f"Error: API_KEY environment variable not set. Raw content from {url}:\n\n{content[:1000]}..."

    # Prepare the prompt for summarization
    system_prompt = "You are a helpful assistant that extracts and summarizes relevant information from web content based on user queries."
    user_prompt = f"""Based on the following query, please extract and summarize the relevant information from the web content below.

Query: {query}

Web Content from {url}:
{content}

Please provide a concise summary focusing on information relevant to the query. If the content doesn't contain relevant information, please state that clearly."""

    try:
        # Call the LLM API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        api_response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        api_response.raise_for_status()

        result = api_response.json()
        summary = result["choices"][0]["message"]["content"]

        return f"Summary from {url}:\n\n{summary}"

    except requests.exceptions.RequestException as e:
        return f"Error calling LLM API: {str(e)}\n\nRaw content from {url}:\n\n{content[:1000]}..."
    except (KeyError, IndexError) as e:
        return f"Error parsing LLM response: {str(e)}\n\nRaw content from {url}:\n\n{content[:1000]}..."


def register_search_tools(agent, executor=None):
    """
    Register search tools with AG2 agents.

    This is a convenience function to register both google_search and fetch_url
    tools with the appropriate agents.

    Args:
        agent: The ConversableAgent that will call the tools (has LLM)
        executor: The ConversableAgent that will execute the tools (no LLM).
                  If None, agent will execute tools itself (for GroupChat scenarios)

    Example:
        >>> from autogen import ConversableAgent
        >>> from ag2_core import get_deepseek_config
        >>>
        >>> agent = ConversableAgent("Agent", llm_config=get_deepseek_config())
        >>> executor = ConversableAgent("Executor", llm_config=False)
        >>> register_search_tools(agent, executor)
        >>>
        >>> # Or for GroupChat (agent executes tools itself)
        >>> register_search_tools(agent)
    """
    # Register google_search
    agent.register_for_llm(
        name="google-search",
        description="Search the web using Google. Returns formatted search results with titles, URLs, and snippets."
    )(google_search)

    if executor is not None:
        executor.register_for_execution(
            name="google-search"
        )(google_search)
    else:
        # Agent executes tools itself (for GroupChat)
        agent.register_for_execution(
            name="google-search"
        )(google_search)

    # Register fetch_url
    agent.register_for_llm(
        name="fetch_url",
        description="Fetch and read content from a specific URL. Useful for getting detailed information from search results."
    )(fetch_url)

    if executor is not None:
        executor.register_for_execution(
            name="fetch_url"
        )(fetch_url)
    else:
        # Agent executes tools itself (for GroupChat)
        agent.register_for_execution(
            name="fetch_url"
        )(fetch_url)

    if executor is not None:
        print(f"✓ Registered search tools for {agent.name} (executor: {executor.name})")
    else:
        print(f"✓ Registered search tools for {agent.name} (self-executing)")


# For backward compatibility
def check_serper_config() -> bool:
    """Check if Serper API is configured."""
    api_key = os.getenv("SERPER_KEY")
    if not api_key:
        raise ValueError("SERPER_KEY environment variable not set")
    print("✓ Serper API configured")
    return True
