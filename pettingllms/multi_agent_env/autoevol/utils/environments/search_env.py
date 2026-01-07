import os
import requests
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse


class SearchEnvironment:
    def __init__(self, serper_key: str = None):
        self.serper_key = serper_key or os.getenv("SERPER_API_KEY")
    
    def search(self, query: str, filter_year: Optional[int] = None, filter_date: Optional[str] = None) -> str:
        """Search using Serper.dev Google search service
        
        Args:
            query: Search query string
            filter_year: Filter results to specific year (YYYY format)
            filter_date: Filter results before this date (YYYY-MM-DD format)
            
        Returns:
            Formatted string of search results
        """
        url = "https://google.serper.dev/search"
        
        payload = {
            "q": query,
            "num": 10
        }
        
        # Add date filter if specified
        if filter_date is not None:
            try:
                date_obj = datetime.strptime(filter_date, '%Y-%m-%d')
                min_date = "01/01/2000"
                max_date = date_obj.strftime('%m/%d/%Y')
                payload["tbs"] = f"cdr:1,cd_min:{min_date},cd_max:{max_date}"
            except ValueError as e:
                return f"Error: Invalid date format '{filter_date}', expected YYYY-MM-DD: {e}"
        elif filter_year is not None:
            payload["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
        
        headers = {
            'X-API-KEY': self.serper_key,
            'Content-Type': 'application/json'
        }
        
        try:
            #print("DEBUG request payload =", payload)
            #print("DEBUG headers =", headers)
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()
            
            #print("DEBUG response =", response.status_code, response.text)

        except requests.exceptions.RequestException as e:
            #import traceback
            #traceback.print_exc()
            return f"Search error: {str(e)}"
        
        # Check for errors
        if "error" in results:
            return f"API error: {results['error']}"
        
        # Check if we have results
        if "organic" not in results or len(results["organic"]) == 0:
            filter_message = ""
            if filter_date is not None:
                filter_message = f" (filtered before {filter_date})"
            elif filter_year is not None:
                filter_message = f" (filtered year: {filter_year})"
            return f"No results found for '{query}'{filter_message}"
        
        # Format search results
        web_snippets = []
        for idx, page in enumerate(results["organic"], 1):
            title = page.get("title", "No title")
            link = page.get("link", "")
            snippet = page.get("snippet", "")
            
            # Extract date if available
            date_published = ""
            if "date" in page:
                date_published = f"\nDate published: {page['date']}"
            
            # Extract source
            source = ""
            if "source" in page:
                source = f"\nSource: {page['source']}"
            elif link:
                domain = urlparse(link).netloc
                if domain:
                    source = f"\nSource: {domain}"
            
            entry = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
            web_snippets.append(entry)
        
        content = f"Google search for '{query}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
        return content
    
    def fetch(self, url: str) -> str:
        """Fetch webpage content
        
        Args:
            url: URL of the webpage to fetch
            
        Returns:
            String content of the webpage
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "")
            
            # Return text content
            if "text/" in content_type.lower() or "html" in content_type.lower():
                return response.text
            else:
                return f"Unsupported content type: {content_type}"
                
        except requests.exceptions.RequestException as e:
            return f"Error fetching page: {str(e)}"


