




import kagglehub
import requests
import json
import time
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import wraps, lru_cache
import hashlib
from keyword_matcher import IntelligentKeywordMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class KaggleDataset:
    """Data class for Kaggle dataset information."""
    title: str
    url: str
    owner: str
    size: str
    downloads: int
    votes: int
    tags: List[str]
    description: str
    last_updated: str
    license: str
    file_count: int

def rate_limit(delay: float = 1.0):
    """Decorator to add rate limiting to API calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def handle_api_errors(func):
    """Decorator to handle common API errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error in {func.__name__}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return []
    return wrapper

def get_cache_path(query: str) -> str:
    """Generate cache file path for a query."""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return f"cache/kaggle_search_{query_hash}.json"

def load_from_cache(query: str) -> Optional[List[Dict]]:
    """Load search results from cache if available."""
    cache_path = get_cache_path(query)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                logger.info(f"Loaded {len(cached_data)} results from cache for query: {query}")
                return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return None

def save_to_cache(query: str, data: List[Dict]) -> None:
    """Save search results to cache."""
    cache_path = get_cache_path(query)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} results to cache for query: {query}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

@rate_limit(delay=1.0)
@handle_api_errors
def search_kaggle_datasets_api(query: str, page_size: int = 100, max_pages: int = 5) -> List[Dict]:
    """
    Search Kaggle datasets using their public API with pagination support.
    
    Args:
        query: Search query string
        page_size: Number of results per page (max 100)
        max_pages: Maximum number of pages to fetch
    
    Returns:
        List of dataset dictionaries
    """
    # Use the original query directly - Kaggle API works better with simple terms
    search_query = query
    # Kaggle's public search endpoint (unofficial but commonly used)
    base_url = "https://www.kaggle.com/api/v1/datasets/list"
     
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    all_datasets = []
    
    for page in range(max_pages):
        params = {
            'search': search_query,
            'pageSize': min(page_size, 100),
            'sortBy': 'relevance',
            'group': 'public',
            'page': page + 1
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Handle different response structures
            if isinstance(data, list):
                datasets = data
            elif isinstance(data, dict):
                datasets = data.get('datasetListItems', data.get('datasets', []))
            else:
                datasets = []
            
            if not datasets:
                logger.info(f"No more datasets found on page {page + 1}, stopping pagination")
                break
                
            all_datasets.extend(datasets)
            logger.info(f"Found {len(datasets)} datasets on page {page + 1} for query: {query}")
            
            # If we got fewer results than page_size, we've reached the end
            if len(datasets) < page_size:
                break
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed on page {page + 1}: {e}")
            break
    
    logger.info(f"Total found {len(all_datasets)} datasets across {max_pages} pages for query: {query}")
    return all_datasets

def parse_kaggle_dataset(dataset_data: Dict) -> KaggleDataset:
    """Parse raw Kaggle dataset data into structured format."""
    try:
        # Handle tags - they might be dicts or strings
        raw_tags = dataset_data.get('tags', [])
        if raw_tags and isinstance(raw_tags[0], dict):
            tags = [tag.get('name', str(tag)) for tag in raw_tags]
        else:
            tags = [str(tag) for tag in raw_tags]
        
        return KaggleDataset(
            title=dataset_data.get('title', 'Unknown'),
            url=f"https://www.kaggle.com/datasets/{dataset_data.get('ref', '')}",
            owner=dataset_data.get('ownerName', 'Unknown'),
            size=dataset_data.get('size', 'Unknown'),
            downloads=dataset_data.get('downloadCount', 0),
            votes=dataset_data.get('voteCount', 0),
            tags=tags,
            description=dataset_data.get('subtitle', 'No description available'),
            last_updated=dataset_data.get('lastUpdated', 'Unknown'),
            license=dataset_data.get('licenseName', 'Unknown'),
            file_count=dataset_data.get('fileCount', 0)
        )
    except Exception as e:
        logger.error(f"Error parsing dataset: {e}")
        return None

def filter_datasets_by_criteria(datasets: List[KaggleDataset], 
                               min_downloads: int = 0,
                               min_votes: int = 0,
                               required_tags: List[str] = None) -> List[KaggleDataset]:
    """Filter datasets based on specified criteria."""
    filtered = []
    
    for dataset in datasets:
        if dataset is None:
            continue
            
        # Check minimum downloads
        if dataset.downloads < min_downloads:
            continue
            
        # Check minimum votes
        if dataset.votes < min_votes:
            continue
            
        # Check required tags
        if required_tags:
            if not any(tag.lower() in [t.lower() for t in dataset.tags] for tag in required_tags):
                continue
        
        filtered.append(dataset)
    
    return filtered

def fetch_kaggle_datasets_multiple_queries(queries: List[str], 
                                          page_size: int = 100,
                                          max_pages: int = 3,
                                          min_downloads: int = 0,
                                          min_votes: int = 0,
                                          required_tags: List[str] = None,
                                          use_cache: bool = True) -> List[KaggleDataset]:
    """
    Search for datasets using multiple queries and combine results.
    
    Args:
        queries: List of search query strings
        page_size: Number of results per page (max 100)
        max_pages: Maximum number of pages per query
        min_downloads: Minimum number of downloads to filter by
        min_votes: Minimum number of votes to filter by
        required_tags: List of tags that must be present
        use_cache: Whether to use cached results if available
    
    Returns:
        Combined list of unique KaggleDataset objects
    """
    all_datasets = []
    seen_urls = set()
    
    for query in queries:
        logger.info(f"Searching with query: '{query}'")
        datasets = fetch_kaggle_datasets(
            query=query,
            page_size=page_size,
            max_pages=max_pages,
            min_downloads=min_downloads,
            min_votes=min_votes,
            required_tags=required_tags,
            use_cache=use_cache
        )
        
        # Add unique datasets only
        for dataset in datasets:
            if dataset.url not in seen_urls:
                all_datasets.append(dataset)
                seen_urls.add(dataset.url)
    
    logger.info(f"Total unique datasets found: {len(all_datasets)}")
    return all_datasets

def fetch_kaggle_datasets(query: str, 
                         page_size: int = 100,
                         max_pages: int = 5,
                         min_downloads: int = 0,
                         min_votes: int = 0,
                         required_tags: List[str] = None,
                         use_cache: bool = True) -> List[KaggleDataset]:
    """
    Search for datasets on Kaggle based on the given query with maximized results.
    
    Args:
        query: Search query string
        page_size: Number of results per page (max 100)
        max_pages: Maximum number of pages to fetch (default 5 = up to 500 results)
        min_downloads: Minimum number of downloads to filter by
        min_votes: Minimum number of votes to filter by
        required_tags: List of tags that must be present
        use_cache: Whether to use cached results if available
    
    Returns:
        List of KaggleDataset objects
    """
    logger.info(f"Searching Kaggle datasets for query: '{query}'")
    
    # Cache disabled - always fetch fresh results
    logger.info(f"Cache disabled - fetching fresh results for query: '{query}'")
    
    # Search via API with expanded query and pagination
    raw_datasets = search_kaggle_datasets_api(query, page_size, max_pages)
    
    if not raw_datasets:
        logger.warning(f"No datasets found for query: {query}")
        return []
    
    # Parse datasets
    datasets = [parse_kaggle_dataset(data) for data in raw_datasets]
    datasets = [d for d in datasets if d is not None]
    
    # Cache disabled - not saving results
    
    # Apply filters
    filtered_datasets = filter_datasets_by_criteria(datasets, min_downloads, min_votes, required_tags)
    
    logger.info(f"Returning {len(filtered_datasets)} filtered datasets")
    return filtered_datasets

def print_dataset_summary(datasets: List[KaggleDataset]) -> None:
    """Print a formatted summary of the datasets."""
    if not datasets:
        print("No datasets found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(datasets)} datasets:")
    print(f"{'='*80}")
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset.title}")
        print(f"   Owner: {dataset.owner}")
        print(f"   Downloads: {dataset.downloads:,}")
        print(f"   Votes: {dataset.votes}")
        print(f"   Size: {dataset.size}")
        print(f"   Files: {dataset.file_count}")
        print(f"   Tags: {', '.join(dataset.tags[:5])}")  # Show first 5 tags
        print(f"   Description: {dataset.description[:100]}...")
        print(f"   URL: {dataset.url}")
        print(f"   Last Updated: {dataset.last_updated}")
        print("-" * 80)



def answer_k(task):
    # Log in to Kaggle using environment variables if provided
    print("kaggle data:")
    
    # Initialize intelligent keyword matcher
    keyword_matcher = IntelligentKeywordMatcher()
    
    # Convert task to proper query format using intelligent matching
    if isinstance(task, str):
        # Use intelligent keyword matching to get expanded terms
        expanded_terms = keyword_matcher.get_expanded_query_terms(task,5)
        print(f"🎯 Intelligent keywords for '{task}': {expanded_terms}")
        
        # Use expanded terms as search queries
        queries = expanded_terms[:8]  # Use top 8 expanded terms
    else:
        queries = ["reasoning dataset", "mathematical reasoning", "logical reasoning"]

    datasets = fetch_kaggle_datasets_multiple_queries(
        queries=queries,
        page_size=100,
        max_pages=3,
        min_downloads=0,
        min_votes=0
    )
    return datasets




# Example usage and testing
#if __name__ == "__main__":
    # Example searches
 #   {"username":"amirhoseinnaderali","key":"f734d61ab06d8107a7a8b146c34928c6"}
    # Log in to Kaggle using username and key programmatically


    # Set Kaggle credentials using environment variables
  #  os.environ['KAGGLE_USERNAME'] = "amirhoseinnaderali"
   # os.environ['KAGGLE_KEY'] = "f734d61ab06d8107a7a8b146c34928c6"

    # You can verify login by making a simple API call with kaggle package (if installed)
    # Example: List your Kaggle datasets (if any)
    # Make sure 'kaggle' package is installed: pip install kaggle

    #try:
     #   import kaggle
      #  kaggle.api.authenticate()
       # print("Kaggle login successful.")
    #except Exception as e:
     #   print(f"Kaggle login failed: {e}")
      #  print("Continuing without Kaggle authentication...")




    # Test with multiple related queries for QA
    qa_queries = [
        "small model reasoning",
        "reasoning",
        "large reasoning model"
        ,
        "code reasoning ",
        "code reasoning model"
        ,

    


    ]
    
    #print(f"\nSearching for QA-related datasets with multiple queries...")
    #datasets = fetch_kaggle_datasets_multiple_queries(
     #   queries=qa_queries,
      #  page_size=100,
       # max_pages=3,  # 3 pages per query = up to 300 per query
        #min_downloads=0,
        #min_votes=0
    #)
    #print_dataset_summary(datasets)
    #print("\n" + "="*100)