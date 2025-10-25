from cgitb import small
import logging
import json
import time
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from huggingface_hub import HfApi, ModelInfo, DatasetInfo
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
import requests
from dataclasses import dataclass, asdict
import re
from keyword_matcher import IntelligentKeywordMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hf = HfApi()

class SearchCache:
    """Intelligent caching system for model search results."""
    
    def __init__(self, cache_dir: str = "cache", max_age_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, query: str, filters: Dict[str, Any]) -> str:
        """Generate a unique cache key for a search query."""
        cache_data = {
            "query": query,
            "filters": filters
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file exists and is not expired."""
        if not os.path.exists(cache_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - file_time
        return age < timedelta(hours=self.max_age_hours)
    
    def get(self, query: str, filters: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Get cached results if available and valid."""
        cache_key = self._get_cache_key(query, filters)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Cache hit for query: {query}")
                return data.get('results', [])
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_path}: {e}")
        
        return None
    
    def set(self, query: str, filters: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        """Cache search results."""
        cache_key = self._get_cache_key(query, filters)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                "query": query,
                "filters": filters,
                "results": results,
                "cached_at": datetime.now().isoformat(),
                "result_count": len(results)
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results cached for query: {query} ({len(results)} models)")
        except Exception as e:
            logger.warning(f"Error writing cache file {cache_path}: {e}")
    
    def clear_expired(self) -> int:
        """Clear expired cache files and return count of cleared files."""
        cleared_count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_path = os.path.join(self.cache_dir, filename)
                    if not self._is_cache_valid(cache_path):
                        os.remove(cache_path)
                        cleared_count += 1
            logger.info(f"Cleared {cleared_count} expired cache files")
        except Exception as e:
            logger.warning(f"Error clearing expired cache: {e}")
        
        return cleared_count

# Global cache instance
search_cache = SearchCache()

@dataclass
class ModelMetadata:
    """Enhanced model metadata structure"""
    source: str
    model_id: str
    architecture: str
    parameters: str
    datasets: str
    benchmark: str
    description: str
    downloads: int = 0
    likes: int = 0
    tags: List[str] = None
    pipeline_tag: str = ""
    library_name: str = ""
    language: str = ""
    license: str = ""
    created_at: str = ""
    last_modified: str = ""
    model_size: str = ""
    precision: str = ""
    quantization: str = ""
    hardware_requirements: str = ""
    performance_metrics: Dict[str, Any] = None
    paper_url: str = ""
    demo_url: str = ""
    repository_url: str = ""
    is_gated: bool = False
    is_private: bool = False
    quality_score: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.performance_metrics is None:
            self.performance_metrics = {}

def _infer_architecture_from_tags_and_id(tags: List[str], model_id: str, description: str) -> str:
    """Best-effort inference of architecture from tags, model_id and description."""
    try:
        lower_id = model_id.lower()
        lower_desc = (description or "").lower()
        lower_tags = [t.lower() for t in (tags or [])]
        known_map = [
            ("llama", "LLaMA"), ("mistral", "Mistral"), ("mixtral", "Mixtral"),
            ("bert", "BERT"), ("roberta", "RoBERTa"), ("gpt", "GPT"), ("t5", "T5"),
            ("bart", "BART"), ("falcon", "Falcon"), ("qwen", "Qwen"), ("phi", "Phi"),
            ("gemma", "Gemma"), ("idefics", "Idefics"), ("smollm", "SmolLM"),
            ("smolvlm", "SmolVLM"), ("florence", "Florence"), ("siglip", "SigLIP"),
        ]
        for key, name in known_map:
            if key in lower_id or key in lower_desc or any(key in t for t in lower_tags):
                return name
        if "transformer" in lower_desc or any("transformer" in t for t in lower_tags):
            return "Transformer"
        if any(x in lower_id for x in ["lstm", "rnn"]):
            return "RNN/LSTM"
        if "cnn" in lower_id:
            return "CNN"
    except Exception:
        pass
    return ""


def get_benchmark_from_tags(tags: List[str]) -> str:
    """Get benchmark from tags like 'benchmark:XYZ' and join them."""
    try:
        benchmark_tags = []
        for t in tags or []:
            if t.startswith('benchmark:'):
                benchmark_tags.append(t.split('benchmark:', 1)[1])
        return ", ".join(sorted(set(benchmark_tags)))
    except Exception:
        return ""

def _infer_parameters_from_tags_and_id(tags: List[str], model_id: str, card_data: Dict[str, Any]) -> str:
    """Infer parameter size from tags, model_id patterns, or card fields."""
    try:
        # Prefer explicit fields if present
        if 'parameters' in card_data and card_data['parameters']:
            return str(card_data['parameters'])
        if 'model_size' in card_data and card_data['model_size']:
            return str(card_data['model_size'])
        lower_id = model_id.lower()
        text = " ".join([model_id] + (tags or []))
        # Common patterns like 7b, 1.7b, 70b, 13b
        import re
        m = re.search(r"(\d+\.?\d*)\s*(b|m)\b", lower_id)
        if m:
            num, unit = m.groups()
            return f"{num.upper()}{unit.upper()}"
        # Look for explicit tags like '7b' or '70B'
        up_tags = [t.upper() for t in (tags or [])]
        for t in up_tags:
            if t.endswith("B") or t.endswith("M"):
                # Basic sanity filter
                if any(ch.isdigit() for ch in t):
                    return t
        # Heuristics by name keywords
        if any(k in lower_id for k in ["7b", "1b", "13b", "16b", "32b", "34b", "70b", "405b","100b","175b","200b","250b","300b","350b","400b","450b","500b","550b","600b","650b","700b","750b","800b","850b","900b","950b","1000b"]):
            return "B-scale"
        if any(k in lower_id for k in ["360m", "135m", "220m", "400m", "500m"]):
            return "M-scale"
    except Exception:
        pass
    return ""

def _extract_datasets_from_tags(tags: List[str]) -> str:
    """Extract dataset hints from tags like 'dataset:XYZ' and join them."""
    try:
        dataset_tags = []
        for t in tags or []:
            if t.startswith('dataset:'):
                dataset_tags.append(t.split('dataset:', 1)[1])
        return ", ".join(sorted(set(dataset_tags)))
    except Exception:
        return ""

def _summarize_benchmarks(card_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse model-index metrics into a flat dict name->value when possible."""
    summary: Dict[str, Any] = {}
    try:
        model_index = card_data.get('model-index') or card_data.get('model_index')
        if not model_index:
            return summary
        # model_index is typically a list
        for entry in model_index:
            results = entry.get('results', [])
            for result in results:
                metrics = result.get('metrics', [])
                # metrics is typically a list of dicts with name/value
                if isinstance(metrics, list):
                    for metric in metrics:
                        name = str(metric.get('name') or metric.get('type') or '').strip()
                        value = metric.get('value')
                        if name:
                            # Keep the best value if duplicate names occur
                            if name not in summary:
                                summary[name] = value
    except Exception:
        pass
    return summary

def _infer_precision_and_quantization_from_tags(tags: List[str]) -> Tuple[str, str]:
    """Infer precision and quantization from common tags like bf16/fp16/int8/gguf/awq/gptq."""
    precision = ""
    quant = ""
    try:
        lower = [t.lower() for t in (tags or [])]
        if any("bf16" in t for t in lower):
            precision = "bf16"
        elif any("fp16" in t or "float16" in t for t in lower):
            precision = "fp16"
        elif any("fp32" in t or "float32" in t for t in lower):
            precision = "fp32"
        if any("gguf" in t for t in lower):
            quant = "gguf"
        elif any("gptq" in t for t in lower):
            quant = "gptq"
        elif any("awq" in t for t in lower):
            quant = "awq"
        elif any("int8" in t or "int4" in t for t in lower):
            quant = "int"
    except Exception:
        pass
    return precision, quant

def get_task_keywords(task: str) -> List[str]:
    """Get comprehensive keywords for a given task with enhanced coverage."""
    task_keywords = {
        "summarization": [
            "summarization", "summarize", "abstractive", "extractive",
            "text summarization", "document summarization", "news summarization", 
            "article summarization", "meeting summarization", "conversation summarization",
            "multi-document summarization", "query-focused summarization",
            "bart", "t5", "pegasus", "led", "longformer", "bigbird"
        ],
        "text classification": [
            "sentiment analysis", "text categorization", "document classification", 
            "topic classification", "emotion detection", "spam detection",
            "intent classification", "named entity recognition", "ner",
            "bert", "roberta", "distilbert", "electra", "deberta"
        ],
        "generation": [
            "text generation", "language modeling", "generative model",
            "GPT", "transformer", "neural text generation", "causal language model",
            "gpt-2", "gpt-3", "gpt-4", "bloom", "opt", "llama", "alpaca",
            "chatgpt", "instructgpt", "code generation", "story generation"
        ],
        "QA": [
            "question answering", "reading comprehension", "machine reading",
            "factual QA", "open domain QA", "closed book QA", "open book QA",
            "squad", "natural questions", "triviaqa", "hotpotqa", "quac"
        ],
        "translation": [
            "machine translation", "neural translation", "multilingual",
            "cross-lingual", "language translation", "mt", "nmt",
            "opus", "marian", "mbart", "m2m", "wmt", "bleu"
        ],
        "reasoning": [
            "reasoning", "reasoning-model", "reasoning-llm", "reasoning-transformer",
            "mathematical reasoning", "logical reasoning", "commonsense reasoning",
            "chain of thought", "cot", "gsm8k", "math", "strategyqa",
            "arc", "hellaswag", "piqa", "winogrande", "mmlu",
            "small reasoning", "reasoning small", "math reasoning",
            "logical reasoning", "step-by-step reasoning", "reasoning ability",
            "mathematical problem solving", "arithmetic reasoning"
        ],
        "code": [
            "code generation", "code completion", "code understanding",
            "programming", "software engineering", "codex", "codegen",
            "incoder", "star-coder", "wizardcoder", "codet5", "codebert"
        ],
        "vision": [
            "computer vision", "image classification", "object detection",
            "image segmentation", "vision transformer", "vit", "clip",
            "dall-e", "stable diffusion", "imagen", "flamingo"
        ],
        "speech": [
            "speech recognition", "automatic speech recognition", "asr",
            "text to speech", "tts", "speech synthesis", "whisper",
            "wav2vec", "hubert", "conformer"
        ]
    }
    
    return task_keywords.get(task.lower(), [task])

def create_search_variations(task: str) -> List[str]:
    """Create comprehensive search variations for a given task to maximize results."""
    base_keywords = get_task_keywords(task)
    variations = []
    
    # Add base keywords
    variations.extend(base_keywords)
    
    # Add task-specific search patterns
    task_patterns = {
        "summarization": [
            "abstractive summarization", "extractive summarization", 
            "text summarization", "document summarization",
            "news summarization", "article summarization",
            "meeting summarization", "conversation summarization"
        ],
        "text classification": [
            "sentiment analysis", "text categorization", 
            "document classification", "topic classification",
            "emotion detection", "spam detection", "intent classification"
        ],
        "QA": [
            "question answering", "reading comprehension",
            "machine reading", "factual QA", "open domain QA",
            "closed book QA", "open book QA"
        ],
        "translation": [
            "machine translation", "neural translation",
            "multilingual", "cross-lingual", "language translation",
            "neural machine translation", "nmt"
        ],
        "generation": [
            "text generation", "language modeling", "generative model",
            "GPT", "transformer", "neural text generation",
            "causal language model", "instruct tuning"
        ],
        "reasoning": [
            "reasoning", "reasoning-model", "reasoning-llm", 
            "mathematical reasoning", "logical reasoning",
            "commonsense reasoning", "chain of thought",
            
        ],
        "code": [
            "code generation", "code completion", "code understanding",
            "programming", "software engineering", "codex"
        ],
        "vision": [
            "computer vision", "image classification", "object detection",
            "image segmentation", "vision transformer"
        ],
        "speech": [
            "speech recognition", "automatic speech recognition",
            "text to speech", "speech synthesis"
        ]
        ,
        "samll model reasoning":["small model reasoning"]
    }
    
    variations.extend(task_patterns.get(task.lower(), []))
    
    # Add general ML and transformer terms
    general_terms = [
        "transformer", "attention", "neural network", "deep learning",
        "pretrained", "fine-tuned", "huggingface", "pytorch", "tensorflow",
        "state-of-the-art", "sota", "benchmark", "evaluation",
        "bf16", "fp16", "fp32", "int8", "gguf", "gptq", "awq"
        "quantization", "precision"
    ]
    variations.extend(general_terms)
    
    # Add model size variations
    size_variations = [
        f"{task} small", f"{task} base", f"{task} large", f"{task} xl",
        f"small {task}", f"base {task}", f"large {task}", f"xl {task}"
    ]
    variations.extend(size_variations)
    
    return list(set(variations))

def create_advanced_search_queries(task: str) -> List[Dict[str, Any]]:
    """Create advanced search queries with filters and sorting options."""
    base_keywords = get_task_keywords(task)
    
    queries = []
    
    # High-quality model searches
    for keyword in base_keywords[:5]:  # Top 5 keywords
        queries.append({
            "query": keyword,
            "filters": {
                "pipeline_tag": task,
                "sort": "downloads",
                "direction": -1,
                "limit": 100
            },
            "priority": "high"
        })
    
    # Popular model searches
    queries.append({
        "query": f"{task} model",
        "filters": {
            "sort": "downloads",
            "direction": -1,
            "limit": 50
        },
        "priority": "high"
    })
    
    # Recent model searches
    queries.append({
        "query": f"{task} 2024",
        "filters": {
            "sort": "created_at",
            "direction": -1,
            "limit": 30
        },
        "priority": "medium"
    })
    
    # Benchmark-focused searches
    queries.append({
        "query": f"{task} benchmark",
        "filters": {
            "sort": "likes",
            "direction": -1,
            "limit": 30
        },
        "priority": "medium"
    })
    
    return queries


def extract_model_metadata(model_info: ModelInfo) -> ModelMetadata:
    """Extract comprehensive metadata from a HuggingFace model."""
    try:
        # Basic information
        model_id = model_info.modelId
        card_data = model_info.cardData or {}
        tags = model_info.tags or []
        description = card_data.get('description', '')
        
        # Extract performance metrics from card data
        performance_metrics = _summarize_benchmarks(card_data)
        
        # Extract paper URL
        paper_url = ""
        if 'model-index' in card_data:
            for model_index in card_data['model-index']:
                if 'paperswithcode_id' in model_index:
                    paper_url = f"https://paperswithcode.com/paper/{model_index['paperswithcode_id']}"
                elif 'paper' in model_index:
                    paper_url = model_index['paper']
        
        # Extract demo URL
        demo_url = ""
        if 'inference' in card_data and 'widget' in card_data['inference']:
            demo_url = f"https://huggingface.co/{model_id}"
        
        # Extract model size information
        model_size = ""
        if 'model_size' in card_data:
            model_size = str(card_data['model_size'])
        elif 'parameters' in card_data:
            params = str(card_data['parameters'])
            if 'B' in params or 'billion' in params.lower():
                model_size = params
        
        # Extract precision information
        precision = ""
        if 'precision' in card_data:
            precision = str(card_data['precision'])
        
        # Extract hardware requirements
        hardware_requirements = ""
        if 'hardware_requirements' in card_data:
            hardware_requirements = str(card_data['hardware_requirements'])
        
        # Calculate quality score based on various factors
        quality_score = calculate_quality_score(model_info, performance_metrics)
        
        # Fill core fields with fallbacks/inference
        architecture = card_data.get('architecture', '')
        if not architecture:
            architecture = _infer_architecture_from_tags_and_id(tags, model_id, description)
        parameters = card_data.get('parameters', '')
        if not parameters:
            parameters = _infer_parameters_from_tags_and_id(tags, model_id, card_data)
        datasets = card_data.get('datasets', '')
        if not datasets:
            datasets = _extract_datasets_from_tags(tags)
        benchmark = get_benchmark_from_tags(tags) or card_data.get('benchmark', '')
        if not benchmark and performance_metrics:
            # Produce a short textual summary like "MMLU: 65.2, HellaSwag: 85.1"
            try:
                items = list(performance_metrics.items())[:3]
                benchmark = ", ".join(f"{k}: {v}" for k, v in items)
            except Exception:
                benchmark = ""

        # Precision/quantization inference from tags if missing
        precision = str(card_data.get('precision', '') or '')
        quantization = str(card_data.get('quantization', '') or '')
        if not precision or not quantization:
            inf_p, inf_q = _infer_precision_and_quantization_from_tags(tags)
            precision = precision or inf_p
            quantization = quantization or inf_q

        return ModelMetadata(
            source="HuggingFace",
            model_id=model_id,
            architecture=architecture,
            parameters=parameters,
            datasets=datasets,
            benchmark=benchmark,
            description=description,
            downloads=model_info.downloads,
            likes=model_info.likes,
            tags=tags,
            pipeline_tag=model_info.pipeline_tag or '',
            library_name=card_data.get('library_name', ''),
            language=card_data.get('language', ''),
            license=card_data.get('license', ''),
            created_at=str(model_info.created_at) if model_info.created_at else '',
            last_modified=str(model_info.last_modified) if model_info.last_modified else '',
            model_size=model_size,
            precision=precision,
            quantization=quantization,
            hardware_requirements=hardware_requirements,
            performance_metrics=performance_metrics,
            paper_url=paper_url,
            demo_url=demo_url,
            repository_url=f"https://huggingface.co/{model_id}",
            is_gated=model_info.gated,
            is_private=model_info.private,
            quality_score=quality_score
        )
    except Exception as e:
        logger.warning(f"Error extracting metadata for {model_info.modelId}: {e}")
        return ModelMetadata(
            source="HuggingFace",
            model_id=model_info.modelId,
            architecture="",
            parameters="",
            datasets="",
            benchmark="",
            description="",
            quality_score=0.0
        )

def calculate_quality_score(model_info: ModelInfo, performance_metrics: Dict[str, Any]) -> float:
    """Calculate a quality score for the model based on various factors."""
    score = 0.0
    
    # Downloads score (0-30 points)
    downloads = model_info.downloads or 0
    if downloads > 1000000:  # 1M+ downloads
        score += 30
    elif downloads > 100000:  # 100K+ downloads
        score += 25
    elif downloads > 10000:  # 10K+ downloads
        score += 20
    elif downloads > 1000:  # 1K+ downloads
        score += 15
    elif downloads > 100:  # 100+ downloads
        score += 10
    else:
        score += 5
    
    # Likes score (0-20 points)
    likes = model_info.likes or 0
    if likes > 1000:
        score += 20
    elif likes > 500:
        score += 15
    elif likes > 100:
        score += 10
    elif likes > 10:
        score += 5
    else:
        score += 2
    
    # Performance metrics score (0-25 points)
    if performance_metrics:
        metric_count = len(performance_metrics)
        if metric_count > 5:
            score += 25
        elif metric_count > 3:
            score += 20
        elif metric_count > 1:
            score += 15
        else:
            score += 10
    
    # Architecture and completeness score (0-15 points)
    card_data = model_info.cardData or {}
    completeness_factors = [
        'architecture', 'parameters', 'datasets', 'benchmark', 'description'
    ]
    completeness_score = sum(1 for factor in completeness_factors if card_data.get(factor))
    score += (completeness_score / len(completeness_factors)) * 15
    
    # Pipeline tag score (0-10 points)
    if model_info.pipeline_tag:
        score += 10
    else:
        score += 5
    
    return min(score, 100.0)  # Cap at 100

def login_huggingface():
    """Login to HuggingFace Hub using environment variable.
    Set HUGGINGFACE_TOKEN in your environment to enable login.
    """
    token = "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"
    if not token:
        logger.info("HUGGINGFACE_TOKEN not set; skipping HuggingFace login.")
        return
    try:
        from huggingface_hub import login
        login(token=token)
        logger.info("HuggingFace login successful.")
    except Exception as e:
        logger.warning(f"HuggingFace login failed: {e}")

     


def fetch_hf_models(phase1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Advanced model search engine with intelligent keyword matching."""
    task = phase1["task"]
    expanded_terms = phase1.get("expanded_terms", [])
    
    logger.info(f"Starting intelligent model search for task: {task}")
    logger.info(f"Using expanded terms: {expanded_terms}")
    
    # Use expanded terms for search if available
    if expanded_terms:
        search_queries = []
        for term in expanded_terms[:5]:  # Use top 5 expanded terms
            search_queries.append({
                "query": term,
                "filters": {
                    "sort": "downloads",
                    "direction": -1,
                    "limit": 50
                },
                "priority": "high"
            })
    else:
        # Fallback to original method
        search_queries = create_advanced_search_queries(task)
    
    keywords = expanded_terms[:10] if expanded_terms else create_search_variations(task)
    
    all_models = []
    model_metadata_list = []
    
    # Execute advanced search queries with caching
    for query_config in search_queries:
        try:
            query = query_config['query']
            filters = query_config.get('filters', {})
            
            # Cache disabled - always fetch fresh results
            logger.info(f"Cache disabled - fetching fresh results for query: {query}")
            
            logger.info(f"Executing query: {query} (priority: {query_config['priority']})")
            
            models = hf.list_models(
                search=query,
                limit=filters.get('limit', 100),
                sort=filters.get('sort', 'downloads'),
                direction=filters.get('direction', -1),
                pipeline_tag=filters.get('pipeline_tag')
            )
            
            model_list = list(models)
            all_models.extend(model_list)
            logger.info(f"Found {len(model_list)} models for query: {query}")
            
            # Cache disabled - not saving results
            
        except Exception as e:
            logger.warning(f"Error in advanced query {query_config['query']}: {e}")
            continue
    
    # Execute keyword-based searches with caching
    for keyword in keywords[:10]:  # Top 10 keywords
        try:
            filters = {"limit": 100, "sort": "downloads", "direction": -1}
            
            # Cache disabled - always fetch fresh results
            logger.info(f"Cache disabled - fetching fresh results for keyword: {keyword}")
            
            logger.info(f"Searching with keyword: {keyword}")
            models = hf.list_models(search=keyword, limit=100, sort="downloads", direction=-1)
            model_list = list(models)
            all_models.extend(model_list)
            logger.info(f"Found {len(model_list)} models for keyword: {keyword}")
            
            # Cache disabled - not saving results
            
        except Exception as e:
            logger.warning(f"Error searching for keyword {keyword}: {e}")
            continue
    
    # Remove duplicates and sort by quality
    seen_ids = set()
    unique_models = []
    
    for model in all_models:
        if model.modelId not in seen_ids:
            seen_ids.add(model.modelId)
            unique_models.append(model)
    
    logger.info(f"Total unique models found: {len(unique_models)}")
    
    # Extract comprehensive metadata for each model
    for model in unique_models:
        try:
            metadata = extract_model_metadata(model)
            model_metadata_list.append(metadata)
        except Exception as e:
            logger.warning(f"Error extracting metadata for {model.modelId}: {e}")
            continue
    
    # Sort by quality score and other factors
    model_metadata_list.sort(
        key=lambda x: (x.quality_score, x.downloads, x.likes), 
        reverse=True
    )
    
    # Filter and limit results
    top_models = model_metadata_list[:200]  # Top 200 models
    
    # Convert to dictionary format for compatibility
    rows = []
    for metadata in top_models:
        try:
            row = {
                "source": metadata.source,
                "model_id": metadata.model_id,
                "architecture": metadata.architecture,
                "parameters": metadata.parameters,
                "datasets": metadata.datasets,
                "benchmark": metadata.benchmark,
                "description": metadata.description,
                "downloads": metadata.downloads,
                "likes": metadata.likes,
                "tags": metadata.tags,
                "pipeline_tag": metadata.pipeline_tag,
                "library_name": metadata.library_name,
                "language": metadata.language,
                "license": metadata.license,
                "created_at": metadata.created_at,
                "last_modified": metadata.last_modified,
                "model_size": metadata.model_size,
                "precision": metadata.precision,
                "quantization": metadata.quantization,
                "hardware_requirements": metadata.hardware_requirements,
                "performance_metrics": metadata.performance_metrics,
                "paper_url": metadata.paper_url,
                "demo_url": metadata.demo_url,
                "repository_url": metadata.repository_url,
                "is_gated": metadata.is_gated,
                "is_private": metadata.is_private,
                "quality_score": metadata.quality_score
            }
            rows.append(row)
        except Exception as e:
            logger.warning(f"Error converting metadata to dict for {metadata.model_id}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(rows)} models with comprehensive metadata")
    return rows


def answer_h(task = "small model reasoning"):
    login_huggingface()
    print("huggingface data:")
    # Cache disabled - no need to clear cache files
    
    # Initialize intelligent keyword matcher
    keyword_matcher = IntelligentKeywordMatcher()
    
    # Convert task to proper format for reasoning models
    if isinstance(task, str):
        # Use intelligent keyword matching to get expanded terms
        expanded_terms = keyword_matcher.get_expanded_query_terms(task, 5)
        print(f"🎯 Intelligent keywords for '{task}': {expanded_terms}")
        
        # Use the first expanded term as the main task
        reasoning_task = expanded_terms[0] if expanded_terms else "reasoning"
    else:
        reasoning_task = task.get("task", "reasoning")
    
    # Fetch models with comprehensive metadata using intelligent keywords
    models = fetch_hf_models({"task": reasoning_task, "expanded_terms": expanded_terms})
    return models





       
    
    
    #cache_files = [f for f in os.listdir(search_cache.cache_dir) if f.endswith('.json')]

        