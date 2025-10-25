import requests
import xml.etree.ElementTree as ET







import requests
import xml.etree.ElementTree as ET
import time
import logging
import pandas as pd
from datetime import datetime
import re
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
from keyword_matcher import IntelligentKeywordMatcher

# تنظیم لاگ‌گیری
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced Model Database for Accurate Extraction
@dataclass
class ModelInfo:
    name: str
    architecture: str
    parameters: str
    variants: List[str]
    description: str

# Comprehensive model database with known models
KNOWN_MODELS = {
    # Transformer-based models
    "bert": ModelInfo("BERT", "Transformer", "110M-340M", ["bert-base", "bert-large"], "Bidirectional Encoder Representations from Transformers"),
    "roberta": ModelInfo("RoBERTa", "Transformer", "125M-355M", ["roberta-base", "roberta-large"], "Robustly Optimized BERT Pretraining Approach"),
    "gpt": ModelInfo("GPT", "Transformer", "117M-175B", ["gpt-2", "gpt-3", "gpt-4"], "Generative Pre-trained Transformer"),
    "t5": ModelInfo("T5", "Transformer", "60M-11B", ["t5-small", "t5-base", "t5-large"], "Text-to-Text Transfer Transformer"),
    "bart": ModelInfo("BART", "Transformer", "140M-400M", ["bart-base", "bart-large"], "Denoising Autoencoder for Pretraining Sequence-to-Sequence Models"),
    "llama": ModelInfo("LLaMA", "Transformer", "7B-70B", ["llama-7b", "llama-13b", "llama-70b"], "Large Language Model Meta AI"),
    "mistral": ModelInfo("Mistral", "Transformer", "7B-8B", ["mistral-7b", "mixtral-8x7b"], "Mistral AI Language Models"),
    "claude": ModelInfo("Claude", "Transformer", "Unknown", ["claude-3", "claude-3.5"], "Anthropic's Language Model"),
    "gemini": ModelInfo("Gemini", "Transformer", "Unknown", ["gemini-pro", "gemini-ultra"], "Google's Multimodal Language Model"),
    "qwen": ModelInfo("Qwen", "Transformer", "1.8B-72B", ["qwen-7b", "qwen-14b", "qwen-72b"], "Alibaba's Large Language Model"),
    "phi": ModelInfo("Phi", "Transformer", "1.3B-3.8B", ["phi-1", "phi-2", "phi-3"], "Microsoft's Compact Language Models"),
    "falcon": ModelInfo("Falcon", "Transformer", "7B-40B", ["falcon-7b", "falcon-40b"], "Technology Innovation Institute's Language Model"),
    "bloom": ModelInfo("BLOOM", "Transformer", "560M-176B", ["bloom-560m", "bloom-1b7", "bloom-176b"], "BigScience Large Open-science Multilingual Model"),
    "opt": ModelInfo("OPT", "Transformer", "125M-175B", ["opt-125m", "opt-350m", "opt-175b"], "Open Pre-trained Transformer Language Models"),
    "alpaca": ModelInfo("Alpaca", "Transformer", "7B", ["alpaca-7b"], "Stanford's Instruction-following Language Model"),
    "vicuna": ModelInfo("Vicuna", "Transformer", "7B-13B", ["vicuna-7b", "vicuna-13b"], "Open-source Chatbot Trained by Fine-tuning LLaMA"),
    "chatglm": ModelInfo("ChatGLM", "Transformer", "6B-32B", ["chatglm-6b", "chatglm-32b"], "Tsinghua's Conversational Language Model"),
    "baichuan": ModelInfo("Baichuan", "Transformer", "7B-13B", ["baichuan-7b", "baichuan-13b"], "百川智能's Large Language Model"),
    "internlm": ModelInfo("InternLM", "Transformer", "7B-20B", ["internlm-7b", "internlm-20b"], "上海AI实验室's Language Model"),
    "xlnet": ModelInfo("XLNet", "Transformer", "110M-340M", ["xlnet-base", "xlnet-large"], "Generalized Autoregressive Pretraining for Language Understanding"),
    "electra": ModelInfo("ELECTRA", "Transformer", "110M-335M", ["electra-base", "electra-large"], "Efficiently Learning an Encoder that Classifies Token Replacements Accurately"),
    "deberta": ModelInfo("DeBERTa", "Transformer", "100M-900M", ["deberta-base", "deberta-large"], "Decoding-enhanced BERT with Disentangled Attention"),
    "distilbert": ModelInfo("DistilBERT", "Transformer", "66M", ["distilbert-base"], "Distilled Version of BERT"),
    "albert": ModelInfo("ALBERT", "Transformer", "12M-18M", ["albert-base", "albert-large"], "A Lite BERT for Self-supervised Learning"),
    
    # Vision-Language Models
    "clip": ModelInfo("CLIP", "Vision-Language", "151M-632M", ["clip-vit", "clip-rn"], "Contrastive Language-Image Pre-training"),
    "blip": ModelInfo("BLIP", "Vision-Language", "224M-1.2B", ["blip-base", "blip-large"], "Bootstrapping Language-Image Pre-training"),
    "dall-e": ModelInfo("DALL-E", "Vision-Language", "12B", ["dall-e-2", "dall-e-3"], "OpenAI's Text-to-Image Generation Model"),
    "stable-diffusion": ModelInfo("Stable Diffusion", "Vision-Language", "860M", ["sd-1.5", "sd-2.1", "sd-xl"], "Stability AI's Text-to-Image Diffusion Model"),
    "imagen": ModelInfo("Imagen", "Vision-Language", "Unknown", ["imagen-3"], "Google's Text-to-Image Generation Model"),
    "midjourney": ModelInfo("Midjourney", "Vision-Language", "Unknown", ["midjourney-v5"], "Midjourney's AI Art Generation Model"),
    
    # Code Models
    "codex": ModelInfo("Codex", "Code Generation", "12B", ["codex-12b"], "OpenAI's Code Generation Model"),
    "codegen": ModelInfo("CodeGen", "Code Generation", "350M-16B", ["codegen-350m", "codegen-16b"], "Salesforce's Code Generation Model"),
    "incoder": ModelInfo("InCoder", "Code Generation", "1B-6B", ["incoder-1b", "incoder-6b"], "Meta's Code Generation Model"),
    "star-coder": ModelInfo("StarCoder", "Code Generation", "15.5B", ["starcoder-15b"], "BigCode's Code Generation Model"),
    "wizardcoder": ModelInfo("WizardCoder", "Code Generation", "7B-15B", ["wizardcoder-7b", "wizardcoder-15b"], "WizardLM's Code Generation Model"),
    "codet5": ModelInfo("CodeT5", "Code Generation", "220M-770M", ["codet5-base", "codet5-large"], "Salesforce's Code Understanding and Generation Model"),
    "codebert": ModelInfo("CodeBERT", "Code Understanding", "125M", ["codebert-base"], "Microsoft's Code Understanding Model"),
    
    # Multimodal Models
    "flamingo": ModelInfo("Flamingo", "Multimodal", "80B", ["flamingo-80b"], "DeepMind's Few-shot Learning Model"),
    "gpt-4v": ModelInfo("GPT-4V", "Multimodal", "Unknown", ["gpt-4-vision"], "OpenAI's Vision-Language Model"),
    "gpt-4o": ModelInfo("GPT-4o", "Multimodal", "Unknown", ["gpt-4-omni"], "OpenAI's Omni Multimodal Model"),
    "gemini-pro-vision": ModelInfo("Gemini Pro Vision", "Multimodal", "Unknown", ["gemini-pro-vision"], "Google's Multimodal Model"),
    "llava": ModelInfo("LLaVA", "Multimodal", "7B-13B", ["llava-7b", "llava-13b"], "Large Language and Vision Assistant"),
    "instructblip": ModelInfo("InstructBLIP", "Multimodal", "1.2B-3.1B", ["instructblip-7b", "instructblip-13b"], "Instruction-tuned BLIP Model"),
    
    # Specialized Models
    "whisper": ModelInfo("Whisper", "Speech Recognition", "39M-1550M", ["whisper-tiny", "whisper-base", "whisper-large"], "OpenAI's Speech Recognition Model"),
    "wav2vec": ModelInfo("Wav2Vec", "Speech Recognition", "95M-317M", ["wav2vec2-base", "wav2vec2-large"], "Facebook's Speech Recognition Model"),
    "hubert": ModelInfo("HuBERT", "Speech Recognition", "95M-317M", ["hubert-base", "hubert-large"], "Facebook's Self-supervised Speech Representation Model"),
    "conformer": ModelInfo("Conformer", "Speech Recognition", "10M-118M", ["conformer-small", "conformer-large"], "Google's Convolution-augmented Transformer"),
    
    # Translation Models
    "marian": ModelInfo("Marian", "Translation", "Unknown", ["marianmt"], "Microsoft's Neural Machine Translation Model"),
    "m2m": ModelInfo("M2M-100", "Translation", "1.2B", ["m2m-100"], "Facebook's Many-to-Many Translation Model"),
    "mbart": ModelInfo("mBART", "Translation", "610M", ["mbart-50"], "Facebook's Multilingual Denoising Pre-training Model"),
    "opus": ModelInfo("OPUS", "Translation", "Unknown", ["opus-mt"], "Helsinki NLP's Translation Models"),
    
    # Summarization Models
    "pegasus": ModelInfo("PEGASUS", "Summarization", "568M", ["pegasus-large"], "Google's Pre-training with Extracted Gap-sentences for Abstractive Summarization"),
    "led": ModelInfo("LED", "Summarization", "163M-568M", ["led-base", "led-large"], "Longformer-Encoder-Decoder for Long Document Summarization"),
    "bigbird": ModelInfo("BigBird", "Summarization", "110M-340M", ["bigbird-base", "bigbird-large"], "Google's Sparse Attention Model for Long Sequences"),
    "longformer": ModelInfo("Longformer", "Summarization", "149M-340M", ["longformer-base", "longformer-large"], "Facebook's Long Document Transformer"),
}

# Enhanced dataset database
KNOWN_DATASETS = {
    # Text Classification
    "imdb": "IMDB Movie Reviews",
    "sst": "Stanford Sentiment Treebank", 
    "ag_news": "AG News Classification",
    "yelp": "Yelp Review Polarity",
    "amazon": "Amazon Product Reviews",
    "rotten_tomatoes": "Rotten Tomatoes Reviews",
    "twitter": "Twitter Sentiment Analysis",
    "reddit": "Reddit Sentiment Analysis",
    
    # Question Answering
    "squad": "Stanford Question Answering Dataset",
    "natural_questions": "Natural Questions",
    "triviaqa": "TriviaQA",
    "hotpotqa": "HotpotQA",
    "quac": "Question Answering in Context",
    "ms_marco": "MS MARCO",
    "drop": "DROP Reading Comprehension",
    "race": "RACE Reading Comprehension",
    
    # Summarization
    "cnn_dailymail": "CNN/DailyMail",
    "xsum": "XSum",
    "multinews": "Multi-News",
    "samsum": "SAMSum",
    "reddit_tifu": "Reddit TIFU",
    "arxiv": "ArXiv Scientific Papers",
    "pubmed": "PubMed Abstracts",
    "scitldr": "SciTLDR",
    "newsroom": "Newsroom",
    "gigaword": "Gigaword",
    
    # Translation
    "wmt": "WMT Translation",
    "iwslt": "IWSLT Translation",
    "europarl": "Europarl",
    "multi30k": "Multi30k",
    "opus": "OPUS Corpus",
    "tatoeba": "Tatoeba",
    "flores": "FLORES",
    "un_mt": "UN Parallel Corpus",
    
    # Generation
    "common_gen": "CommonGen",
    "wikitext": "WikiText",
    "c4": "C4 Corpus",
    "the_pile": "The Pile",
    "openwebtext": "OpenWebText",
    "bookcorpus": "BookCorpus",
    "stories": "WritingPrompts Stories",
    
    # Reasoning
    "gsm8k": "GSM8K Math Problems",
    "math": "MATH Dataset",
    "strategyqa": "StrategyQA",
    "arc": "ARC Challenge",
    "hellaswag": "HellaSwag",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "mmlu": "MMLU",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "codexglue": "CodeXGLUE",
    
    # Vision
    "imagenet": "ImageNet",
    "coco": "COCO",
    "flickr30k": "Flickr30k",
    "vqa": "Visual Question Answering",
    "gqa": "GQA",
    "okvqa": "OK-VQA",
    "nocaps": "NoCaps",
    "flickr8k": "Flickr8k",
    "flickr30k": "Flickr30k",
    
    # Speech
    "librispeech": "LibriSpeech",
    "common_voice": "Common Voice",
    "voxforge": "VoxForge",
    "ted_lium": "TED-LIUM",
    "switchboard": "Switchboard",
    "fisher": "Fisher",
    "ami": "AMI Meeting Corpus",
}

# Enhanced metrics database
KNOWN_METRICS = {
    # Classification Metrics
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "precision": "Precision", 
    "recall": "Recall",
    "auc": "Area Under Curve",
    "macro_f1": "Macro F1",
    "micro_f1": "Micro F1",
    "weighted_f1": "Weighted F1",
    
    # Generation Metrics
    "bleu": "BLEU",
    "rouge": "ROUGE",
    "rouge-1": "ROUGE-1",
    "rouge-2": "ROUGE-2", 
    "rouge-l": "ROUGE-L",
    "meteor": "METEOR",
    "bert_score": "BERTScore",
    "bleurt": "BLEURT",
    "comet": "COMET",
    "ter": "Translation Error Rate",
    "chr_f": "chrF",
    "sacrebleu": "SacreBLEU",
    
    # Question Answering Metrics
    "exact_match": "Exact Match",
    "f1_score": "F1 Score",
    "em": "Exact Match",
    
    # Reasoning Metrics
    "pass_at_k": "Pass@K",
    "pass@1": "Pass@1",
    "pass@10": "Pass@10",
    "pass@100": "Pass@100",
    
    # Language Modeling Metrics
    "perplexity": "Perplexity",
    "ppl": "Perplexity",
    "bits_per_character": "Bits per Character",
    "bits_per_word": "Bits per Word",
    
    # Vision Metrics
    "top1_accuracy": "Top-1 Accuracy",
    "top5_accuracy": "Top-5 Accuracy",
    "mAP": "Mean Average Precision",
    "iou": "Intersection over Union",
    "dice": "Dice Score",
    
    # Speech Metrics
    "wer": "Word Error Rate",
    "cer": "Character Error Rate",
    "bleu": "BLEU",
    "asr_bleu": "ASR BLEU",
}

# Enhanced extraction functions with comprehensive pattern matching
@lru_cache(maxsize=1000)
def extract_model_name_enhanced(text: str) -> str:
    """Enhanced model name extraction using comprehensive pattern matching."""
    if not text:
        return "Unknown"
    
    text_lower = text.lower()
    
    # Direct model name matching with priority order
    for model_key, model_info in KNOWN_MODELS.items():
        # Check for exact model name matches
        if model_key in text_lower:
            return model_info.name
        
        # Check for variant matches
        for variant in model_info.variants:
            if variant.lower() in text_lower:
                return model_info.name
    
    # Pattern-based extraction for unknown models
    patterns = [
        r'\b([A-Z][a-z]+-[A-Z][a-z]+)\b',  # CamelCase-CamelCase
        r'\b([A-Z]{2,}[a-z]*)\b',          # ALLCAPS or Acronyms
        r'\b([a-z]+-[0-9]+[a-z]*)\b',      # lowercase-number
        r'\b([A-Z][a-z]+[0-9]+[a-z]*)\b',  # CamelCase-number
        r'\b([a-z]+_[a-z]+)\b',            # snake_case
        r'\b([A-Z][a-z]+[A-Z][a-z]+)\b',   # CamelCase
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the most likely model name (longest match)
            return max(matches, key=len)
    
    return "Unknown"

@lru_cache(maxsize=1000)
def extract_architecture_enhanced(text: str) -> str:
    """Enhanced architecture detection with comprehensive patterns."""
    if not text:
        return "Unknown"
    
    text_lower = text.lower()
    
    # Direct architecture matching from known models
    for model_key, model_info in KNOWN_MODELS.items():
        if model_key in text_lower:
            return model_info.architecture
    
    # Pattern-based architecture detection
    architecture_patterns = {
        "Transformer": [
            r'\btransformer\b', r'\battention\b', r'\bself-attention\b',
            r'\bmulti-head\b', r'\bencoder-decoder\b', r'\bdecoder-only\b'
        ],
        "RNN/LSTM": [
            r'\blstm\b', r'\brnn\b', r'\bgru\b', r'\brecurrent\b',
            r'\bbidirectional\b', r'\bsequence-to-sequence\b'
        ],
        "CNN": [
            r'\bcnn\b', r'\bconvolutional\b', r'\bconv\b',
            r'\bconvolution\b', r'\bmax-pooling\b'
        ],
        "Vision-Language": [
            r'\bclip\b', r'\bvision-language\b', r'\bmultimodal\b',
            r'\bimage-text\b', r'\bvisual-language\b'
        ],
        "Code Generation": [
            r'\bcode\s+generation\b', r'\bcode\s+completion\b',
            r'\bprogramming\b', r'\bsoftware\s+engineering\b'
        ],
        "Speech Recognition": [
            r'\bspeech\s+recognition\b', r'\basr\b', r'\bautomatic\s+speech\b',
            r'\bwhisper\b', r'\bwav2vec\b', r'\bhubert\b'
        ],
        "Translation": [
            r'\bmachine\s+translation\b', r'\bneural\s+translation\b',
            r'\bmultilingual\b', r'\bcross-lingual\b'
        ],
        "Summarization": [
            r'\bsummarization\b', r'\babstractive\b', r'\bextractive\b',
            r'\bdocument\s+summarization\b'
        ]
    }
    
    for arch, patterns in architecture_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return arch
    
    return "Unknown"

@lru_cache(maxsize=1000)
def extract_parameters_enhanced(text: str) -> str:
    """Enhanced parameter estimation with specific model size patterns."""
    if not text:
        return "Unknown"
    
    text_lower = text.lower()
    
    # Direct parameter matching from known models
    for model_key, model_info in KNOWN_MODELS.items():
        if model_key in text_lower:
            return model_info.parameters
    
    # Specific parameter size patterns
    param_patterns = [
        # Billion-scale models
        (r'\b(\d+(?:\.\d+)?)\s*billion\b', lambda m: f"{m.group(1)}B"),
        (r'\b(\d+(?:\.\d+)?)\s*b\b', lambda m: f"{m.group(1)}B"),
        (r'\b(\d+(?:\.\d+)?)b\b', lambda m: f"{m.group(1)}B"),
        
        # Million-scale models  
        (r'\b(\d+(?:\.\d+)?)\s*million\b', lambda m: f"{m.group(1)}M"),
        (r'\b(\d+(?:\.\d+)?)\s*m\b', lambda m: f"{m.group(1)}M"),
        (r'\b(\d+(?:\.\d+)?)m\b', lambda m: f"{m.group(1)}M"),
        
        # Specific model sizes
        (r'\b(\d+(?:\.\d+)?)\s*gb\b', lambda m: f"{m.group(1)}GB"),
        (r'\b(\d+(?:\.\d+)?)\s*tb\b', lambda m: f"{m.group(1)}TB"),
    ]
    
    for pattern, formatter in param_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return formatter(match)
    
    # Size category estimation
    size_indicators = {
        ">1B": ["large", "big", "xl", "huge", "massive", "7b", "13b", "70b", "175b", "405b"],
        "<1B": ["small", "tiny", "mini", "base", "1b", "3b", "125m", "350m", "560m"],
        "100M-1B": ["medium", "standard", "regular"]
    }
    
    for size_cat, indicators in size_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            return size_cat
    
    return "Unknown"

@lru_cache(maxsize=1000)
def extract_datasets_enhanced(text: str) -> str:
    """Enhanced dataset extraction with comprehensive dataset name patterns."""
    if not text:
        return "Unknown"
    
    text_lower = text.lower()
    found_datasets = []
    
    # Direct dataset matching
    for dataset_key, dataset_name in KNOWN_DATASETS.items():
        if dataset_key in text_lower or dataset_name.lower() in text_lower:
            found_datasets.append(dataset_name)
    
    # Pattern-based dataset extraction
    dataset_patterns = [
        r'\b([A-Z][a-z]+/[A-Z][a-z]+)\b',  # CNN/DailyMail format
        r'\b([A-Z][a-z]+-[A-Z][a-z]+)\b',  # XSum format
        r'\b([A-Z]{2,}[a-z]*)\b',          # SQuAD, BLEU format
        r'\b([a-z]+_[a-z]+)\b',            # snake_case datasets
        r'\b([A-Z][a-z]+[0-9]+[a-z]*)\b',  # Dataset with numbers
    ]
    
    for pattern in dataset_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) > 3 and match.lower() not in [d.lower() for d in found_datasets]:
                found_datasets.append(match)
    
    return ", ".join(found_datasets[:5]) if found_datasets else "Unknown"

@lru_cache(maxsize=1000)
def extract_metrics_enhanced(text: str) -> str:
    """Enhanced metrics extraction with comprehensive metric recognition."""
    if not text:
        return "Unknown"
    
    text_lower = text.lower()
    found_metrics = []
    
    # Direct metric matching
    for metric_key, metric_name in KNOWN_METRICS.items():
        if metric_key in text_lower:
            found_metrics.append(metric_name)
    
    # Pattern-based metric extraction with scores
    metric_patterns = [
        r'\b(rouge-[12l])\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(bleu)\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(f1)\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(accuracy)\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(precision)\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(recall)\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(meteor)\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(bert_score)\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(perplexity)\s*:?\s*(\d+(?:\.\d+)?)\b',
        r'\b(exact_match)\s*:?\s*(\d+(?:\.\d+)?)\b',
    ]
    
    for pattern in metric_patterns:
        matches = re.findall(pattern, text_lower)
        for metric, score in matches:
            metric_name = KNOWN_METRICS.get(metric, metric.upper())
            found_metrics.append(f"{metric_name}: {score}")
    
    return ", ".join(found_metrics[:5]) if found_metrics else "Unknown"

def calculate_quality_score(paper_data: Dict[str, any]) -> float:
    """Calculate quality score based on various factors."""
    score = 0.0
    
    # Model name detection (0-20 points)
    if paper_data.get("ModelName") != "Unknown":
        score += 20
    elif paper_data.get("Architecture") != "Unknown":
        score += 10
    
    # Architecture detection (0-15 points)
    if paper_data.get("Architecture") != "Unknown":
        score += 15
    
    # Parameter estimation (0-10 points)
    if paper_data.get("Parameters") != "Unknown":
        score += 10
    
    # Dataset information (0-15 points)
    if paper_data.get("Datasets") != "Unknown":
        score += 15
    
    # Metrics information (0-15 points)
    if paper_data.get("Metrics") != "Unknown":
        score += 15
    
    # SOTA indication (0-10 points)
    if paper_data.get("IsSOTA"):
        score += 10
    
    # Citation count (0-10 points)
    citation_count = paper_data.get("CitationCount", 0)
    if citation_count > 100:
        score += 10
    elif citation_count > 50:
        score += 7
    elif citation_count > 10:
        score += 5
    elif citation_count > 0:
        score += 2
    
    # Description quality (0-5 points)
    description = paper_data.get("Description", "")
    if len(description) > 200:
        score += 5
    elif len(description) > 100:
        score += 3
    elif len(description) > 50:
        score += 1
    
    return min(score, 100.0)  # Cap at 100

def fetch_arxiv_models_1(phase1, max_results: int = 100, start_year: int = 2023, 
                      categories: List[str] = ["cs.CL", "cs.AI"], page_size: int = 50) -> List[Dict]:

    try:
        # تنظیمات API
        arxiv_url = "http://export.arxiv.org/api/query"
        headers = {"User-Agent": "AutoLLM-Builder/1.0"}
        
        # کلمات کلیدی برای تسک‌های مختلف
        task_keywords = {
            "summarization": ["summarization", "abstractive summarization", "extractive summarization", "text summarization"],
            "text classification": ["text classification", "sentiment analysis", "text categorization"],
            "QA": ["question answering", "QA", "reading comprehension", "answer generation"],
            "translation": ["machine translation", "multilingual translation", "cross-lingual", "neural translation"],
            "generation": ["text generation", "language modeling", "generative model", "text synthesis"],

            "reasoning": ["reasoning", "reasoning-model", "reasoning-llm", "reasoning-transformer", "mathematical reasoning", "logical reasoning", "commonsense reasoning", "chain of thought", "step-by-step reasoning", "arithmetic reasoning", "math reasoning", "reasoning ability", "reasoning capability"],
            "code": ["code generation", "code completion", "code understanding", "programming", "software engineering", "codex"],
            "vision": ["vision", "vision-model", "vision-llm", "vision-transformer", "vision-llm-model", "vision-transformer-model"],
            "speech": ["speech", "speech-model", "speech-llm", "speech-transformer", "speech-llm-model", "speech-transformer-model"],       
            "audio": ["audio", "audio-model", "audio-llm", "audio-transformer", "audio-llm-model", "audio-transformer-model"],     
            "small reasoning":["small reasoning"] 
            
        }
        
        # دیتاست‌های شناخته‌شده برای تسک‌ها
        task_datasets = {
            "translation": ["WMT", "IWSLT", "Europarl", "Multi30k", "OPUS", "Tatoeba"],
            "summarization": ["CNN/DailyMail", "XSum", "arXiv", "PubMed", "SciTLDR"],
            "text classification": ["IMDB", "SST", "AG News", "Yelp"],
            "QA": ["SQuAD", "Natural Questions", "TriviaQA", "HotpotQA"],
            "generation": ["CommonGen", "WikiText", "C4", "The Pile"]
            ,
        }
        
        # معیارهای عملکرد برای تسک‌ها
        task_metrics = {
            "translation": ["BLEU", "chrF", "METEOR", "TER", "COMET"],
            "summarization": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore"],
            "text classification": ["Accuracy", "F1", "Precision", "Recall"],
            "QA": ["F1", "Exact Match", "BLEU"],
            "generation": ["Perplexity", "BLEU", "ROUGE"],
            "reasoning": ["Accuracy", "F1", "Precision", "Recall"],
            "code": ["Accuracy", "F1", "Precision", "Recall"],
            "vision": ["Accuracy", "F1", "Precision", "Recall"],
            "speech": ["Accuracy", "F1", "Precision", "Recall"],
            "audio": ["Accuracy", "F1", "Precision", "Recall"],
            "video": ["Accuracy", "F1", "Precision", "Recall"],
            "NLP": ["Accuracy", "F1", "Precision", "Recall"],
            "Computer Vision": ["Accuracy", "F1", "Precision", "Recall"],
            "Speech Recognition": ["Accuracy", "F1", "Precision", "Recall"],
            "Audio Processing": ["Accuracy", "F1", "Precision", "Recall"],
            "Video Processing": ["Accuracy", "F1", "Precision", "Recall"],
            "Natural Language Processing": ["Accuracy", "F1", "Precision", "Recall"],
            "Machine Learning": ["Accuracy", "F1", "Precision", "Recall"],
            "Deep Learning": ["Accuracy", "F1", "Precision", "Recall"],
            "Artificial Intelligence": ["Accuracy", "F1", "Precision", "Recall"],
            "Neural Networks": ["Accuracy", "F1", "Precision", "Recall"],
            "Convolutional Neural Networks": ["Accuracy", "F1", "Precision", "Recall"],
            "Recurrent Neural Networks": ["Accuracy", "F1", "Precision", "Recall"],
            "Long Short-Term Memory": ["Accuracy", "F1", "Precision", "Recall"],
            "Gated Recurrent Units": ["Accuracy", "F1", "Precision", "Recall"],
            "Bidirectional Recurrent Neural Networks": ["Accuracy", "F1", "Precision", "Recall"],
            "Transformer": ["Accuracy", "F1", "Precision", "Recall"],
            "BERT": ["Accuracy", "F1", "Precision", "Recall"],
            "RoBERTa": ["Accuracy", "F1", "Precision", "Recall"],
            "GPT": ["Accuracy", "F1", "Precision", "Recall"],
            "T5": ["Accuracy", "F1", "Precision", "Recall"],
            "Bart": ["Accuracy", "F1", "Precision", "Recall"],
            "LLM": ["Accuracy", "F1", "Precision", "Recall"],
        }
        
        task = phase1["task"].lower()
        keywords = task_keywords.get(task, [task])
        datasets = task_datasets.get(task, [])
        metrics = task_metrics.get(task, [])
        
        # Enhanced search query construction
        task_terms = " OR ".join([f'"{keyword}"' for keyword in keywords])
        category_terms = " OR ".join([f'cat:{cat}' for cat in categories])
        
        # Enhanced ML terms for better coverage
        ml_terms = '"machine learning" OR "deep learning" OR "neural network" OR "artificial intelligence" OR "AI" OR "ML"'
        
        # Add model-specific terms for better relevance
        model_terms = ""
        if task == "reasoning":
            model_terms = ' OR "large language model" OR "LLM" OR "reasoning model" OR "mathematical reasoning"'
        elif task == "summarization":
            model_terms = ' OR "abstractive summarization" OR "extractive summarization" OR "text summarization"'
        elif task == "translation":
            model_terms = ' OR "neural machine translation" OR "NMT" OR "multilingual" OR "cross-lingual"'
        
        # Construct comprehensive search query
        search_query = f'({category_terms}) AND ({task_terms}{model_terms}) AND ({ml_terms})'
        
        logger.info(f"Search query constructed: {search_query}")
        logger.info(f"Task: {task}, Keywords: {keywords[:5]}...")
        
        # لیست برای ذخیره نتایج
        rows = []
        total_fetched = 0
        
        # صفحه‌بندی برای درخواست‌های بزرگ
        for start in range(0, max_results, page_size):
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": min(page_size, max_results - start),
                "sortBy": "submittedDate",  # مرتب‌سازی بر اساس تازگی
                "sortOrder": "descending"
            }
            
            try:
                response = requests.get(arxiv_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse XML
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                root = ET.fromstring(response.content)
                
                for entry in root.findall('atom:entry', ns):
                    try:
                        # فیلتر تاریخ
                        published_elem = entry.find('atom:published', ns)
                        if published_elem is None:
                            continue
                        published_date = datetime.strptime(published_elem.text, '%Y-%m-%dT%H:%M:%SZ')
                        if published_date.year < start_year:
                            continue
                        
                        # استخراج اطلاعات پایه
                        arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                        title = entry.find('atom:title', ns).text if entry.find('atom:title', ns) is not None else ""
                        summary = entry.find('atom:summary', ns).text if entry.find('atom:summary', ns) is not None else ""
                        authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns) 
                                 if author.find('atom:name', ns) is not None]
                        pdf_url = entry.find("atom:link[@title='pdf']", ns).attrib['href'] if entry.find("atom:link[@title='pdf']", ns) is not None else ""
                        
                        # Create combined text for analysis
                        text = title + " " + summary
                        
                        # Use enhanced extraction functions
                        model_name = extract_model_name_enhanced(text)
                        architecture = extract_architecture_enhanced(text)
                        parameters = extract_parameters_enhanced(text)
                        datasets_extracted = extract_datasets_enhanced(summary)
                        metrics_extracted = extract_metrics_enhanced(summary)
                        
                        # بررسی SOTA
                        def is_sota(text: str) -> bool:
                            return "state-of-the-art" in text.lower() or "sota" in text.lower()
                        
                        # استخراج ژورنال
                        journal_ref = entry.find('atom:journal_ref', ns)
                        journal = journal_ref.text if journal_ref is not None else "Not published"
                        
                        # گرفتن تعداد ارجاعات از Semantic Scholar
                        def get_citation_count(arxiv_id: str) -> int:
                            try:
                                url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount"
                                response = requests.get(url, timeout=5)
                                response.raise_for_status()
                                return response.json().get('citationCount', 0)
                            except:
                                return 0
                        
                        # Create paper data structure
                        paper_data = {
                            "Source": "ArXiv",
                            "PaperID": arxiv_id,
                            "Title": title,
                            "ModelName": model_name,
                            "Architecture": architecture,
                            "Parameters": parameters,
                            "Datasets": datasets_extracted,
                            "Metrics": metrics_extracted,
                            "IsSOTA": is_sota(text),
                            "CitationCount": get_citation_count(arxiv_id),
                            "Authors": ", ".join(authors[:3]),
                            "Published": published_date.strftime('%Y-%m-%d'),
                            "Journal": journal,
                            "Description": summary[:200] + "..." if len(summary) > 200 else summary,
                            "PDF": pdf_url,
                            "Categories": ", ".join(entry.findall('atom:category', ns)[0].attrib.get('term', [])),
                            "Link": entry.find("atom:link[@title='pdf']", ns).attrib['href'] if entry.find("atom:link[@title='pdf']", ns) is not None else ""
                        }
                        
                        # Calculate quality score
                        paper_data["QualityScore"] = calculate_quality_score(paper_data)
                        
                        rows.append(paper_data)
                        
                        total_fetched += 1
                        if total_fetched >= max_results:
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing ArXiv entry {arxiv_id}: {e}")
                        logger.debug(f"Entry details - Title: {title[:50]}..., Authors: {len(authors)}")
                        continue
                
                # رعایت محدودیت نرخ API
                time.sleep(3)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching ArXiv page {start}: {e}")
                logger.debug(f"Request details - URL: {arxiv_url}, Params: {params}")
                continue
        
        # Enhanced sorting by quality score, SOTA status, and citations
        rows = sorted(rows, key=lambda x: (x.get('QualityScore', 0), x['IsSOTA'], x['CitationCount'], x['Published']), reverse=True)
        
        # ذخیره نتایج در CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(f"arxiv_{task}_results.csv", index=False)
            logger.info(f"Results saved to arxiv_{task}_results.csv")
            logger.info(f"Total papers processed: {len(rows)}")
            logger.info(f"Average quality score: {sum(row.get('QualityScore', 0) for row in rows) / len(rows):.2f}")
            logger.info(f"SOTA papers found: {sum(1 for row in rows if row.get('IsSOTA', False))}")
            
            # Log top 3 results for verification
            logger.info("Top 3 results by quality score:")
            for i, row in enumerate(rows[:3]):
                logger.info(f"{i+1}. {row['Title'][:60]}... (Score: {row.get('QualityScore', 0):.1f})")
        else:
            logger.warning("No results found for the given criteria")
        
        return rows if rows else [{"Error": "No results found"}]
        
    except Exception as e:
        logger.error(f"Critical error in fetch_arxiv_models: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
        return [{"Error": str(e)}]





def answer_a(task):
    # Handle both string and dict inputs
    print("arxiv data:")
    
    # Initialize intelligent keyword matcher
    keyword_matcher = IntelligentKeywordMatcher()
    
    if isinstance(task, str):
        # Use intelligent keyword matching to get expanded terms
        expanded_terms = keyword_matcher.get_expanded_query_terms(task, 5)
        print(f"🎯 Intelligent keywords for '{task}': {expanded_terms}")
        
        # Use the first expanded term as the main task
        reasoning_task = expanded_terms[0] if expanded_terms else "reasoning"
    else:
        reasoning_task = "reasoning"
    
    # Convert to proper reasoning task format
    phase1 = {"task": reasoning_task, "expanded_terms": expanded_terms}
    
    result = fetch_arxiv_models_1(phase1, max_results=100, start_year=2023)
    return result

#if __name__ == "__main__":
 #   phase1 = {"task": "small reasoning"}
  #  results = fetch_arxiv_models_1(phase1, max_results=100, start_year=2023)
   # for item in results:
    #    print(item)
     #   print("\n")






















