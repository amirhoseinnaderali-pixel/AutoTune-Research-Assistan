import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from google import genai
from huggingface_search import answer_h
from arxiv_search import answer_a
from kaggle_search import answer_k

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gemini API configuration
API_KEY = "AIzaSyD4-OYglZP9aqgtvLiJ5zLdWWmWMYMWENQ"
MODEL_NAME = "gemini-2.0-flash"

# Initialize Gemini client
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")
        client = None
else:
    logger.warning("No Gemini API key found - running in debug mode")

class FineTuningAgent:
    """
    Conversational agent that helps users specify their fine-tuning requirements
    and generates comprehensive research reports using Gemini API.
    """
    
    def __init__(self):
        self.conversation_history = []
        self.user_requirements = {}
        self.search_results = {}
        self.report_generated = False
        # Accumulate good search terms across searches
        self.accumulated_search_terms = {
            "huggingface": [],
            "arxiv": [],
            "kaggle": []
        }
    
    def add_search_terms(self, platform: str, terms: List[str]) -> None:
        """
        Add new search terms to the accumulated list for a specific platform.
        Avoids duplicates and keeps the most recent terms at the end.
        """
        if platform not in self.accumulated_search_terms:
            return
        
        for term in terms:
            if term not in self.accumulated_search_terms[platform]:
                self.accumulated_search_terms[platform].append(term)
                logger.info(f"Added new search term for {platform}: {term}")
    
    def get_combined_search_terms(self, platform: str, new_terms: List[str], max_terms: int = 5) -> List[str]:
        """
        Combine accumulated terms with new terms, prioritizing recent terms.
        Returns up to max_terms total terms.
        """
        if platform not in self.accumulated_search_terms:
            return new_terms[:max_terms]
        
        # Combine accumulated and new terms, removing duplicates
        combined = []
        
        # Add new terms first (higher priority)
        for term in new_terms:
            if term not in combined:
                combined.append(term)
        
        # Add accumulated terms that aren't already included
        for term in self.accumulated_search_terms[platform]:
            if term not in combined:
                combined.append(term)
        
        # Return up to max_terms
        return combined[:max_terms]
    
    def update_search_terms_from_results(self, platform: str, queries: List[str], results: List[Any]) -> None:
        """
        Update accumulated search terms based on successful search results.
        If results are good, keep the queries; if poor, try to improve them.
        """
        if not results or len(results) < 3:
            logger.info(f"Poor results for {platform}, keeping queries for potential improvement")
            # Keep the queries even if results are poor - they might work better later
            self.add_search_terms(platform, queries)
        else:
            logger.info(f"Good results for {platform}, adding successful queries")
            self.add_search_terms(platform, queries)
    
    def get_accumulated_terms_summary(self) -> str:
        """
        Get a summary of accumulated search terms for display.
        """
        summary = []
        for platform, terms in self.accumulated_search_terms.items():
            if terms:
                summary.append(f"{platform.title()}: {', '.join(terms[:3])}{'...' if len(terms) > 3 else ''}")
        return " | ".join(summary) if summary else "No accumulated terms yet"
    
    def query_gemini(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Query Gemini API with retry logic and error handling."""
        if client is None:
            logger.warning("Gemini client not initialized - returning None")
            return None
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Querying Gemini API (attempt {attempt + 1}/{max_retries})")
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
                
                if response and response.text:
                    logger.info("Successfully received response from Gemini")
                    return response.text
                else:
                    logger.warning("Empty response from Gemini API")
                    if attempt == max_retries - 1:
                        return None
                    
            except Exception as e:
                error_msg = str(e)
                if "403" in error_msg or "Forbidden" in error_msg:
                    logger.error("Gemini API access denied. Please check your API key and permissions.")
                    return None
                elif "401" in error_msg or "Unauthorized" in error_msg:
                    logger.error("Gemini API authentication failed. Please check your API key.")
                    return None
                else:
                    logger.error(f"Error querying Gemini API (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached. Gemini API may be unavailable.")
                        return None
        
        return None
    
    def analyze_user_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Analyze user input to extract fine-tuning requirements using Gemini.
        """
        prompt = f"""
You are an expert AI assistant that helps users specify their fine-tuning requirements.

User input: "{user_input}"

Extract and infer the following information from the user's input:

1. **Task Type**: What type of task do they want to perform?
   - text classification (sentiment analysis, categorization)
   - summarization (document summarization, abstractive summarization)
   - QA (question answering, reading comprehension)
   - translation (machine translation, multilingual)
   - generation (text generation, language modeling)
   - reasoning (mathematical reasoning, logical reasoning, chain of thought)
   - code (code generation, code completion)
   - vision (computer vision, image processing)
   - speech (speech recognition, speech synthesis)

2. **Model Size**: What size model do they need?
   - small (<3B parameters) - for mobile/laptop deployment
   - medium (<7B parameters) - balanced performance
   - large (>7B parameters) - high performance, server deployment

3. **Dataset Requirements**: What kind of data do they have or need?
   - domain-specific datasets
   - multilingual data
   - specific data formats
   - data size requirements

4. **Performance Requirements**: What are their accuracy/performance needs?
   - high accuracy (state-of-the-art performance)
   - medium accuracy (good performance)
   - low accuracy (basic functionality)

5. **Resource Constraints**: What are their limitations?
   - budget constraints
   - time constraints
   - hardware limitations
   - deployment environment

6. **Specific Requirements**: Any specific needs or preferences?
   - specific model architectures
   - particular datasets
   - specific evaluation metrics
   - deployment requirements

Return a JSON object with the following structure:
{{
    "task": "extracted_task_type",
    "model_size": "small|medium|large",
    "dataset_requirements": "description_of_data_needs",
    "performance_requirements": "high|medium|low",
    "resource_constraints": "description_of_limitations",
    "specific_requirements": "any_specific_needs",
    "confidence": 0.0-1.0,
    "missing_info": ["list_of_unclear_aspects"]
}}

If any information is unclear, make reasonable inferences based on context.
Be specific and helpful in your analysis.
"""

        response = self.query_gemini(prompt)
        
        if response:
            try:
                # Extract JSON from response
                response = response.strip()
                if response.startswith('{') and response.endswith('}'):
                    data = json.loads(response)
                else:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx+1]
                        data = json.loads(json_str)
                    else:
                        data = {}
                
                logger.info(f"Successfully analyzed user intent: {data}")
                return data
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from Gemini response: {e}")
                return {}
        else:
            logger.warning("No response received from Gemini API")
            return {}
    
    def generate_search_queries(self, requirements: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate optimized search queries for HuggingFace, ArXiv, and Kaggle based on requirements.
        """
        task = requirements.get("task", "reasoning")
        model_size = requirements.get("model_size", "medium")
        dataset_req = requirements.get("dataset_requirements", "")
        performance_req = requirements.get("performance_requirements", "medium")
        
        prompt = f"""
You are an expert at generating search queries for AI/ML research and resources.

Based on these fine-tuning requirements:
- Task: {task}
- Model Size: {model_size}
- Dataset Requirements: {dataset_req}
- Performance Requirements: {performance_req}

Generate optimized search queries for each platform:

1. **HuggingFace Models**: Generate 5-7 queries to find relevant pre-trained models
2. **ArXiv Papers**: Generate 5-7 queries to find relevant research papers
3. **Kaggle Datasets**: Generate 5-7 queries to find relevant datasets

For each platform, consider:
- Task-specific terminology
- Model architecture names
- Dataset names
- Performance metrics
- Recent developments (2023-2024)
- Size-specific terms (small, medium, large models)

Return a JSON object:
{{
    "huggingface": ["query1", "query2", "query3", "query4", "query5"],
    "arxiv": ["query1", "query2", "query3", "query4", "query5"],
    "kaggle": ["query1", "query2", "query3", "query4", "query5"]
}}

Make queries specific, relevant, and likely to return high-quality results.
"""

        response = self.query_gemini(prompt)
        
        if response:
            try:
                response = response.strip()
                if response.startswith('{') and response.endswith('}'):
                    queries = json.loads(response)
                else:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx+1]
                        queries = json.loads(json_str)
                    else:
                        queries = {}
                
                logger.info(f"Generated search queries: {queries}")
                return queries
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse search queries: {e}")
                return {}
        else:
            # Fallback queries
            return {
                "huggingface": [task, f"{task} model", f"{model_size} {task}"],
                "arxiv": [task, f"{task} 2024", f"{task} transformer"],
                "kaggle": [f"{task} dataset", f"{task} data", f"{task} training data"]
            }
    
    def collect_search_results(self, queries: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Collect search results from all three sources using accumulated and new search terms.
        """
        logger.info("Starting comprehensive search across all sources...")
        
        results = {
            "huggingface": [],
            "arxiv": [],
            "kaggle": []
        }
        
        # Search HuggingFace with accumulated terms
        try:
            logger.info("Searching HuggingFace...")
            new_hf_queries = queries.get("huggingface", [queries.get("task", "reasoning")])
            hf_queries = self.get_combined_search_terms("huggingface", new_hf_queries, max_terms=5)
            
            logger.info(f"Using {len(hf_queries)} HuggingFace queries: {hf_queries}")
            
            for query in hf_queries:
                hf_results = answer_h(query)
                if isinstance(hf_results, list):
                    results["huggingface"].extend(hf_results[:20])  # Limit results per query
            
            # Update accumulated terms based on results
            self.update_search_terms_from_results("huggingface", hf_queries, results["huggingface"])
            logger.info(f"Found {len(results['huggingface'])} HuggingFace results")
        except Exception as e:
            logger.error(f"Error searching HuggingFace: {e}")
        
        # Search ArXiv with accumulated terms
        try:
            logger.info("Searching ArXiv...")
            new_arxiv_queries = queries.get("arxiv", [queries.get("task", "reasoning")])
            arxiv_queries = self.get_combined_search_terms("arxiv", new_arxiv_queries, max_terms=5)
            
            logger.info(f"Using {len(arxiv_queries)} ArXiv queries: {arxiv_queries}")
            
            for query in arxiv_queries:
                arxiv_results = answer_a(query)
                if isinstance(arxiv_results, list):
                    results["arxiv"].extend(arxiv_results[:20])  # Limit results per query
            
            # Update accumulated terms based on results
            self.update_search_terms_from_results("arxiv", arxiv_queries, results["arxiv"])
            logger.info(f"Found {len(results['arxiv'])} ArXiv results")
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
        
        # Search Kaggle with accumulated terms
        try:
            logger.info("Searching Kaggle...")
            new_kaggle_queries = queries.get("kaggle", [queries.get("task", "reasoning")])
            kaggle_queries = self.get_combined_search_terms("kaggle", new_kaggle_queries, max_terms=5)
            
            logger.info(f"Using {len(kaggle_queries)} Kaggle queries: {kaggle_queries}")
            
            for query in kaggle_queries:
                kaggle_results = answer_k(query)
                if isinstance(kaggle_results, list):
                    # Convert KaggleDataset objects to dictionaries
                    for result in kaggle_results[:20]:  # Limit results per query
                        if hasattr(result, '__dict__'):
                            results["kaggle"].append(result.__dict__)
                        else:
                            results["kaggle"].append(result)
            
            # Update accumulated terms based on results
            self.update_search_terms_from_results("kaggle", kaggle_queries, results["kaggle"])
            logger.info(f"Found {len(results['kaggle'])} Kaggle results")
        except Exception as e:
            logger.error(f"Error searching Kaggle: {e}")
        
        self.search_results = results
        return results
    
    def generate_comprehensive_report(self, requirements: Dict[str, Any], 
                                    search_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive research report similar to DeepResearch using Gemini.
        """
        prompt = f"""
You are an expert AI research analyst creating a comprehensive fine-tuning recommendation report.

**User Requirements:**
- Task: {requirements.get('task', 'Unknown')}
- Model Size: {requirements.get('model_size', 'Unknown')}
- Dataset Requirements: {requirements.get('dataset_requirements', 'Unknown')}
- Performance Requirements: {requirements.get('performance_requirements', 'Unknown')}
- Resource Constraints: {requirements.get('resource_constraints', 'Unknown')}
- Specific Requirements: {requirements.get('specific_requirements', 'None')}

**Search Results Summary:**
- HuggingFace Models: {len(search_results.get('huggingface', []))} models found
- ArXiv Papers: {len(search_results.get('arxiv', []))} papers found
- Kaggle Datasets: {len(search_results.get('kaggle', []))} datasets found

**Detailed Search Results:**

**HuggingFace Models (Top 10):**
{json.dumps(search_results.get('huggingface', [])[:3], indent=2)}

**ArXiv Papers (Top 10):**
{json.dumps(search_results.get('arxiv', [])[:3], indent=2)}

**Kaggle Datasets (Top 10):**
{json.dumps(search_results.get('kaggle', [])[:3], indent=2)}

Create a comprehensive report with the following structure:

# 🚀 Fine-Tuning Research Report

## 📋 Executive Summary
- Brief overview of the task and requirements
- Key findings and recommendations

## 🎯 Task Analysis
- Detailed analysis of the specified task
- Performance expectations and benchmarks
- Common challenges and solutions

## 🤖 Model Recommendations
### Top 3 Recommended Models
1. **Model Name** - Brief description, pros/cons, suitability
2. **Model Name** - Brief description, pros/cons, suitability  
3. **Model Name** - Brief description, pros/cons, suitability

### Model Comparison Table
| Model | Architecture | Parameters | Performance | Suitability |
|-------|-------------|------------|-------------|------------|
| Model 1 | ... | ... | ... | ... |
| Model 2 | ... | ... | ... | ... |
| Model 3 | ... | ... | ... | ... |

## 📊 Dataset Recommendations
### Top 3 Recommended Datasets
1. **Dataset Name** - Description, size, quality, relevance
2. **Dataset Name** - Description, size, quality, relevance
3. **Dataset Name** - Description, size, quality, relevance

### Dataset Comparison Table
| Dataset | Size | Quality | Relevance | Download Count |
|---------|------|---------|-----------|----------------|
| Dataset 1 | ... | ... | ... | ... |
| Dataset 2 | ... | ... | ... | ... |
| Dataset 3 | ... | ... | ... | ... |

## 📚 Research Insights
### Key Papers and Findings
- Summary of relevant research papers
- Latest developments and SOTA methods
- Implementation recommendations

## 🛠️ Implementation Strategy
### Recommended Approach
1. **Data Preparation**: Steps for data preprocessing
2. **Model Selection**: Rationale for chosen model
3. **Training Strategy**: Fine-tuning approach and hyperparameters
4. **Evaluation**: Metrics and evaluation methodology
5. **Deployment**: Considerations for production deployment

## 💰 Cost and Resource Analysis
- Estimated computational requirements
- Training time estimates
- Hardware recommendations
- Budget considerations

## ⚠️ Potential Challenges and Solutions
- Common issues and how to address them
- Risk mitigation strategies
- Alternative approaches

## 🔗 Resources and References
- Links to recommended models, datasets, and papers
- Additional reading materials
- Community resources

## 📈 Expected Outcomes
- Performance expectations
- Success metrics
- Timeline estimates

Make the report detailed, actionable, and professional. Use emojis and formatting to make it visually appealing and easy to read.
"""

        response = self.query_gemini(prompt)
        
        if response:
            self.report_generated = True
            return response
        else:
            return "Error: Could not generate report. Please check Gemini API configuration."
    
    def start_conversation(self) -> Dict[str, Any]:
        """
        Start the conversational interface to gather requirements and generate report.
        """
        print("🤖 Welcome to AutoTune Fine-Tuning Assistant!")
        print("=" * 60)
        print("I'll help you find the best models, datasets, and approaches for your fine-tuning project.")
        print("Just tell me what you want to accomplish, and I'll do the research for you!")
        print("\nType 'quit' to exit at any time.\n")
        
        conversation_active = True
        
        while conversation_active:
            try:
                user_input = input("💬 What would you like to fine-tune a model for? ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Happy fine-tuning!")
                    conversation_active = False
                    break
                
                if not user_input:
                    print("Please tell me what you'd like to accomplish with fine-tuning.")
                    continue
                
                # Store conversation
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user_input": user_input,
                    "type": "user"
                })
                
                print("\n🔍 Analyzing your requirements...")
                
                # Analyze user intent
                requirements = self.analyze_user_intent(user_input)
                self.user_requirements = requirements
                
                if not requirements:
                    print("❌ Sorry, I couldn't understand your requirements. Please try again.")
                    continue
                
                # Display extracted requirements
                print("\n📋 I understand you want to:")
                print(f"   • Task: {requirements.get('task', 'Not specified')}")
                print(f"   • Model Size: {requirements.get('model_size', 'Not specified')}")
                print(f"   • Performance: {requirements.get('performance_requirements', 'Not specified')}")
                
                missing_info = requirements.get('missing_info', [])
                if missing_info:
                    print(f"   • Missing info: {', '.join(missing_info)}")
                
                # Ask for clarification if needed
                if requirements.get('confidence', 1.0) < 0.7 or missing_info:
                    clarification = input("\n❓ Would you like to provide more details? (y/n): ").strip().lower()
                    if clarification in ['y', 'yes']:
                        additional_input = input("Please provide more details: ").strip()
                        if additional_input:
                            # Refine requirements
                            refined_req = self.analyze_user_intent(f"{user_input} {additional_input}")
                            requirements.update(refined_req)
                            self.user_requirements = requirements
                
                # Generate search queries
                print("\n🔍 Generating optimized search queries...")
                queries = self.generate_search_queries(requirements)
                
                # Show accumulated terms if any
                accumulated_summary = self.get_accumulated_terms_summary()
                if accumulated_summary != "No accumulated terms yet":
                    print(f"📚 Reusing previous search terms: {accumulated_summary}")
                
                # Collect search results
                print("\n📊 Searching across HuggingFace, ArXiv, and Kaggle...")
                search_results = self.collect_search_results(queries)
                
                # Generate comprehensive report
                print("\n📝 Generating comprehensive research report...")
                report = self.generate_comprehensive_report(requirements, search_results)
                
                # Display report
                print("\n" + "="*80)
                print(report)
                print("="*80)
                
                # Store the report
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "content": report,
                    "type": "report"
                })
                
                # Ask if user wants to continue
                continue_choice = input("\n🔄 Would you like to explore a different fine-tuning project? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    conversation_active = False
                    print("👋 Thank you for using AutoTune! Good luck with your fine-tuning project!")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy fine-tuning!")
                conversation_active = False
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                print(f"❌ An error occurred: {e}")
                print("Please try again.")
        
        return {
            "requirements": self.user_requirements,
            "search_results": self.search_results,
            "report_generated": self.report_generated,
            "conversation_history": self.conversation_history
        }

def main():
    """Main function to run the conversational agent."""
    agent = FineTuningAgent()
    
    # Check if Gemini API is available
    if client is None:
        print("⚠️  Warning: Gemini API key not found. The agent will run in limited mode.")
        print("Please set GEMINI_API_KEY environment variable for full functionality.")
        print()
    
    # Start the conversation
    result = agent.start_conversation()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"autotune_session_{timestamp}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Session saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving session: {e}")

if __name__ == "__main__":
    main()
