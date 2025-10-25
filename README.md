# 🚀 AutoTune Research Assistant

<div align="center">

![AutoTune Logo](https://img.shields.io/badge/AutoTune-Research%20Assistant-blue?style=for-the-badge&logo=robot)

**An Intelligent AI Fine-Tuning Research Assistant**

*Powered by Gemini AI • Searches HuggingFace, ArXiv & Kaggle • Generates Comprehensive Reports*

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Gemini AI](https://img.shields.io/badge/Powered%20by-Gemini%20AI-orange?style=flat&logo=google)](https://ai.google.dev)

</div>

---

## 🌟 What is AutoTune?

**AutoTune** is an intelligent conversational AI assistant that helps researchers, developers, and AI enthusiasts find the perfect models, datasets, and approaches for their fine-tuning projects. Simply describe what you want to accomplish, and AutoTune will:

- 🔍 **Search** across HuggingFace Hub, ArXiv papers, and Kaggle datasets
- 🧠 **Analyze** your requirements using advanced AI
- 📊 **Generate** comprehensive research reports with recommendations
- 🎯 **Provide** actionable insights for your specific use case

## ✨ Key Features

### 🤖 Intelligent Analysis
- **Natural Language Understanding**: Describe your project in plain English
- **Requirement Extraction**: Automatically identifies task type, model size, performance needs
- **Smart Query Generation**: Creates optimized search queries for each platform

### 🔍 Multi-Platform Search
- **HuggingFace Hub**: Find pre-trained models with comprehensive metadata
- **ArXiv Papers**: Discover latest research and SOTA methods
- **Kaggle Datasets**: Locate relevant training data and benchmarks

### 📈 Advanced Intelligence
- **Intelligent Keyword Matching**: Uses TF-IDF and cosine similarity for better results
- **Quality Scoring**: Ranks results by downloads, citations, and relevance
- **Accumulated Learning**: Improves search terms over multiple queries

### 📋 Comprehensive Reports
- **Model Recommendations**: Top 3 models with pros/cons analysis
- **Dataset Suggestions**: Curated datasets with quality metrics
- **Research Insights**: Latest papers and SOTA developments
- **Implementation Strategy**: Step-by-step approach for your project
- **Cost Analysis**: Resource requirements and budget considerations

## 🚀 Quick Start

### Prerequisites
```bash
pip install google-generativeai huggingface-hub requests pandas scikit-learn numpy
```

### Basic Usage
```python
from main import FineTuningAgent

# Initialize the assistant
agent = FineTuningAgent()

# Start interactive session
agent.start_conversation()
```

### Example Interaction
```
🤖 Welcome to AutoTune Fine-Tuning Assistant!
💬 What would you like to fine-tune a model for? 
> I want to create a small model for mathematical reasoning

🔍 Analyzing your requirements...
📋 I understand you want to:
   • Task: reasoning
   • Model Size: small
   • Performance: medium

📊 Searching across HuggingFace, ArXiv, and Kaggle...
📝 Generating comprehensive research report...

🚀 Fine-Tuning Research Report
## 📋 Executive Summary
...
```

## 🛠️ Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/autotune-research-assistant.git
cd autotune-research-assistant
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up API Keys
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: Set HuggingFace token for authenticated access
export HUGGINGFACE_TOKEN="your_hf_token_here"
```

## 📚 Usage Examples

### 1. Text Classification Project
```python
agent = FineTuningAgent()
# Tell the agent: "I need a model for sentiment analysis on social media posts"
```

### 2. Code Generation Task
```python
agent = FineTuningAgent()
# Tell the agent: "I want to fine-tune a model for Python code completion"
```

### 3. Multilingual Translation
```python
agent = FineTuningAgent()
# Tell the agent: "I need a model for English to Persian translation"
```

## 🏗️ Architecture

```
AutoTune Research Assistant
├── 🤖 Main Agent (main.py)
│   ├── Requirement Analysis
│   ├── Query Generation
│   └── Report Generation
├── 🔍 Search Modules
│   ├── HuggingFace Search (huggingface_search.py)
│   ├── ArXiv Search (arxiv_search.py)
│   └── Kaggle Search (kaggle_search.py)
├── 🧠 Intelligence Layer
│   ├── Keyword Matcher (keyword_matcher.py)
│   └── Vocabulary System (vocabulary.py)
└── 📊 Generated Reports
    └── Fine-tuning Recommendations
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional
HUGGINGFACE_TOKEN=your_hf_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### Customization
- **Search Limits**: Modify `max_results` in search functions
- **Quality Thresholds**: Adjust filtering criteria
- **Report Templates**: Customize report generation prompts

## 📊 Supported Tasks

| Task Category | Examples | Supported |
|---------------|----------|-----------|
| **Text Classification** | Sentiment Analysis, Topic Classification | ✅ |
| **Text Generation** | Story Writing, Code Generation | ✅ |
| **Question Answering** | Reading Comprehension, Factual QA | ✅ |
| **Summarization** | Document Summarization, News Summarization | ✅ |
| **Translation** | Machine Translation, Multilingual | ✅ |
| **Reasoning** | Mathematical Reasoning, Logical Reasoning | ✅ |
| **Code** | Code Completion, Code Understanding | ✅ |
| **Vision** | Image Classification, Object Detection | ✅ |
| **Speech** | Speech Recognition, Text-to-Speech | ✅ |

## 🎯 Model Size Support

- **Small Models** (<3B parameters): Mobile/laptop deployment
- **Medium Models** (<7B parameters): Balanced performance
- **Large Models** (>7B parameters): High performance, server deployment

## 📈 Performance Metrics

AutoTune evaluates models based on:
- **Download Count**: Popularity and adoption
- **Citations**: Research impact
- **Benchmark Scores**: Performance metrics
- **Quality Score**: Comprehensive evaluation (0-100)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 Bug Reports
- Use GitHub Issues to report bugs
- Include error messages and steps to reproduce

### 💡 Feature Requests
- Suggest new features via GitHub Issues
- Describe the use case and expected behavior

### 🔧 Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### 📝 Documentation
- Improve README sections
- Add code examples
- Translate to other languages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for powerful language understanding
- **HuggingFace** for the amazing model hub
- **ArXiv** for open research papers
- **Kaggle** for datasets and competitions
- **Open Source Community** for inspiration and tools

## 📞 Support

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and community support
- **Email**: [your-email@example.com](mailto:your-email@example.com)

---

<div align="center">

**Made with ❤️ for the AI Research Community**

[⭐ Star this repo](https://github.com/yourusername/autotune-research-assistant) • [🐛 Report Bug](https://github.com/yourusername/autotune-research-assistant/issues) • [💡 Request Feature](https://github.com/yourusername/autotune-research-assistant/issues)

</div>

---

## 🇮🇷 فارسی

### خوش آمدید به AutoTune Research Assistant

**AutoTune** یک دستیار هوشمند تحقیقاتی است که به شما کمک می‌کند بهترین مدل‌ها، دیتاست‌ها و روش‌های fine-tuning را برای پروژه‌های هوش مصنوعی خود پیدا کنید.

### ویژگی‌های کلیدی:
- 🔍 **جستجوی هوشمند** در HuggingFace، ArXiv و Kaggle
- 🧠 **تحلیل پیشرفته** نیازهای شما با استفاده از Gemini AI
- 📊 **گزارش‌های جامع** با توصیه‌های عملی
- 🎯 **پیشنهادات شخصی‌سازی شده** برای پروژه شما

### نحوه استفاده:
```python
from main import FineTuningAgent
agent = FineTuningAgent()
agent.start_conversation()
```

### مثال:
```
💬 چه کاری می‌خواهید با fine-tuning انجام دهید؟
> می‌خواهم یک مدل کوچک برای استدلال ریاضی بسازم

🔍 در حال تحلیل نیازهای شما...
📊 جستجو در HuggingFace، ArXiv و Kaggle...
📝 تولید گزارش تحقیقاتی جامع...
```

### پشتیبانی:
- برای گزارش باگ و درخواست ویژگی جدید از GitHub Issues استفاده کنید
- برای سوالات و بحث‌های جامعه از Discussions استفاده کنید

**با ❤️ برای جامعه تحقیقاتی هوش مصنوعی ساخته شده است**

---

*اگر از این پروژه خوشتان آمد، لطفاً آن را ستاره ⭐ کنید و نظرات خود را با ما در میان بگذارید!*