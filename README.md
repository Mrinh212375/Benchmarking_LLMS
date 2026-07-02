# 🚀 LLM Benchmarking Suite

A comprehensive benchmarking framework for evaluating and comparing Large Language Models (LLMs) on real-world customer data. This project systematically measures performance across multiple dimensions including accuracy, latency, token costs, and classification metrics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Metrics & Evaluation](#metrics--evaluation)
- [Results & Analysis](#results--analysis)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

---

## 📊 Overview

This repository contains a production-ready benchmarking framework designed to rigorously evaluate multiple LLM implementations on diverse customer CSV datasets. The framework automates the evaluation process and generates detailed performance metrics to help you make data-driven decisions about model selection and optimization.

**Key Focus Areas:**
- Multi-model performance comparison
- Real-world customer data evaluation
- Comprehensive accuracy metrics
- Latency profiling and analysis
- Cost analysis (input & output tokens)
- Confusion matrix generation for classification tasks

---

## ✨ Features

### 🎯 Core Capabilities

- **Multi-Model Support**: Benchmark multiple LLMs simultaneously
- **Accuracy Metrics**: Comprehensive accuracy calculations and validation
- **Latency Analysis**: Detailed response time profiling and statistical analysis
- **Cost Tracking**: Input/output token counting and cost computation
- **Confusion Matrix**: Classification performance visualization
- **Batch Processing**: Efficient processing of large customer datasets
- **Automated Reporting**: Generate comprehensive benchmark reports

### 📈 Advanced Metrics

- **Precision, Recall, F1-Score**: Standard classification metrics
- **Response Time Distribution**: Percentile-based latency analysis
- **Token Efficiency**: Cost-per-task and tokens-per-response metrics
- **Cross-Model Comparison**: Side-by-side performance analysis

---

## 📁 Project Structure

```
Benchmarking_LLMS/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/
│   ├── llm_config.yaml      # LLM model configurations
│   └── benchmark_config.yaml # Benchmark parameters
├── data/
│   ├── raw/                 # Raw customer CSV data
│   ├── processed/           # Cleaned and processed datasets
│   └── sample_data.csv      # Sample data for testing
├── src/
│   ├── __init__.py
│   ├── benchmark.py         # Main benchmarking engine
│   ├── llm_handler.py       # LLM API interactions
│   ├── metrics.py           # Metrics computation
│   ├── evaluator.py         # Model evaluation logic
│   └── utils.py             # Utility functions
├── results/
│   ├── metrics/             # Output metrics files
│   ├── logs/                # Execution logs
│   └── reports/             # Generated reports
├── tests/
│   ├── test_metrics.py
│   ├── test_benchmark.py
│   └── test_evaluator.py
└── notebooks/
    └── analysis.ipynb       # Jupyter notebook for visualization
```

---

## 🔧 Prerequisites

- **Python**: 3.8 or higher
- **API Access**: Valid API keys for target LLMs
- **Data**: Customer CSV files with consistent schema
- **System Resources**: Sufficient compute for parallel model inference

### Required Libraries

- `requests` - HTTP requests
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Metrics and utilities
- `pyyaml` - Configuration management
- `python-dotenv` - Environment variable management

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mrinh212375/Benchmarking_LLMS.git
cd Benchmarking_LLMS
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```
LLM_API_KEY_1=your_api_key_here
LLM_API_KEY_2=your_second_api_key
API_ENDPOINT_1=https://api.example.com/v1
API_ENDPOINT_2=https://api.example.com/v2
```

---

## ⚙️ Configuration

### LLM Configuration (`config/llm_config.yaml`)

```yaml
models:
  - name: "GPT-4"
    provider: "OpenAI"
    api_key_env: "LLM_API_KEY_1"
    endpoint: "https://api.openai.com/v1/chat/completions"
    timeout: 30
    
  - name: "Claude-3"
    provider: "Anthropic"
    api_key_env: "LLM_API_KEY_2"
    endpoint: "https://api.anthropic.com/v1/messages"
    timeout: 30
```

### Benchmark Configuration (`config/benchmark_config.yaml`)

```yaml
benchmark:
  batch_size: 32
  parallel_workers: 4
  max_retries: 3
  timeout_seconds: 30
  
dataset:
  path: "data/raw/customers.csv"
  test_split: 0.2
  random_seed: 42

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - latency
    - token_cost
  generate_confusion_matrix: true
```

---

## 🚀 Usage

### Basic Benchmarking

```python
from src.benchmark import BenchmarkRunner
from src.metrics import MetricsCollector

# Initialize benchmark runner
runner = BenchmarkRunner(config_path="config/benchmark_config.yaml")

# Run benchmark
results = runner.run()

# Collect and display metrics
metrics = MetricsCollector(results)
metrics.generate_report()
```

### Command-Line Execution

```bash
# Run full benchmark suite
python -m src.benchmark --config config/benchmark_config.yaml --output results/

# Run specific models only
python -m src.benchmark --models "GPT-4" "Claude-3" --output results/

# Quick test run
python -m src.benchmark --test-mode --sample-size 100
```

### Advanced Usage

```python
from src.benchmark import BenchmarkRunner
from src.evaluator import ModelEvaluator

# Initialize with custom configuration
runner = BenchmarkRunner(
    models=["GPT-4", "Claude-3"],
    dataset_path="data/raw/customers.csv",
    batch_size=32
)

# Run benchmark with callbacks
def on_batch_complete(batch_results):
    print(f"Batch complete: {batch_results}")

results = runner.run(on_batch_complete=on_batch_complete)

# Generate comparison report
evaluator = ModelEvaluator(results)
evaluator.generate_comparison_report()
evaluator.save_confusion_matrices("results/confusion_matrices/")
```

---

## 📊 Metrics & Evaluation

### Accuracy Metrics

- **Exact Match**: Percentage of predictions matching ground truth
- **Semantic Similarity**: Token-level or embedding-based similarity scoring
- **Task-Specific Metrics**: Custom metrics based on evaluation task

### Latency Metrics

- **Mean Response Time**: Average inference latency
- **P50/P95/P99 Latency**: Percentile-based latency analysis
- **Throughput**: Requests per second

### Cost Metrics

- **Input Tokens**: Total tokens consumed for prompts
- **Output Tokens**: Total tokens generated in responses
- **Cost Per Request**: Average cost per inference
- **Cost Per Correct Answer**: Efficiency metric combining accuracy and cost

### Classification Metrics

- **Confusion Matrix**: True Positives, False Positives, True Negatives, False Negatives
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Precision-Recall Curve**: Trade-off analysis between precision and recall

---

## 📈 Results & Analysis

### Output Structure

```
results/
├── metrics/
│   ├── accuracy_scores.json
│   ├── latency_analysis.json
│   ├── token_costs.json
│   └── confusion_matrices.json
├── reports/
│   ├── benchmark_report.html
│   ├── comparison_analysis.csv
│   └── summary_statistics.txt
└── logs/
    ├── benchmark_run_2024-01-15.log
    └── model_inference_errors.log
```

### Visualization & Analysis

Use the provided Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook includes:
- Performance comparison charts
- Latency distribution plots
- Cost-accuracy trade-off analysis
- Confusion matrix heatmaps
- Trend analysis across dataset

---

## ⚡ Performance Optimization

### Tips for Faster Benchmarking

1. **Parallel Processing**: Increase `parallel_workers` in config
2. **Batch Size**: Adjust `batch_size` based on available memory
3. **Sampling**: Use `--sample-size` flag for quick validation runs
4. **Caching**: Enable response caching for repeated queries
5. **Model Quantization**: Use quantized models for faster inference

### Resource Management

```python
# Monitor resource usage
from src.benchmark import BenchmarkRunner

runner = BenchmarkRunner(config_path="config/benchmark_config.yaml")
results = runner.run(monitor_resources=True)

# Results include memory and CPU usage metrics
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Submit a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support & Contact

For issues, questions, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/Mrinh212375/Benchmarking_LLMS/issues)
- **Project Maintainer**: [Mrinh212375](https://github.com/Mrinh212375)

---

## 🙏 Acknowledgments

- Built for comprehensive LLM evaluation and comparison
- Designed with production use cases in mind
- Inspired by best practices in ML benchmarking

---

## 📚 Additional Resources

- [LLM Benchmarking Best Practices](https://openai.com/research)
- [Evaluation Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cost Analysis for LLM APIs](https://platform.openai.com/docs/guides/tokens)

---

**Last Updated**: 2024 | **Status**: Active Development
