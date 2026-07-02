# Benchmarking LLMs

Benchmarking two LLMs on customer feedback data for sentiment classification. Calculates accuracy, latency, token costs (input & output), and generates a confusion matrix.

## Project Overview

This project compares two LLMs:
- **OpenAI GPT-OSS-120B** (via Groq)
- **Llama-3.3-70B** (via Groq)

Both models are evaluated on the same sentiment classification task using customer feedback data.

## What It Does

The benchmarking script:
1. Loads customer feedback from CSV
2. Sends each feedback to both LLMs for sentiment classification (Positive, Negative, or Neutral)
3. Measures:
   - **Accuracy**: How many predictions match the ground truth
   - **Latency**: Average response time in milliseconds
   - **Token Costs**: Input and output tokens consumed with estimated costs
   - **Confusion Matrix**: Breakdown of predictions by true label

## Results

Results are saved to `benchmark_results.json`. Example output:

```json
{
    "openai/gpt-oss-120b": {
        "Overall_Accuracy": 0.92,
        "Average_Latency": 527.81,
        "Estimated_Total_Cost": 0.00156,
        "Confusion_matrix": { ... }
    },
    "llama-3.3-70b-versatile": {
        "Overall_Accuracy": 0.92,
        "Average_Latency": 172.89,
        "Estimated_Total_Cost": 0.00203,
        "Confusion_matrix": { ... }
    }
}
```

## Setup

### Requirements
- Python 3.8+
- See `requirements.txt` for the full list of Python dependencies.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mrinh212375/Benchmarking_LLMS.git
cd Benchmarking_LLMS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
Google_API_key=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## Running the Benchmark

```bash
python benchmark_llms.py
```

The script will:
1. Load the feedback data from `customer_feedback.csv`
2. Call both LLMs for sentiment classification
3. Calculate metrics and confusion matrices
4. Save results to `benchmark_results.json`

## File Structure

- `benchmark_llms.py` - Main benchmarking script
- `customer_feedback.csv` - Labeled customer feedback data (24 samples)
- `benchmark_results.json` - Output with benchmark results
- `dataset.txt` - Source data file
- `.env` - Environment variables for API keys
- `requirements.txt` - Python dependencies

## Data Format

The CSV file contains:
- `feedback_text`: Customer feedback text
- `true_sentiment`: Ground truth label (Positive, Negative, or Neutral)

Example:
```csv
feedback_text,true_sentiment
"The new dashboard is incredibly fast and intuitive. Great job team!",Positive
"The app crashed three times today while I was trying to save my work.",Negative
"How do I change my payment information?",Neutral
```

## License

Apache License 2.0 - See LICENSE file for details
