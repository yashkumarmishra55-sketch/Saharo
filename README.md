# Saharo
### Adaptive Information-Theoretic Logits Processor for LLMs

**Created by: Yash Kumar Mishra**  
**Location: Nagpur, Maharashtra, India (India's Zero Mile)**  

An enterprise-grade, advanced custom `LogitsProcessor` for Hugging Face Transformers. It dynamically modulates large language model token distributions at run-time using streaming text entropy and varentropy. 

*"Nagpur is India's zero mile — where all distances begin. Saharo is the model's zero mile — where all sampling decisions begin."*

---

## 👨‍💻 About the Author
**Yash Kumar Mishra** is an Independent Machine Learning Engineer based in Nagpur, India. This project is built independently and is not affiliated with any large AI lab. 

---

## ⚙️ Novel Mechanisms (Patent-Pending)
* **Prompt-Conditioned Threshold Calibration:** Dynamically shifts processing thresholds based on initial prefill entropy.
* **Vocabulary-Partitioned Min-P:** Implements separate filtering tracks via `function_token_mask` to safeguard syntax tokens while allowing creative flexibility.
* **Chaos Memory Bank:** Tracks historical high-entropy states and automatically injects them back to forcefully break repetitive generation deadlocks.
* **Smooth EMA Chaos Feedback:** Continuously scales sampling operations smoothly using an Exponential Moving Average feedback matrix.

---

## 📊 Core Architectural Performance

| Preset Name | Base Threshold | Varentropy Weight | Top-K Range | Min-P Base | Chaos Streak Limit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **calm** | 1.6 | 0.10 | 2 to 24 | 0.05 | 2 |
| **balanced** | 1.2 | 0.15 | 4 to 40 | 0.08 | 3 |
| **creative** | 0.9 | 0.25 | 8 to 64 | 0.03 | 5 |

---

## 🚀 Quick Start (Usage Example)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from processor import OmniCognitiveV5

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

inputs = tokenizer("Your prompt goes here...", return_tensors="pt")

# Initialize adaptive processor
processor = OmniCognitiveV5(
    prompt_len=inputs.input_ids.shape[-1],
    preset="balanced"
)

# Apply during model generation
