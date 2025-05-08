import os

# 1) FORCE all HF cache into /tmp/hf_home (writable in Spaces)
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home"

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class EmailClassifier:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ):
        cache_dir = os.environ["HF_HOME"]
        os.makedirs(cache_dir, exist_ok=True)

        # 2) Load tokenizer & model with explicit cache_dir
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # 3) Create pipeline from our pre-loaded model+tokenizer
        self.classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
        )

    def classify(self, prompt: str) -> str:
        result = self.classifier(prompt, truncation=True)[0]
        return f"{result['label']} (score: {result['score']:.2f})"
