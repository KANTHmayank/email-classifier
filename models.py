from transformers import pipeline

class EmailClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the classifier using a Hugging Face model for sentiment or text classification.
        You can replace the model_name with any other suitable model hosted on Hugging Face Hub.
        """
        self.classifier = pipeline("text-classification", model=model_name)

    def classify(self, prompt: str) -> str:
        """
        Classifies the given prompt using the Hugging Face pipeline.
        
        Returns:
            str: The label and score from the classification result.
        """
        result = self.classifier(prompt, truncation=True)[0]
        return f"{result['label']} (score: {result['score']:.2f})"
