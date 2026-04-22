from transformers import pipeline


def run_bert(prompt,model_name = 'distilbert-base-uncased-finetuned-sst-2-english'):
        # Truncate or pad prompt to exactly 512 tokens
        if len(prompt) > 512:
            prompt = prompt[:512]
        
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True,
            device=0  # Uses Metal GPU acceleration on Mac M3
        )

        results = classifier(prompt)[0]

        # Find best prediction
        best = max(results, key=lambda x: x["score"])
        scores = [result["score"] for result in results]
        return best['label'], scores
#run_bert("Maram was really beautiful in makeup only",model_name = 'distilbert-base-uncased-finetuned-sst-2-english')
