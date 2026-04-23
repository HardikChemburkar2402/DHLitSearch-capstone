class BartSummarizer:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text: str) -> str:
        if not text or len(text.split()) < 50:
            return text
            
        try:
            inputs = self.tokenizer(
                text, 
                max_length=1024, 
                return_tensors="pt", 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"], 
                    max_length=130, 
                    min_length=40, 
                    length_penalty=2.0, 
                    num_beams=4, 
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            return f"Summary generation failed: {e}"

if __name__ == "__main__":
    tester = BartSummarizer()
    sample = "Digital health literacy is defined as the ability to seek, find, understand, and appraise health information from electronic sources. " * 5 
    print(tester.summarize(sample))