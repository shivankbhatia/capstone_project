"""
Day 5 — LLM + retrieval (RAG) predictive layer
Lightweight LM (distilGPT2 / T5-small) producing next-character priors,
conditioned on real previously-decoded context. Retrieval layer over a
public AAC/common-phrase corpus for simulated personalization.
Fill in during Day 5.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LanguagePriorEngine:
    def __init__(self, grid_size=(6, 6), model_name='distilgpt2'):
        """
        Initializes the LLM and the Retrieval Layer for next-character prediction.
        """
        print(f"Loading {model_name} for predictive typing...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Optimize for fast CPU/Mac inference
        self.model.eval()
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Map out the 36 valid characters (Standard Study D matrix assumed for init)
        self.valid_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_")
        
        # Precompute tokenizer IDs for our valid characters to speed up inference
        self.char_token_ids = {}
        for char in self.valid_chars:
            # GPT-2 tokenization can be tricky with leading spaces, we grab the raw string token
            token_id = self.tokenizer.encode(char, add_special_tokens=False)[0]
            self.char_token_ids[char] = token_id

        # Initialize the Simulated Personalization (RAG) Corpus
        self._init_retrieval_corpus()

    def _init_retrieval_corpus(self):
        """
        Simulates a user's historical personalized vocabulary. 
        In Phase 2, this will be replaced with real user message history.
        """
        self.aac_corpus = [
            "HELLO HOW ARE YOU",
            "I NEED WATER",
            "PLEASE TURN ON THE LIGHT",
            "YES",
            "NO",
            "THANK YOU",
            "WHAT TIME IS IT",
            "I AM HUNGRY"
        ]
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.corpus_tfidf = self.vectorizer.fit_transform(self.aac_corpus)

    def get_retrieval_prior(self, current_word):
        """
        Searches the personal corpus for words starting with the current prefix.
        Returns a probability boost for characters that complete known words.
        """
        retrieval_probs = np.zeros(len(self.valid_chars))
        if not current_word:
            return retrieval_probs + (1.0 / len(self.valid_chars)) # Uniform if empty
            
        # Very simple RAG matching: boost characters that match our vocabulary
        for phrase in self.aac_corpus:
            words = phrase.split()
            for word in words:
                if word.startswith(current_word) and len(word) > len(current_word):
                    next_char = word[len(current_word)]
                    if next_char in self.valid_chars:
                        idx = self.valid_chars.index(next_char)
                        retrieval_probs[idx] += 1.0
                        
        # Normalize
        if np.sum(retrieval_probs) > 0:
            retrieval_probs /= np.sum(retrieval_probs)
        else:
            retrieval_probs += (1.0 / len(self.valid_chars))
            
        return retrieval_probs

    def get_llm_prior(self, context_text):
        """
        Passes the prior text through DistilGPT2 to get the next-character distribution.
        """
        if not context_text:
            return np.ones(len(self.valid_chars)) / len(self.valid_chars)
            
        inputs = self.tokenizer(context_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get logits of the very last token
            next_token_logits = outputs.logits[0, -1, :]
            
        # Extract logits only for our 36 valid grid characters
        char_logits = np.zeros(len(self.valid_chars))
        for i, char in enumerate(self.valid_chars):
            token_id = self.char_token_ids.get(char)
            char_logits[i] = next_token_logits[token_id].item()
            
        # Apply Softmax with temperature scaling to prevent overconfidence
        temperature = 1.5 
        exp_logits = np.exp(char_logits / temperature)
        llm_probs = exp_logits / np.sum(exp_logits)
        
        return llm_probs

    def get_fused_prior(self, context_text, current_word, alpha=0.7):
        """
        Combines the general English LLM prior with the personalized RAG prior.
        Args:
            alpha: Weight given to the general LLM (0.0 to 1.0).
        """
        llm_prior = self.get_llm_prior(context_text)
        rag_prior = self.get_retrieval_prior(current_word)
        
        fused_prior = (alpha * llm_prior) + ((1 - alpha) * rag_prior)
        return fused_prior

# ==========================================
# Day 5 Testing Harness
# ==========================================
if __name__ == "__main__":
    import time
    print("--- Testing Language Prior Engine ---")
    engine = LanguagePriorEngine(model_name='distilgpt2')
    
    # Simulating a user spelling "HELLO HOW ARE Y" -> Next char should be 'O'
    full_context = "HELLO HOW ARE Y"
    current_word = "Y" 
    
    start_time = time.time()
    prior = engine.get_fused_prior(full_context, current_word, alpha=0.6)
    inference_time = (time.time() - start_time) * 1000
    
    print(f"\nContext: '{full_context}'")
    print(f"Inference Time: {inference_time:.2f} ms")
    
    # Show Top 3 Predictions
    top_indices = np.argsort(prior)[::-1][:3]
    print("\nTop 3 Next-Character Priors:")
    for idx in top_indices:
        char = engine.valid_chars[idx]
        prob = prior[idx]
        print(f"Character: {char} | Probability: {prob:.4f}")