import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy

class BayesianFusionEngine:
    def __init__(self, mode='fixed', base_alpha=1.0, epsilon=1e-9):
        """
        Initializes the fusion engine to combine EEG posteriors with LLM priors.
        """
        if mode not in ['fixed', 'adaptive']:
            raise ValueError("Mode must be either 'fixed' or 'adaptive'")
            
        self.mode = mode
        self.base_alpha = base_alpha
        self.epsilon = epsilon
        self.is_active = True  
        
        self.num_classes = 36
        self.max_entropy = np.log(self.num_classes)

    def _safe_log(self, probabilities):
        return np.log(np.clip(probabilities, self.epsilon, 1.0))

    def _calculate_adaptive_alpha(self, eeg_probs):
        current_entropy = entropy(eeg_probs + self.epsilon)
        normalized_entropy = current_entropy / self.max_entropy
        return self.base_alpha * normalized_entropy

    def fuse(self, eeg_probs, llm_probs):
        if not self.is_active:
            return eeg_probs

        log_eeg = self._safe_log(eeg_probs)
        log_llm = self._safe_log(llm_probs)

        current_alpha = self.base_alpha
        if self.mode == 'adaptive':
            current_alpha = self._calculate_adaptive_alpha(eeg_probs)

        unnormalized_log_posterior = log_eeg + (current_alpha * log_llm)

        log_z = logsumexp(unnormalized_log_posterior)
        normalized_log_posterior = unnormalized_log_posterior - log_z

        return np.exp(normalized_log_posterior)
    
    def toggle(self, state: bool):
        self.is_active = state