import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy

class BayesianFusionEngine:
    def __init__(self, num_classes, mode='fixed', base_alpha=1.0, epsilon=1e-9):
        """
        Initializes the fusion engine to combine EEG posteriors with LLM priors.

        num_classes: MUST match the actual grid size being decoded (e.g. 72
        for Study D's full keyboard, not the 36-class 6x6 default used
        elsewhere in the pipeline). This directly affects max_entropy, which
        the 'adaptive' mode uses to normalize alpha -- a wrong value here
        silently miscalibrates every adaptive-mode fusion result.
        """
        if mode not in ['fixed', 'adaptive']:
            raise ValueError("Mode must be either 'fixed' or 'adaptive'")

        self.mode = mode
        self.base_alpha = base_alpha
        self.epsilon = epsilon
        self.is_active = True

        self.num_classes = num_classes
        self.max_entropy = np.log(self.num_classes)

    def _safe_log(self, probabilities, unmapped_mask=None):
        """Log with floor at epsilon. Classes in unmapped_mask get -inf so they
        never influence the posterior (avoids epsilon-clipping zero-prob classes)."""
        log_p = np.log(np.clip(probabilities, self.epsilon, 1.0))
        if unmapped_mask is not None:
            log_p[unmapped_mask] = -np.inf
        return log_p

    def _calculate_adaptive_alpha(self, eeg_probs):
        current_entropy = entropy(eeg_probs + self.epsilon)
        normalized_entropy = current_entropy / self.max_entropy
        return self.base_alpha * normalized_entropy

    def fuse(self, eeg_probs, llm_probs):
        if not self.is_active:
            return eeg_probs

        # Chars the LLM assigns zero probability to (unmapped special keys)
        # should be fully excluded, not clipped to epsilon.
        unmapped = (llm_probs == 0)
        log_eeg = self._safe_log(eeg_probs)
        log_llm = self._safe_log(llm_probs, unmapped_mask=unmapped)

        current_alpha = self.base_alpha
        if self.mode == 'adaptive':
            current_alpha = self._calculate_adaptive_alpha(eeg_probs)

        unnormalized_log_posterior = log_eeg + (current_alpha * log_llm)

        log_z = logsumexp(unnormalized_log_posterior)
        normalized_log_posterior = unnormalized_log_posterior - log_z

        return np.exp(normalized_log_posterior)

    def toggle(self, state: bool):
        self.is_active = state
    
    def get_initial_log_bias(self, llm_probs):
        """
        Returns the LM prior's one-time log-domain contribution to the initial
        belief state, BEFORE any EEG evidence has been observed for this
        character. This should be added to the decoder's accumulated_log_probs
        exactly ONCE per character (at reset), NOT re-added on every flash
        sequence -- fuse() is for per-sequence EEG evidence only.

        In 'adaptive' mode, alpha is entropy-scaled by the EEG posterior -- but
        before any EEG evidence exists, the natural assumption is maximum
        uncertainty (a uniform EEG posterior), which makes normalized_entropy
        exactly 1.0 and reduces adaptive alpha to base_alpha here. This isn't a
        special case -- it's the correct adaptive-mode value at zero evidence,
        and lets adaptive-mode trust in the LLM decay naturally as real EEG
        evidence accumulates afterward, without needing a second alpha formula.
        """
        if not self.is_active:
            return np.zeros(self.num_classes)
        unmapped = (llm_probs == 0)
        return self.base_alpha * self._safe_log(llm_probs, unmapped_mask=unmapped)