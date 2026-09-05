import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# Import your existing components
from src.models.decoder import P300Decoder
from src.models.fusion import BayesianFusionEngine
from src.models.llm_predictor import LLMPredictor

# Import the evaluation loop from your pipeline script
from run_pipeline import run_evaluation

def run_threshold_sweep():
    print("Initializing components for Threshold Sweep (REAL DATA)...")
    
    # 1. Setup Matrix & LLM
    spelling_matrix = np.array([
        ['A', 'B', 'C', 'D', 'E', 'F'],
        ['G', 'H', 'I', 'J', 'K', 'L'],
        ['M', 'N', 'O', 'P', 'Q', 'R'],
        ['S', 'T', 'U', 'V', 'W', 'X'],
        ['Y', 'Z', '1', '2', '3', '4'],
        ['5', '6', '7', '8', '9', '_']
    ])
    
    decoder = P300Decoder(spelling_matrix)
    llm = LLMPredictor(spelling_matrix)
    
    # 2. Load the REAL Classifier
    # =========================================================================
    # TODO: Change this to the exact path where your SWLDA model is saved.
    # If looping through subjects, dynamically change this path (e.g., f"models/swlda_subj_{subj}.pkl")
    MODEL_PATH = 'data/processed/swlda_model.pkl' 
    # =========================================================================
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading real classifier from {MODEL_PATH}...")
        clf = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"⚠️ Real model NOT FOUND at {MODEL_PATH}. "
                                "Please update MODEL_PATH to point to your saved .pkl file.")
    
    # 3. Define the Fusion Engines (using the optimized weights)
    baseline_fusion = BayesianFusionEngine(mode='fixed', base_alpha=0.0)
    baseline_fusion.toggle(False)
    
    fixed_fusion = BayesianFusionEngine(mode='fixed', base_alpha=0.15, epsilon=0.02)
    adaptive_fusion = BayesianFusionEngine(mode='adaptive', base_alpha=0.25, epsilon=0.02)
    
    # 4. Define the thresholds to test
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    # Data storage for plotting
    results = {
        'baseline': {'acc': [], 'itr': []},
        'fixed': {'acc': [], 'itr': []},
        'adaptive': {'acc': [], 'itr': []}
    }
    
    # 5. Run the sweep
    for t in thresholds:
        print(f"\nEvaluating Confidence Threshold: {t}")
        
        # Baseline
        m, itr = run_evaluation(decoder, llm, baseline_fusion, confidence_threshold=t, clf=clf)
        results['baseline']['acc'].append(m.accuracy)
        results['baseline']['itr'].append(itr)
        
        # Fixed
        m, itr = run_evaluation(decoder, llm, fixed_fusion, confidence_threshold=t, clf=clf)
        results['fixed']['acc'].append(m.accuracy)
        results['fixed']['itr'].append(itr)
        
        # Adaptive
        m, itr = run_evaluation(decoder, llm, adaptive_fusion, confidence_threshold=t, clf=clf)
        results['adaptive']['acc'].append(m.accuracy)
        results['adaptive']['itr'].append(itr)

    # 6. Generate the Plot
    print("\nGenerating Plots...")
    
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Accuracy vs Threshold
    ax1.plot(thresholds, results['baseline']['acc'], marker='o', linestyle='-', color='gray', label='Baseline (EEG Only)')
    ax1.plot(thresholds, results['fixed']['acc'], marker='s', linestyle='--', color='blue', label='Fusion (Fixed)')
    ax1.plot(thresholds, results['adaptive']['acc'], marker='^', linestyle='-', color='orange', linewidth=2, label='Fusion (Adaptive)')
    
    ax1.set_xlabel('Stopping Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs. Stopping Threshold', fontsize=14, fontweight='bold')
    ax1.set_ylim([70, 105]) 
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='lower right')
    
    # Plot 2: ITR vs Threshold
    ax2.plot(thresholds, results['baseline']['itr'], marker='o', linestyle='-', color='gray', label='Baseline (EEG Only)')
    ax2.plot(thresholds, results['fixed']['itr'], marker='s', linestyle='--', color='blue', label='Fusion (Fixed)')
    ax2.plot(thresholds, results['adaptive']['itr'], marker='^', linestyle='-', color='orange', linewidth=2, label='Fusion (Adaptive)')
    
    ax2.set_xlabel('Stopping Confidence Threshold', fontsize=12)
    ax2.set_ylabel('ITR (bits/min)', fontsize=12)
    ax2.set_title('Information Transfer Rate vs. Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plot_path = 'results/threshold_sweep.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot successfully saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    run_threshold_sweep()