# ACReFOSC — A Companion Repository to the French OLDI Seed Corpus

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Paper](https://img.shields.io/badge/paper-WMT_2025-b31b1b.svg)](https://arxiv.org/abs/2508.02290)

This is A Companion REpository to the French OLDI Seed Corpus (ACReFOSC). This repository provides the full set of machine-generated translation candidates and their final post-edited reference, designed to support preference optimization research in machine translation and automatic post-editing.

This repository contains two main files located in the `/data` directory:

1.  **`acrefosc_raw.json`**: Contains the core data, including the English source text, the final human post-edited French reference, all nine machine translation hypotheses, and their corresponding quality scores.
2.  **`acrefosc_prompts.json`**: Contains the prompts used to generate the `Llama-4-Scout_segment-level` hypotheses.

---

## 📊 Data Structure

### `acrefosc_raw.json`

This file is a single JSON array where each object corresponds to one segment and has the following structure:

* `id` (integer): A unique identifier for the segment.
* `url` (string): The source Wikipedia URL.
* `text` (string): The original English source text.
* `post_edited_text` (string): The final, human post-edited French translation. This can be considered the "chosen" or "winning" response.
* **System Keys** (string): Keys for each of the nine MT systems (e.g., `DeepSeek-R1`, `Llama-4-Scout_document-level`, etc.). The value is the raw machine-translated text. These can be considered the "rejected" responses.
* **Score Keys** (float): Keys ending in `_COMET_QE_Score` or `_MetricX_QE_Score` contain the quality scores for the preceding system's translation.

### `acrefosc_prompts.json`

This file is a single JSON array where each object links a segment ID to the prompt used for its translation.

* `id` (integer): The identifier for the segment, which matches the ID in `acrefosc_raw.json`.
* `prompt` (string): The full prompt provided to the `Llama-4-Scout` model for segment-level translation.

---

## 💻 Code Examples

Here is a Python example demonstrating how to load these files and combine them to create a preference dataset suitable for methods like DPO (Direct Preference Optimization). The "baseline" approach (`all`) is to create a preference pair for every machine translation that differs from the post-edited reference. However, one can employ more advanced strategies by using the provided Quality Estimation (QE) scores (COMET_QE_Score and MetricX_QE_Score) to select more informative "losing" examples. For instance, one might select the rejected translation with the highest quality score (`hardest`) or the lowest quality score (`easiest`). The script below implements these strategies.

```python
import json

# 1. Load both datasets from their files
with open('data/acrefosc_raw.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

with open('data/acrefosc_prompts.json', 'r', encoding='utf-8') as f:
    prompts_data = json.load(f)

# 2. Create a dictionary for quick prompt lookup by ID
prompts_map = {item['id']: item['prompt'] for item in prompts_data}

# 3. List the keys for the MT systems
mt_system_keys = [
    "DeepSeek-R1",
    "Llama-4-Scout_document-level",
    "Llama-4-Scout_document-level_no-instruction",
    "Llama-4-Scout_document-level_Wikipedia",
    "Llama-4-Scout_segment-level",
    "MADLAD-400-3B",
    "NLLB-200-600M-Distilled",
    "NLLB-3.3B",
    "OPUS-MT_en-fr",
]

# 4. Generate the preference dataset with advanced selection strategies
def create_preference_dataset(raw_data, prompts_map, system_keys, 
                              format_style='openai', 
                              selection_strategy='all',
                              selection_metric='COMET'):
    """
    Generates a preference dataset with flexible selection strategies.

    Args:
        raw_data (list): Data from acrefosc_raw.json.
        prompts_map (dict): Dictionary mapping IDs to prompts.
        system_keys (list): List of MT system key names.
        format_style (str): Output format ('simple' or 'openai').
        selection_strategy (str): How to choose rejected examples.
            'all': Naive approach, creates a pair for every non-identical hypothesis.
            'hardest': Selects the best-scoring non-identical hypothesis.
            'easiest': Selects the worst-scoring hypothesis.
        selection_metric (str): Which QE score to use ('COMET' or 'MetricX').
    """
    preference_dataset = []
    for instance in raw_data:
        instance_id = instance['id']
        prompt = prompts_map.get(instance_id, f"Translate to French: {instance['text']}")
        chosen_response = instance['post_edited_text']
        
        hypotheses = []
        for key in system_keys:
            if key in instance and instance[key] != chosen_response:
                score_key = f"{key}_{selection_metric}_QE_Score"
                hypotheses.append({
                    "text": instance[key],
                    "score": instance.get(score_key),
                    "system": key
                })
        
        # Filter out hypotheses without a valid score for selection strategies
        scored_hypotheses = [h for h in hypotheses if h['score'] is not None]
        if not scored_hypotheses and selection_strategy != 'all':
            continue

        pairs_to_add = []
        if selection_strategy == 'all':
            pairs_to_add = hypotheses
        elif selection_strategy == 'hardest':
            # [cite_start]Note: MetricX is an error score (lower is better) [cite: 177]
            # So the "hardest" (best) negative has the MINIMUM MetricX score.
            best_hypo = min(scored_hypotheses, key=lambda x: x['score']) if selection_metric == 'MetricX' \
                        else max(scored_hypotheses, key=lambda x: x['score'])
            pairs_to_add = [best_hypo]
        elif selection_strategy == 'easiest':
            # The "easiest" (worst) negative has the MAXIMUM MetricX score.
            worst_hypo = max(scored_hypotheses, key=lambda x: x['score']) if selection_metric == 'MetricX' \
                         else min(scored_hypotheses, key=lambda x: x['score'])
            pairs_to_add = [worst_hypo]

        for hypo in pairs_to_add:
            if format_style == 'openai':
                pair = {
                    "input": {"messages": [{"role": "user", "content": prompt}]},
                    "preferred_output": {"messages": [{"role": "assistant", "content": f"<translation>{chosen_response}</translation>"}]},
                    "non_preferred_output": {"messages": [{"role": "assistant", "content": f"<translation>{hypo['text']}</translation>"}]}
                }
            else:
                pair = {"prompt": prompt, "chosen": f"<translation>{chosen_response}</translation>", "rejected": f"<translation>{hypo['text']}</translation>"}
            preference_dataset.append(pair)
            
    return preference_dataset

# --- Create and save the dataset ---

# You can now choose your strategy.
# 'all': Creates the most pairs.
# 'hardest': Creates one pair per instance with the best-scoring negative.
# 'easiest': Creates one pair per instance with the worst-scoring negative.
strategy = 'all'
metric = 'MetricX' # or 'COMET'
dpo_dataset = create_preference_dataset(raw_data, prompts_map, mt_system_keys, 
                                        selection_strategy=strategy, 
                                        selection_metric=metric)

# Define the output filename
output_filename = f'dpo_preference_dataset_{strategy}_{metric}.jsonl'

# Save the generated dataset to a .jsonl file
with open(output_filename, 'w', encoding='utf-8') as f:
    for pair in dpo_dataset:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f"✅ Success! Generated {len(dpo_dataset)} preference pairs using the '{strategy}' strategy with {metric} scores.")
print(f"Dataset saved to '{output_filename}'")
```
## 📜 Citation

```bibtex
@misc{marmonier2025frenchversionoldiseed,
      title={A French Version of the OLDI Seed Corpus}, 
      author={Malik Marmonier and Benoît Sagot and Rachel Bawden},
      year={2025},
      eprint={2508.02290},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.02290}, 
}
```

## 📄 License

This dataset is released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
