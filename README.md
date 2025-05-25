# LOFin benchmark
## Hierarchical Retrieval with Evidence Curation for Open-Domain Financial Question Answering on Standardized Documents (ACL 2025 Findings)

🚧 **Note: This repository is currently under revision.**  
The current version is **being updated and may not reflect the final implementation**.  
Please refer to future commits for the finalized version of the framework and benchmark dataset.

---
The entire document collection is shared on the following drive:  
- [Google drive](https://drive.google.com/drive/u/0/folders/1Z_G6HGPFYzQaKU5fEea5_w3p7rigov_1) . 

- SEC filings are stored as **PDF** files.  
- Refer to `summarization_results.json` for first-page summaries.  
- Use the extracted `.tar` contents as the **path for the VectorDB**.

---

## 🚀 Running the HiREC Framework

To start the framework:
 you need to run `main_process_iter.py`.

You can also run it with specific argument settings. The available arguments are as follows:

- `--output_dir`: Path to save results.
- `--framework_name`: The framework being used. Ours is `finrag`.
- `--generate_method`: If set to `numeric`, it uses `Pot`; if set to `textual`, it uses `cot`.
- `--dataset`: The type of dataset to evaluate.
- `--use_gpt`: Determines the model to be used for the Evidence Curation process.
- `--max_relevant_ids`: The number of evidence items to collect.
- `--iteration`: The number of iterations for the framework.
- `--max_contexts`: The number of documents to be input in the Generator stage.
- `--use_full_page`: Determines whether the input in the Generator stage is at the passage level or the full page level.
- `--do_generate`: Whether to continue the generation process within the framework.
- `--continue_iteration`: An option to continue running after a specific number of iterations.

If `--do_generate` is set to `False`, only searching will be performed, and no GPT API key is required since the generation process using the GPT-4o model will be skipped.

### 🛠 Environment Setup

Please check the `requirements.txt` file for the main environment settings

pip install -r requirements.txt

### 📊 Running Baseline Experiments

The script `main_process_baseline.py` is used to run existing baselines for comparison experiments.
