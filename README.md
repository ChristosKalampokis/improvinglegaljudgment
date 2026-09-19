# Improving Legal Judgment Prediction with Hybrid LLM Pipelines

Code, results and LoRA adapters for the MSc thesis of **Christos Kalampokis**,
MSc «Advanced Informatics and Computing Systems – Software Development and Artificial Intelligence»,
Department of Informatics, University of Piraeus (2026). Supervisor: Dionisios Sotiropoulos.

The thesis compares fifty model and method configurations on
[CaseHOLD](https://huggingface.co/datasets/coastalcph/lex_glue) (LexGLUE `case_hold`: 5-way multiple choice,
pick the holding a cited case stands for; 3,600 test items). The configurations span an embedding baseline,
fine-tuned encoders (BERT, RoBERTa, two Legal-BERT variants), FLAN-T5 base/large with prompting,
retrieval-augmented generation (RAG) and LoRA fine-tuning, and Gemini 2.5 Flash/Pro via Vertex AI.

## Main results (test accuracy)

| Model / method | Accuracy | 3-seed mean ± std | MRR |
|---|---|---|---|
| **LoRA FLAN-T5-large** | **89.4%** | n/a | .940 |
| LoRA FLAN-T5-base | 85.2% | 85.5 ± 0.3 | .913 |
| Gemini 2.5 Pro, RAG k=5 | 83.5% | n/a | n/a |
| Gemini 2.5 Flash, RAG k=3 | 78.3% | n/a | n/a |
| Legal-BERT (nlpaueb) | 75.1% | 75.2 ± 0.3 | .856 |
| Legal-BERT (casehold/custom) | 74.2% | 74.9 ± 0.7 | .851 |
| RoBERTa-base | 73.0% | 73.3 ± 0.7 | .843 |
| BERT-base | 72.4% | 71.3 ± 1.0 | .839 |
| FLAN-T5-large, zero-shot | 67.6% | n/a | n/a |
| FLAN-T5-base, RAG k=3 | 53.9% | n/a | n/a |
| SBERT embedding similarity | 52.3% | n/a | .705 |

Key findings:

1. Parameter-efficient fine-tuning wins: a LoRA-adapted FLAN-T5-large trained on one 8 GB consumer GPU
   beats the best cloud configuration by about 6 points.
2. RAG is a threshold capability: retrieval adds nothing to FLAN-T5-base or -large, but helps Gemini.
3. RAG and fine-tuning act as substitutes: adding retrieval to the LoRA models lowers accuracy
   (base 85.6 → 82.2, large 89.9 → 87.6 from k=0 to k=5).
4. Legal-domain pretraining improves the encoders by 2-3 points over BERT and RoBERTa under an identical
   fine-tuning budget.

All numbers are in `results/summary.json`; significance tests in `results/significance.json`.

## Repository layout

```
src/                     experiment and analysis scripts (run from inside src/)
  data_loader.py           CaseHOLD loading (Hugging Face datasets)
  baseline_sbert.py        SBERT embedding baseline
  baseline_bert.py         BERT / RoBERTa / Legal-BERT multiple-choice fine-tuning
  prompt_engineering.py    zero-shot / few-shot / chain-of-thought with FLAN-T5
  rag_pipeline.py          FAISS retrieval + FLAN-T5, pluggable retriever, optional LoRA adapter
  finetune_lora.py         LoRA fine-tuning of FLAN-T5
  gemini_experiments.py    Gemini 2.5 prompting and RAG through Vertex AI
  error_analysis.py, significance.py, semantic_score.py, citation_signals.py,
  attention_heatmaps.py, embedding_viz.py, evaluate_all.py, visualize_results.py
run_multiseed.py         extra-seed runs for the variance estimates
run_pending.py           sequential queue for long GPU runs
final_analysis.py        re-runs the whole evaluation and figure stack
results/                 per-experiment JSON results, figures, LoRA adapters
  *_lora_best/             trained LoRA adapters (PEFT format)
```

The full encoder checkpoints (`results/*.pt`, 438-499 MB each) are not included because of GitHub's
file-size limit; `baseline_bert.py` recreates them in roughly 3.5 hours each on an RTX 3070 (8 GB).

## Setup

Python 3.12 and an NVIDIA GPU with 8 GB+ VRAM for the local models.

```bash
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

PyTorch 2.6 or newer is required: Transformers 5.x refuses to load the legacy `pytorch_model.bin`
checkpoints that both Legal-BERT models ship with on older PyTorch versions.

## Running

All scripts run from `src/` and write to `../results/`. Examples:

```bash
cd src
python baseline_bert.py --model_name nlpaueb/legal-bert-base-uncased
python finetune_lora.py --model_name google/flan-t5-large --epochs 3 --batch_size 2 --grad_accum 2 --bf16 --gradient_checkpointing
python rag_pipeline.py --llm_model google/flan-t5-base --k_values 0,1,3,5
```

To evaluate the included FLAN-T5-large adapter without training, run the RAG pipeline with no retrieval:

```bash
python rag_pipeline.py --llm_model google/flan-t5-large --lora_adapter ../results/google_flan-t5-large_lora_best --k_values 0
```

This scores 89.9%. The headline 89.4% comes from the scoring inside `finetune_lora.py`; the two
decoding paths differ slightly, which the thesis discusses.

The Gemini experiments need a Google Cloud project with Vertex AI enabled and
`gcloud auth application-default login`; pass your own project with `--project`.
They are paid API calls.

## Citation

If you use this code, please cite the thesis:

> Kalampokis, C. (2026). *Improving Legal Judgment Prediction with Hybrid LLM Pipelines*.
> MSc thesis, Department of Informatics, University of Piraeus.

The data is CaseHOLD, loaded through the LexGLUE benchmark. If you use it, please also cite:

- Zheng, L., Guha, N., Anderson, B. R., Henderson, P., & Ho, D. E. (2021). When does pretraining help?
  Assessing self-supervised learning for law and the CaseHOLD dataset. ICAIL 2021.
- Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., & Aletras, N. (2022).
  LexGLUE: A benchmark dataset for legal language understanding in English. ACL 2022.
