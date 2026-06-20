# Automated Financial News Analysis and Recommendation Generation Using Deep Learning and Large Language Models 

A dual-pipeline NLP system that automatically classifies the sentiment of Bangladeshi financial news and generates investment recommendations for negative-sentiment articles, using fine-tuned BERT variants and seq2seq LLMs.

**Bachelor of Science Thesis** — Department of Computer Science and Engineering (CSE)
Faculty of Science and Technology (FST), American International University-Bangladesh (AIUB)
Fall 2025–2026 Semester

## Authors

| Name | Student ID |
|---|---|
| Tanvir Islam Akib | 22-47816-2 |
| Muzahidul Islam Saki | 22-47819-2 |
| Emon Das | 22-46599-1 |
| Mashudh Ahmed | 22-47811-2 |

**Supervisor:** Shaikat Das Joy, Lecturer, Department of Computer Science, AIUB

## Overview

The rapid growth of digital financial media has produced an enormous volume of unstructured text that is valuable but impractical to process manually. This project builds an end-to-end pipeline that:

1. **Scrapes** financial news from three major Bangladeshi sources.
2. **Classifies** each article's sentiment (Positive / Negative / Neutral) using fine-tuned BERT-family transformers.
3. **Generates** investment recommendations for articles classified as Negative using fine-tuned sequence-to-sequence language models.

This addresses a gap in financial NLP research, which has historically focused on Western financial corpora (e.g., Financial PhraseBank, FinBERT) and underrepresents South Asian, and specifically Bangladeshi, financial journalism.

## System Architecture

```
Phase 1: Data Acquisition
  └─ Scrape news from The Business Standard, The Daily Star, The Financial Express
        └─ Raw Financial News Dataset (~8,000 articles)

Phase 2: Preprocessing & Human Annotation
  └─ Cleaning, normalization, manual sentiment labeling
        └─ Labeled Master Dataset (1,700 articles)

        ┌───────────────────────────┐        ┌──────────────────────────────┐
        │ Branch A: Classification  │        │ Branch B: Recommendation Gen │
        │ Fine-tune BERT variants   │        │ Isolate Negative articles    │
        │ on 80/10/10 split         │        │ Pair with expert recs        │
        │                           │        │ Fine-tune T5/FLAN-T5         │
        └───────────────────────────┘        └──────────────────────────────┘
                    │                                       │
                    └──────────────► Evaluation ◄───────────┘
                                         │
                              Final Results & Conclusion
```

### Branch A — Sentiment Classification
Three BERT-family models were fine-tuned for 3-class sentiment classification (Positive / Negative / Neutral):
- `bert-base-uncased`
- `ProsusAI/finbert`
- `yiyanghkust/finbert-tone`

### Branch B — Recommendation Generation
Negative-sentiment articles were paired with domain-expert-authored recommendation text and used to fine-tune:
- `t5-small`
- `google/flan-t5-small`
- `TinyLlama-1.1B` (zero-shot baseline, no fine-tuning)

## Dataset

| Item | Count |
|---|---|
| Raw scraped articles | ~8,000 |
| Manually annotated sentiment dataset | 1,700 |
| Negative articles isolated for generation | 718 |
| Usable article–recommendation pairs (after cleaning) | 687 |

**Sentiment label distribution:** ~660 Positive, ~530 Neutral, ~510 Negative

**Dataset fields:**

| Feature | Description |
|---|---|
| `Text` | Financial news article content |
| `Sentiment_Label` | Positive / Neutral / Negative |
| `Source` | Daily Star, Financial Express, or TBS |
| `Date` | Publication date |
| `Recommendation_Text` | Human-authored ground-truth recommendation (Negative articles only) |

Sources: The Business Standard (TBS), The Daily Star, The Financial Express.

## Results

### Sentiment Classification (Branch A)

| Model | Test Accuracy | F1-Macro | F1-Weighted | ROC-AUC (OvR) |
|---|---|---|---|---|
| **bert-base-uncased** | **0.712** | **0.704** | **0.707** | **0.863** |
| ProsusAI/finbert | 0.676 | 0.675 | 0.677 | 0.842 |
| yiyanghkust/finbert-tone | 0.624 | 0.623 | 0.625 | 0.818 |

The general-purpose `bert-base-uncased` model outperformed both domain-specific FinBERT variants — suggesting that Bangladeshi financial news differs linguistically from the Western corpora FinBERT was trained on.

### Recommendation Generation (Branch B)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|
| **T5-Small (fine-tuned)** | 0.1740 | 0.0239 | **0.1417** ★ | 0.0135 |
| FLAN-T5 (fine-tuned) | 0.1456 | 0.0000 | 0.1140 | 0.0069 |
| TinyLlama (zero-shot) | 0.2362 | 0.0468 | 0.1231 | 0.0120 |

Fine-tuned T5-Small produced the most contextually coherent and actionable recommendations, despite TinyLlama's higher ROUGE-1 due to its larger pretrained vocabulary.

## Tools & Technologies

- **Language:** Python 3.10
- **Modeling:** Hugging Face Transformers, PyTorch
- **Data Handling:** Pandas, NumPy
- **Evaluation:** Scikit-learn, ROUGE Score, NLTK (BLEU)
- **Visualization:** Matplotlib, Seaborn
- **Scraping:** BeautifulSoup, Requests
- **Compute:** Google Colab, Kaggle (NVIDIA T4 / V100 GPUs)

## Repository Structure

```
.
├── data/
│   ├── raw/                  # Raw scraped news CSV
│   ├── labeled/              # 1,700-article annotated sentiment dataset
│   └── recommendations/      # 718-article negative + recommendation pairs
├── scraping/                 # Web scraping scripts (BeautifulSoup/Requests)
├── classification/           # Branch A: BERT fine-tuning & evaluation
├── generation/               # Branch B: T5/FLAN-T5/TinyLlama fine-tuning & evaluation
├── notebooks/                # Colab/Kaggle experiment notebooks
├── results/                  # Metrics, plots, confusion matrices, ROC curves
├── appendix/                 # Sample outputs, hyperparameter configs
└── README.md
```

> Adjust this structure to match your actual repository layout.

## Hyperparameters

| Parameter | Classification (BERT) | Generation (T5 / FLAN-T5) |
|---|---|---|
| Learning Rate | 2e-5 | 3e-4 |
| Batch Size | 16 | 2 (+ 4 grad. accumulation) |
| Epochs | 3 | 3 |
| Max Sequence Length | 128 | 512 / 256 (input / output) |
| Optimizer | AdamW | AdamW |
| FP16 | Yes | Yes |

## Key Contributions

- First manually annotated Bangladeshi financial news sentiment dataset (1,700 articles).
- Comparative benchmark of three BERT-family models on a South Asian financial corpus.
- A novel parallel corpus of 718 negative articles with expert-authored investment recommendations.
- Fine-tuned T5-Small / FLAN-T5-Small models benchmarked against a zero-shot LLM baseline.
- A reusable, fully documented dual-pipeline architecture for financial NLP in emerging markets.

## Limitations

- Sentiment dataset (1,700 articles) is modest compared to standard NLP benchmarks.
- Generation models were fine-tuned on a 300-sample subset due to GPU memory constraints.
- Low absolute ROUGE/BLEU scores reflect the open-ended nature of recommendation generation; human evaluation at scale was not performed.
- English-language news only — Bengali-language financial media was excluded.

## Future Work

- Expand the sentiment dataset with additional sources and Bengali-language articles.
- Apply active learning to reduce annotation cost while scaling the dataset.
- Fine-tune generation models on the full corpus with higher-capacity GPUs.
- Conduct large-scale human evaluation of generated recommendations.
- Extend to multi-label/mixed-sentiment detection.
- Deploy as a real-time scraping + inference API.
- Adapt the framework to other emerging-market financial domains (India, Pakistan, Africa).

## Ethical Considerations

Outputs from this system are intended as **decision-support tools, not autonomous investment advice**. Sentiment labels and recommendation texts were authored manually and may embed annotator/expert bias. All scraped data is publicly available; no personally identifiable information was collected.

## References

Key works referenced include Devlin et al. (2019, BERT), Araci (2019, FinBERT), Raffel et al. (2020, T5), Wei et al. (2022, FLAN), and Touvron et al. (2023, LLaMA). See the full thesis document for the complete reference list.

## License

Specify your license here (e.g., MIT, Apache 2.0) or note that this is academic research code shared for educational purposes.

## Acknowledgements

Department of Computer Science and Faculty of Science and Technology, AIUB. Thanks to the open-source community behind Hugging Face Transformers, Google Colab, and Kaggle for accessible models and compute.

