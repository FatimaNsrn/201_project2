# Farsi NLP Sequence Tagging Project

This project was developed as part of an **NLP course** and focuses on **sequence tagging tasks in Persian (Farsi)**.  
We compare **classical statistical models**, **neural networks**, and **Transformer-based architectures** on two core NLP tasks:

- Part-of-Speech (POS) Tagging  
- Named Entity Recognition (NER)

---

## Tasks Overview

### Part-of-Speech (POS) Tagging
Assigning grammatical labels (Noun, Verb, Adjective, etc.) to each token in a Persian sentence.

### Named Entity Recognition (NER)
Identifying and classifying named entities such as **Person**, **Location**, and **Organization**.

---

## Datasets

| Task | Dataset | Format | Notes |
|-----|--------|--------|------|
| POS Tagging | Farsi Universal Dependencies (UD) | `.conllu` | `PosData/fa_perdt-ud-*.conllu` |
| NER | PEYMA–ARMAN (Mixed) | Hugging Face Dataset | Requires FastText (`cc.fa.300.vec`) |

---

## Experimental Setup

- Python 3.8+
- GPU recommended (Google Colab or local CUDA setup)
- Multiple notebooks per model family

---

## Models

### Part-of-Speech (POS) Tagging Models

| Model | Methodology / Description |
|------|---------------------------|
| HMM | Classical generative statistical model that learns transition and emission probabilities from labeled data and uses the Viterbi algorithm for decoding. Serves as an interpretable baseline. |
| MEMM (Approx) | Feature-based discriminative model inspired by MEMM. Uses handcrafted linguistic features such as prefixes, suffixes, and orthographic patterns with a shallow neural classifier. |
| Bi-LSTM + Softmax | Neural sequence labeling model using word embeddings and bidirectional LSTM layers to capture both past and future context. Tags are predicted independently using a Softmax layer. |

---

### Named Entity Recognition (NER) Models

| Model | Methodology / Description |
|------|---------------------------|
| CRF | Feature-based statistical sequence model that enforces global label consistency using handcrafted lexical and contextual features. |
| Char-CNN + BiLSTM (Softmax) | Hybrid neural model combining character-level CNNs for morphology and OOV handling with a Bi-LSTM for contextual representation, decoded using Softmax. |
| Char-CNN + BiLSTM + CRF | Enhanced hybrid architecture that adds a CRF layer on top of Bi-LSTM outputs to enforce valid label transitions and improve sequence-level predictions. |
| ParsBERT Fine-Tuning | Transformer-based encoder model pre-trained on large-scale Persian text and fine-tuned for token classification, leveraging rich contextual representations. |
| T5 Fine-Tuning (Seq2Seq) | Encoder–decoder Transformer model fine-tuned to generate NER tag sequences as text outputs, framing NER as a sequence-to-sequence task. |


---

## Results Summary

### POS Tagging Results

| Model | Model Type | Metric | Score |
|-----|-----------|-------|-------|
| Bi-LSTM + Softmax | Neural (State-of-the-Art) | Accuracy | **95.16%** |
| HMM | Statistical Baseline | Accuracy | 91.01% |
| MEMM (Approx) | Feature-based Neural | Accuracy (Dev) | 85.89% |

---

### Named Entity Recognition (NER) Results

| Model | Model Type | Metric | Score |
|-----|-----------|-------|-------|
| ParsBERT Fine-Tuning | Transformer (SOTA) | Micro F1 | **95.12%** |
| Char-CNN + BiLSTM + CRF | Hybrid Deep Learning | Micro F1 | 90.23% |
| Char-CNN + BiLSTM (Softmax) | Hybrid Deep Learning | Micro F1 | 88.75% |
| CRF | Statistical Baseline | Micro F1 | 87.11% |
| T5 Fine-Tuning | Transformer (Seq2Seq) | Accuracy | 87.92% |

---

## Analysis

### POS Tagging
- Bi-LSTM achieved the highest accuracy by effectively capturing bidirectional context.
- HMM provided a strong and interpretable statistical baseline.
- MEMM approximation was limited by reliance on handcrafted features and lack of deep contextual modeling.

### Named Entity Recognition
- ParsBERT achieved state-of-the-art performance due to Farsi-specific pretraining.
- Adding a CRF layer significantly improved label consistency.
- Character-level modeling helped handle morphological richness and OOV words.
- Seq2Seq (T5) struggled with strict tag sequence generation.

---

## Dependencies

### Statistical & Basic Neural Models (POS)
```bash
pip install pyconll scikit-learn numpy tensorflow keras hmmlearn
