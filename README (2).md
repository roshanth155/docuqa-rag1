# DocuQA: Document Question Answering with RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshanth155/docuqa-rag/blob/main/docuqa_rag.ipynb)

Ask questions about a PDF (here, a 57-page college rulebook) and get a one-sentence answer with page-level citations.
Questions the document cannot answer are refused instead of hallucinated.

## Pipeline

1. **Guard**: the best dense cosine similarity between the question and any chunk must exceed a threshold τ.
2. **Retrieve**: hybrid search (BM25 + `bge-small-en-v1.5` embeddings, z-score fusion) returns the top 20 chunks.
3. **Rerank**: a cross-encoder (`ms-marco-MiniLM-L-6-v2`) keeps the best 3.
4. **Generate**: `flan-t5-large` writes a one-sentence answer using only those 3 chunks.

## Results

Retrieval on 40 answerable questions (MRR, higher is better):

| Chunk size | BM25 | Dense | Hybrid | Rerank |
|---|---|---|---|---|
| 128 | 0.624 | 0.649 | 0.695 | **0.807** |
| 256 | 0.703 | 0.667 | 0.731 | 0.726 |
| 512 | 0.746 | 0.611 | 0.756 | 0.728 |

Rerank at 128 words is the most accurate (Recall@1 = 0.725) but about 5x slower than hybrid (60 ms vs 12 ms per query).
Full numbers are in `results.csv`.

Refusal guard (9 out-of-scope, 40 answerable questions):

| τ | Out-of-scope refused | Answerable wrongly refused |
|---|---|---|
| 0.60 | 77.8% | 0.0% |
| 0.65 | 88.9% | 2.5% |
| 0.70 | 100% | 12.5% |

See `threshold_table.csv`.

## Run it

1. Open the notebook in Colab (badge above) with a T4 GPU.
2. Upload `syllabus.pdf` and `qa.csv` (columns: `question`, `snippet`, `answerable`).
3. Run all cells. The last cell launches the Gradio demo.

Or locally: `pip install -r requirements.txt`, then run the notebook in Jupyter.

## Limitations

- Small evaluation set (40 answerable, 9 out-of-scope), so small differences between methods may not be significant.
- The refusal threshold was tuned on this one document and may need adjusting for others.
- Flan-T5 handles direct lookups well but can struggle with questions that need reasoning across pages.
- Retrieval is evaluated, but generated answer quality is not.
