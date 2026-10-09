# KM-KENet — Kurdish Medical Knowledge Extraction Network

KM-KENet takes the **Kurdish Medical Corpus (KMC)**, written in Central Kurdish (Sorani), and builds a
medical knowledge graph from it. You can then ask questions in English (or Kurdish) and get answers in
English that are based on the Kurdish sources.

```
Data Collection ─► Data Pre-processing ─► Weak Supervision Labeling ─► KM-KENet Model Training
 (KMC corpus)       (cleaning,             (dictionary / pattern /       (XLM-R fine-tuning:
                     normalization,         heuristic LFs → noisy          NER + Relation Extraction)
                     tokenization)          labels → Dawid–Skene →                  │
                                            refinement → dataset)                   ▼
             Cross-Lingual Query Translation and Retrieval ◄──── Knowledge Graph Construction
             (Aya multilingual model, BM25 + dense + KG)          (structured medical KG)
```

## Project layout

| Folder | Step | Main outputs |
| --- | --- | --- |
| `step1_data_collection/` | Load KMC instruction/response JSON and book text, plus optional local/web/Hugging Face sources | `data/interim/corpus.jsonl` |
| `step2_preprocessing/` | Cleaning, Sorani orthographic normalization, sentence splitting, script filter, tokenization | `data/processed/sentences.jsonl` |
| `step3_weak_supervision/` | Dictionary, pattern and heuristic labeling functions → label matrix → Dawid–Skene label model → refinement → relation labels | `data/weak_labels/{train,dev,test}.jsonl` |
| `step4_model_training/` | KM-KENet: an XLM-R encoder with a joint NER head and a relation-extraction head | `models/km_kenet/` |
| `step5_knowledge_graph/` | Model inference, entity canonicalization, triple aggregation | `outputs/knowledge_graph/` (JSON, GraphML, CSV, Cypher, PNG) |
| `step6_cross_lingual_retrieval/` | Query translation with Aya, hybrid retrieval, grounded answers | `outputs/answers/*.json` |
| `setup/` | Downloads the KMC dataset and the pre-trained models (`python -m setup.download_assets`) | `../Kurdish_Medical_Corpus_KMC/`, Hugging Face cache |
| `common/` | Configuration, logging, I/O, annotation schema, lexicon loading | — |
| `resources/` | Sorani medical lexicon, stopwords, English–Kurdish glossary | — |

**Entity types:** `DISEASE`, `SYMPTOM`, `TREATMENT`, `BODY_PART`, `TEST`.
**Relation types:** `TREATS`, `HAS_SYMPTOM`, `AFFECTS`, `DIAGNOSED_BY`.

## Dataset

The pipeline is built around **`kurdish_medical_corpus_kmc.json`** (KMC V1, 121,918 records,
[`shkomq/Kurdish_Medical_Corpus_KMC`](https://huggingface.co/datasets/shkomq/Kurdish_Medical_Corpus_KMC)).
It expects the dataset next to the project folder, at `../Kurdish_Medical_Corpus_KMC/` (`paths.corpus_root`).
Step 1 handles the following details of this file:

| Dataset detail | How it is handled |
| --- | --- |
| A `/* ... */` licence header before the JSON array | Removed before parsing |
| `{id, instruction, response, document{title, source_url, publication_date, domain}, language{code, region}}` | Each `response` becomes a document; the `id` (`CKB-MED-…`) is kept as `doc_id` |
| ~2.4k records with an empty `document.title` | The group name falls back to the PDF file name in `source_url`, so train/dev/test splits stay leak-free per source book |
| Questions about the book itself (author, printing, deposit number) | Skipped using `kmc_json.exclude_instruction_patterns` |
| 759 duplicated responses | Exact de-duplication |
| Markdown (`**bold**`, `*   ` bullets) | Removed by the step 2 cleaner |
| Arabic code points (`ك`, `ي`, `ه‌`) | Normalized to standard Sorani letters |
| English drug/disease names inside Sorani text (`Amoxicillin`, `Insulin`, …) | Lower-cased and matched against the English names in the lexicon |

## Installation

```bash
./install.sh                     # venv + PyTorch + requirements + dataset + models
./install.sh --cuda cu121        # PyTorch built for a specific CUDA version
./install.sh --skip-aya          # skip the ~16 GB Aya model (step 6 then uses its fallback)
```

`install.sh` creates `.venv`, installs the packages, checks the imports and the available devices
(CUDA/MPS), and then runs `python -m setup.download_assets`. That command downloads:

| Asset | Source (from `configs/config.yaml`) | Used by |
| --- | --- | --- |
| Dataset | `data_collection.kmc_json.hf_repo_id`, JSON only (the 1.3 GB of PDFs are skipped) | Step 1 |
| Encoder | `model.encoder` = `FacebookAI/xlm-roberta-base` | Step 4 |
| Embeddings | `retrieval.embedding_model` = `intfloat/multilingual-e5-base` | Step 6 |
| Aya | `retrieval.aya.hf_model` = `CohereLabs/aya-expanse-8b` (gated) | Step 6 |

Models are saved in the Hugging Face cache (`$HF_HOME`). After the download you can run fully offline with
`export HF_HUB_OFFLINE=1`. Aya is gated: first accept its licence on its Hugging Face page, then set
`export HF_TOKEN=hf_...` (or run `huggingface-cli login`). Files that are already present are not
downloaded again.

```bash
python -m setup.download_assets --only dataset    # one asset
python -m setup.download_assets --skip aya        # everything except Aya
```

## Running

```bash
./run.sh --venv .venv                     # all six steps
./run.sh --download                       # download assets first, then run
./run.sh --from 4                         # resume from model training
./run.sh --only 6 -q "How is thalassemia treated?"
./run.sh --override training.epochs=2 --override retrieval.aya.backend=none
```

You can also run each step on its own from the project root:

```bash
python -m step3_weak_supervision.main --config configs/config.yaml
python -m step6_cross_lingual_retrieval.main --interactive
```

## Hardware notes

* **Step 4** fine-tunes XLM-R. A CUDA GPU is strongly recommended. Apple Silicon (MPS) and CPU also
  work but are slower. `training.max_train_examples` limits how many training examples are used.
* **Step 6** uses Aya (`CohereLabs/aya-expanse-8b` by default). This model is gated: accept its licence on
  Hugging Face and export `HF_TOKEN`. It needs a GPU; for smaller GPUs, set `retrieval.aya.load_in_4bit: true`.
  You can also set `backend: cohere` and export `COHERE_API_KEY` to use the Cohere API. If Aya is
  unavailable, step 6 still runs: it translates queries with the lexicon and builds the answer from a
  template using the knowledge-graph facts.

## Extending the knowledge

To add terms, edit `resources/medical_lexicon.tsv` (Kurdish term, English gloss, type, and aliases separated
by `|`). New constructions can be added as labeling functions in
`step3_weak_supervision/labeling_functions/`. After a change, rerun from step 3.
