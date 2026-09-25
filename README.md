# LLM FrontEnd

Generate LLM responses to original and perturbed questions, select a representative response using embeddings, and evaluate response consistency and quality. The repository also includes a graphical tool for manually reviewing dataset statements.

The current generation loop makes **19 model calls per question**: one original question and 18 typo-perturbed variants. Although the internal parameter is named `synonyms`, the active perturbation is `apply_typo(text, 0.09)`.

## Setup
Clone this repo: "https://github.com/yuh-zha/AlignScore.git"
Run commands from the repository root so relative paths resolve correctly. Use a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pandas datasets tqdm scikit-learn \
  transformers sentence-transformers huggingface-hub sentencepiece \
  python-dotenv requests spacy typo ollama openai together google-genai \
  bert-score matplotlib
python -m pip install ./AlignScore
python -m spacy download en_core_web_sm
```

These packages follow the source imports; the project does not provide a pinned, tested environment for the entire pipeline. AlignScore has its own dependency constraints in [AlignScore/pyproject.toml](AlignScore/pyproject.toml).

The hallucination metric loads two models from the local Hugging Face cache. Download them once while online:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
snapshot_download("cross-encoder/nli-deberta-v3-base")
PY
```

### API-key files

The generation script's imports read these files before argument parsing, including when using Ollama or requesting `--help`:

```text
../API_Key/openai.txt
../API_Key/together.txt
../API_Key/gemini.txt
../API_Key/MerriamWebster.txt
```

Create the directory and files locally, placing the relevant key in each provider's file. Files for unused services can be empty; valid credentials are required for services you call. Keep credentials out of Git. Existing environment variables alone do not bypass these file reads.

The `--API_Key` option is accepted, but the current synonym module reads `../API_Key/MerriamWebster.txt` directly. The active typo perturbation does not call the thesaurus service.

## Run `LLM_FrontEnd.py`

### Prepare the dataset

The default dataset option is `mbzuai_expanded`. In `load_process_data()` inside [LLM_FrontEnd.py](LLM_FrontEnd.py), replace the hard-coded path in that branch with your local CSV path. Its current value is:

```text
/Users/saeedalahmari/Documents/LLM_ensemble_USF/code/LLMFrontEnd/mbzuai_extended_with_correct.csv
```

The CSV must contain `question` and `category`; an `id` column is optional:

```csv
id,question,category
0,Which country won the first World Cup in 1922?,did_not_happen
1,What is the capital of France?,correct
```

`--dataset` selects a named branch; it does not accept an arbitrary CSV path. The `mbzuai` branch loads `MBZUAI/LaMini-Hallucination`, but its Hugging Face split must be converted to a pandas DataFrame before passing it to `generate_responses()`, which uses `.head()` and `.iterrows()`. Other dataset branches also need their columns and prompt adapted to the current `question`/`category` workflow.

### Generate with Ollama

Install and start Ollama, then download a model recognized by the dispatcher:

```bash
ollama pull gemma3:12b
```

If Ollama is not already running as an application/service, run `ollama serve` in another terminal. To use the cached MiniLM model for embeddings:

```bash
export EMBEDDING_BACKEND=sentence-transformers
python LLM_FrontEnd.py \
  --dataset mbzuai_expanded \
  --LLM gemma3:12b \
  --save_results ./results
```

The default embedding backend is `ollama`, using `phi4-mini`, independently of the generation model selected with `--LLM`. Its fallback embedding server may require `OLLAMA_LLAMA_SERVER` to point to a local `llama-server` executable. The sentence-transformers setting above avoids that server requirement.

### Generation models recognized by the code

Pass the exact identifier to `--LLM`. These are the routing entries in [runModel_API.py](runModel_API.py), not a guarantee of current hosted-provider availability or account access.

| Backend | Recognized model identifiers |
| --- | --- |
| Ollama | `gemma3:4b`, `gemma3:12b` (default), `phi4:latest`, `phi4-mini`, `wao/phi4-mini-instruct:latest`, `ministral-3:3b` |
| OpenAI | `gpt-5`, `gpt-4o`, `gpt-5-nano`, `o4-mini-2025-04-16` |
| Together | `openai/gpt-oss-20b`, `google/gemma-4-31B-it`, `ssalahmari/google/gemma-3-12b-it-fcdf3056`, `google/gemma-3n-E4B-it`, `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`, `mistralai/Ministral-3-14B-Instruct-2512` |

For example, after setting up the Together key file:

```bash
EMBEDDING_BACKEND=sentence-transformers python LLM_FrontEnd.py \
  --dataset mbzuai_expanded \
  --LLM openai/gpt-oss-20b \
  --save_results ./results
```

Custom Together deployment identifiers require access to that deployment. Unlisted model identifiers require adding a routing entry in `run_model()`. Names containing `gemini` initialize a Gemini client, but the active generation branch uses the OpenAI-style completion interface; Gemini generation needs a code change before use.

### Generation arguments and outputs

| Argument | Default | Purpose |
| --- | --- | --- |
| `--dataset` | `mbzuai_expanded` | Dataset branch to load |
| `--LLM` | `gemma3:12b` | Generation model identifier |
| `--save_results` | `../results` | Output directory |
| `--API_Key` | `../API_Key/MerriamWebster.txt` | Legacy thesaurus key argument; see setup note |

Generation writes a timestamped response CSV and a metrics text file. The CSV includes `sentence`, `text_edited`, `LLM_response` (a serialized response list), `center_response`, `label_gt`, and similarity/hallucination metrics. The first response is the clean response; the remaining responses are perturbed variants.

**Current prompt caveat:** the generation prompt lists `correct` among categories that should produce `True`. Boolean analysis maps `correct` to `False`. Before a boolean evaluation run, remove `correct` from the prompt's True-category list so generation and scoring use the same definition. The current generation CLI does **not** expose `--true-false-output`; use that flag with `analysis.py`.

## Run `analysis.py`

Analysis reads a generated response CSV, computes metrics, **updates that CSV in place**, prints means and population standard deviations, and saves a combined box plot. Copy the CSV first if you want to preserve the original metrics.

Required columns are `LLM_response`, `center_response`, and `label_gt`. Each `LLM_response` cell must contain a Python-style serialized list, for example `['True: never happened', 'False: known event']`.

### Download an AlignScore checkpoint

The bundled [AlignScore documentation](AlignScore/README.md) lists base and large checkpoints. For the default `roberta-base` backbone:

```bash
curl -L --fail \
  https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt \
  -o ./AlignScore/AlignScore-base.ckpt
```

### Analyze True/False responses

Replace `./results/responses.csv` with the CSV produced by generation:

```bash
python analysis.py \
  --csv_file ./results/responses.csv \
  --alignscore-checkpoint ./AlignScore/AlignScore-base.ckpt \
  --true-false-output \
  --response-position-output \
  --device cpu
```

`--true-false-output` adds clean/label and center/label accuracy using this mapping:

| Dataset label | Expected answer |
| --- | --- |
| `did_not_happen`, `far_future`, `nonsense`, `obscure` | `True` |
| Every other label, including `correct` | `False` |

The scorer uses `category` when present, otherwise `label_gt`. True/False matching is case-insensitive and allows accompanying justification. A nonempty response containing neither value or both values is incorrect. Empty responses are excluded from the relevant averages.

Boolean scoring is additional to the text metrics; it still requires the metric models and AlignScore checkpoint.

### Analyze category or free-text responses

For responses beginning with an MBZUAI category, enable category accuracy:

```bash
python analysis.py \
  --csv_file ./results/responses.csv \
  --alignscore-checkpoint ./AlignScore/AlignScore-base.ckpt \
  --dataset-name mbzuai
```

Category mode validates labels against the four underscore-form categories above. For general free-text evaluation, omit `--dataset-name` and `--true-false-output`. A CSV whose `dataset_name` column identifies MBZUAI automatically enables category mode unless boolean mode is selected.

### Analysis options and outputs

| Argument | Purpose |
| --- | --- |
| `--csv_file` | Required input CSV; updated in place |
| `--alignscore-checkpoint` | Required local checkpoint path |
| `--alignscore-model` | `roberta-base` (default) or `roberta-large`; must match checkpoint |
| `--bertscore-model` | Optional BERTScore model override |
| `--batch-size` | Inference batch size; default `32` |
| `--device` | `auto` (default), `cpu`, `cuda:0`, or `mps` |
| `--true-false-output` | Enable boolean accuracy |
| `--dataset-name` / `--dataset_name` | Enable MBZUAI category accuracy |
| `--boxplot-file` | Optional PNG/PDF output path |
| `--response-position-output [PATH]` | Also save per-position metric means; optional output CSV path |

Metrics include hallucination score, BERTScore F1, AlignScore, and BLEU for clean/label, center/label, and center/clean comparisons. Enabled accuracy columns are `accuracy_clean_vs_label` and `accuracy_center_vs_label`.

With `--response-position-output`, the separate summary contains `C` (clean), `P1` through the last perturbed position, and `Center`, including `mean_accuracy` when enabled. Without an explicit path, it is saved as `<input_stem>_response_position_metrics.csv`. The default plot is `<input_stem>_metrics_boxplot.png`.


