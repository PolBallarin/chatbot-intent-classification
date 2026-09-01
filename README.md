# Spanish intent classification for a caregiving assistant

Master's thesis (Big Data & AI). Personal project, sole developer.
The training pipeline, the raw result files and the deployment artifacts are all in this repository.
The app they were built for is private; paths under `functions/` refer to that private codebase.

Five intent classifiers trained on 118,445 synthetic Spanish utterances across 12 intents. The
best of them was exported to ONNX and wired into a Firebase callable function. The interesting
part of this repository is not the accuracy table — it is the audit that shows why the accuracy
table is close to meaningless.

## TL;DR

- Naive Bayes, Random Forest, LinearSVC, a BiLSTM and multilingual BERT all land between
  **99.39% and 99.57%** test accuracy. Twenty-one errors separate best from worst, over 11,845
  test examples. Their 95% Wilson intervals overlap completely, so the ranking is not defensible.
- I then measured what the test set actually contains: **98.66% of test examples were generated
  from a template that also appears in train**, and **4 of 103,574 test tokens** are out of
  vocabulary. The headline number measures template memorisation, not generalisation.
- **52 of the BiLSTM's 53 errors** fall on one template string emitted under two different labels
  by my own generator (`build_dataset.py:407` and `build_dataset.py:582`). The residual error is a
  dataset bug, not a modelling limit.
- The BiLSTM was exported to ONNX and served from a Cloud Function. That deployment has a real,
  reproducible **train/serve tokenisation skew**, measured in section 7. It is **not merged into
  the published app**.

---

## 1. The task

Plamily is a Flutter app I built for families coordinating the care of an older relative:
medication schedules, reminders, a shared shopping list, contacts. The thesis asks whether a text
(later voice) entry point can collapse a multi-screen flow into one sentence —
*"ya me he tomado el paracetamol"*.

The pipeline splits into three stages, and only the first one is the thesis's subject:

| Stage | How it is solved | In this repo |
|---|---|---|
| Intent classification | Trained model, 12 classes | Yes — the whole comparison |
| Entity extraction | Claude Haiku API call (`functions/index.js:3565`) | Deployed, not benchmarked |
| Action execution | Firestore writes, per intent (`functions/index.js:3742`) | Deployed, not benchmarked |

This matters for reading the results honestly: **the trained model does intent classification only**.
Entity extraction is delegated to an LLM at request time. The original plan included a NER head; it
was not built.

### The 12 intents

`build_dataset.py` asks each generator for a target count and stops early when intra-intent
deduplication exhausts the template space. Five intents hit that ceiling, which is where the class
imbalance comes from:

| Intent | Target | Generated | Test support | What it does in the app |
|---|---:|---:|---:|---|
| `add_medication` | 20,000 | 20,000 | 2,038 | Creates a medication with dose, frequency and schedule |
| `add_reminder` | 20,000 | 20,000 | 2,019 | Creates a dated reminder for a tracked person |
| `mark_medication_taken` | 15,000 | 15,000 | 1,446 | Marks the pending dose of the day as taken |
| `add_contact` | 15,000 | 15,000 | 1,506 | Adds a phone contact to a tracked person |
| `add_shopping_item` | 15,000 | 15,000 | 1,529 | Appends items to the shared shopping list |
| `list_medications` | 10,000 | 10,000 | 1,014 | Lists active medications for a person |
| `check_medication_status` | 10,000 | **5,802** | 565 | Reports which doses are still pending today |
| `mark_shopping_done` | 10,000 | **8,096** | 791 | Marks a shopping item as bought |
| `list_reminders` | 8,000 | 8,000 | 791 | Lists upcoming reminders |
| `list_shopping` | 5,000 | **460** | 45 | Shows the pending shopping list |
| `greeting` | 3,000 | **607** | 44 | Direct canned reply |
| `help` | 3,000 | **480** | 57 | Direct canned reply |
| **Total** | 134,000 | **118,445** | 11,845 | |

Class imbalance is **43.5x** (`add_medication` 20,000 vs `list_shopping` 460). Random Forest and
LinearSVC use `class_weight="balanced"` (`train_random_forest.py:80`, `train_svm.py:82`); Naive
Bayes and the two neural models do not compensate at all. This is why **F1-macro, not accuracy, is
the metric to read** in the table below.

---

## 2. The dataset is 100% synthetic

There is no way around stating this first, because it is the first question any reviewer should ask.

**Why synthetic.** The app had no conversational surface, so there was no log of real user phrasing
to mine — a cold start with zero in-domain data. Two options were on the table:

1. Generate with an LLM (`generate_dataset.py`, `claude-sonnet-4-20250514` at line 326).
   Estimated ~$80 for the target volume. **Never executed** — discarded on cost
   (`TFM_PROGRESS.md:196`).
2. Generate locally from hand-written templates plus randomised entity slots
   (`build_dataset.py`, 969 lines). Free, instant, deterministic. **This is what produced the data.**

**How it works.** Twelve generator functions compose f-string templates with curated entity pools —
73 person references (`"mi abuela"`, `"la yaya"`, `"mamá"`, proper names), 67 medication names
including colloquial ones (`"la pastilla de la tensión"`), 90 products, 25 stores, 41 times, 34
dates. An 8% probability typo/orthography corruption pass (`build_dataset.py:217`) simulates the
accent-dropping and abbreviation habits of mobile Spanish input (`"qué"` → `"que"`,
`"por favor"` → `"xfa"`).

Reconstructing every example's skeleton (substituting entity values back into placeholders) gives
**2,424 distinct `(intent, skeleton)` pairs** for 118,445 examples — an average of 49 examples per
skeleton. Restricted to the nine entity-bearing intents, it is 1,337 skeletons for 117,358
examples, or 88 examples per skeleton. Section 4.2 is the direct consequence.

**Reproducibility.** `random.seed(42)` at `build_dataset.py:14`, no API calls, no external state.
I re-ran the generator from a clean checkout while writing this README and got the identical split,
example for example, in 1.5 s:

```
TOTAL: 118,445 examples generated
  train: 94,756   val: 11,844   test: 11,845
```

**What the data is not.** It is not real user speech. Vocabulary, sentence length and syntactic
variety are bounded by what I thought of while writing the templates. Every claim in this
repository is bounded by that ceiling too.

---

## 3. The five models

Trained on an i9 / 64 GB / RTX 4060 workstation (`TFM_PROGRESS.md:108`). Classical models on CPU,
neural models on GPU.

| Model | Config | Accuracy | F1-macro | Errors /11,845 | Params | Train time |
|---|---|---:|---:|---:|---:|---:|
| Naive Bayes + TF-IDF | `MultinomialNB(alpha=0.1)`, 1-2 grams, 50k features | 0.9939 | 0.9890 | 72 | — | **0.036 s** |
| Random Forest + TF-IDF | 200 trees, balanced | 0.9942 | 0.9896 | 69 | — | 6.81 s |
| SVM (LinearSVC) + TF-IDF | `C=1.0`, `max_iter=10000`, balanced | 0.9949 | 0.9912 | 60 | — | 4.94 s |
| **BiLSTM** | 2x256 hidden, 128-dim emb., dropout 0.3, 15 epochs (best: 6) | 0.9955 | **0.9941** | 53 | 2,550,796 | 131 s |
| BERT multilingual | `bert-base-multilingual-cased`, lr 2e-5, 3 epochs (best: 2) | **0.9957** | 0.9937 | 51 | 177,862,668 | 3,276 s |

All three TF-IDF vectorisers keep accents (`strip_accents=None`) — an explicit choice for Spanish.
The BiLSTM vocabulary is built from train only, with `MIN_WORD_FREQ = 2`, tokenised as
`text.lower().split()` (`train_bilstm.py:77-88`).

Per-intent, the BiLSTM scores **F1 = 1.0000 on nine of the twelve classes**. A perfect score on
nine classes is not an achievement; it is a symptom, and it is what sent me looking.

---

## 4. Why these numbers don't mean what they look like

This is the section the thesis is actually about.

### 4.1 The ranking is statistically empty

Best minus worst is **21 errors out of 11,845**. Wilson 95% intervals:

| Model | Accuracy | Wilson 95% CI |
|---|---:|---|
| Naive Bayes | 99.39% | [99.24, 99.52] |
| Random Forest | 99.42% | [99.26, 99.54] |
| SVM | 99.49% | [99.35, 99.61] |
| BiLSTM | 99.55% | [99.42, 99.66] |
| BERT | 99.57% | [99.43, 99.67] |

Every interval overlaps every other one. Naive Bayes's upper bound (99.52) sits above BERT's lower
bound (99.43). **"BERT is the best model" is not a claim this experiment can support.** Any thesis
that reports the ordering as a finding is over-reading its own data.

### 4.2 98.66% of the test set leaks from train

The generator produces all 118,445 examples first, then applies a flat `random.shuffle()` before
slicing 80/10/10 (`build_dataset.py:945-955`). Templates are not held out, and with an average of
49 examples per skeleton, essentially every skeleton lands on all three sides of the split.

Measured by reconstructing each example's skeleton and intersecting the sets:

```
distinct (intent, skeleton) pairs in train       : 2,105
test examples whose skeleton is already in train : 11,686 / 11,845 = 98.66%
```

This is leakage of phrasing, not of rows: deduplication is exact-string within each intent, so only
**16 of 11,845** test examples appear verbatim in train. The model has not memorised the test rows;
it has memorised the 2,105 sentence patterns they were stamped from.

Of the 159 test examples whose skeleton is genuinely new, 146 belong to the three intents with no
entity slots (`help` 57, `list_shopping` 45, `greeting` 44), where skeleton and text are the same
string by construction. That leaves **13 examples out of 11,845** across the nine entity-bearing
intents that present phrasing the model has never seen.

### 4.3 The vocabulary leaks too

```
test tokens                                   : 103,574
tokens unseen in train                        :       4  (0.004%)
test examples containing zero unseen tokens   : 11,841 / 11,845  (99.97%)
```

The four unseen tokens are `dias!`, `dia?`, `ola!`, `saludos!!` — punctuation artefacts, not new
words. (Using the model's real vocabulary, which drops hapaxes at `MIN_WORD_FREQ = 2`, the count
rises to 7 out of 103,574 — same conclusion.)

**Together, 4.2 and 4.3 say the test set is a memorisation probe.** A model that stores
skeleton→label pairs and does nothing else scores about 98.7% here. That is the floor these five
models are clustered just above, which is exactly what the 21-error spread looks like.

### 4.4 Most of the remaining error is my own label noise

Deduplication in `generate_intent()` uses a `seen_texts` set that is **local to each intent**
(`build_dataset.py:883`). Nothing checks for the same string being emitted under two different
labels. Measured across the full dataset:

```
texts appearing with contradictory labels : 87
  (list_medications, list_reminders)      : 86
  (greeting, help)                        :  1
```

All 86 come from one template — `"¿Qué le toca a {person} {date}?"` — written into
`gen_list_medications` at **`build_dataset.py:407`** and, verbatim, into `gen_list_reminders` at
**`build_dataset.py:582`**. The sentence is genuinely ambiguous in Spanish ("what's due for X on
Y?" — medication or appointment?), and I labelled it both ways. Across the whole dataset that
skeleton carries the `list_medications` label 501 times and `list_reminders` 476 times.

The BiLSTM's confusion matrix contains exactly two off-diagonal cells:

```
list_medications -> list_reminders : 52
greeting         -> list_reminders :  1
```

| Intent | Precision | Recall | Support |
|---|---:|---:|---:|
| `list_medications` | 1.0000 | 0.9487 | 1,014 |
| `list_reminders` | 0.9372 | 1.0000 | 791 |
| `greeting` | 1.0000 | 0.9773 | 44 |

The test split contains **exactly 52** `list_medications` examples built from the colliding
skeleton, and the model makes **exactly 52** `list_medications` errors, all of them predicted as
`list_reminders`. The confusion matrix alone does not identify examples one by one, but the counts
coincide and the mechanism is visible in the generator source. **52 of the model's 53 errors sit on
label noise I injected myself**, on a sentence with no correct answer. The 53rd is a single
`greeting` example, also predicted as `list_reminders`, which the collision does not explain.

The BiLSTM is not 99.55% accurate because it is good. It is 99.55% accurate because 98.66% of the
test set is memorisable and most of the remaining 1.34% is a labelling contradiction.

### 4.5 Minority classes have almost no test support

`greeting` has 44 test examples, `list_shopping` 45, `help` 57. One error moves `greeting`'s recall
by 2.3 points. F1-macro — the metric I argue is the right one under 43.5x imbalance — is therefore
dominated by three classes whose measurement is very noisy. That is a second reason not to read the
macro ranking as a fine-grained ordering.

### 4.6 What I would do differently

1. **Group split by template.** Assign each `(intent, skeleton)` to exactly one of train/val/test
   *before* filling entity slots, using `GroupShuffleSplit` on the skeleton id. Then test accuracy
   measures generalisation to unseen phrasing, which is the quantity the app actually needs.
2. **Global deduplication with conflict detection.** Move `seen_texts` out of `generate_intent()`,
   make it dataset-wide, and hard-fail on any string emitted under two labels. That surfaces the
   `"¿Qué le toca a...?"` collision at generation time instead of in the confusion matrix.
3. **Merge or disambiguate the colliding intents.** `list_medications` and `list_reminders` need
   either a shared "what's scheduled" intent with a downstream disambiguator, or templates that are
   actually distinguishable.
4. **Report McNemar's test** between model pairs rather than a raw accuracy ordering.

**My expectation — untested, and labelled as such:** under a template-level group split I expect
all five models to drop materially, and I expect the gap between them to open up, with the
embedding-based models (BiLSTM, mBERT) degrading less than TF-IDF n-grams on unseen phrasings,
since a bag of unseen bigrams carries no signal. I have not run this experiment, so this is a
hypothesis, not a result.

---

## 5. What the comparison does show

Stripping out the ranking, two conclusions survive the audit.

**A 178M-parameter transformer does not beat a 36-millisecond Naive Bayes on this task.** BERT
costs **3,276 s of GPU training (55 min) and 69.7x more parameters** than the BiLSTM to buy
**0.017 percentage points** of accuracy — two errors out of 11,845, well inside the noise — while
*losing* on F1-macro. Against Naive Bayes it buys 21 errors for a 92,000x increase in training
time. For a bounded, closed-set intent vocabulary, model capacity is not the binding constraint;
the data is.

**The BiLSTM is the right pick under imbalance.** It wins F1-macro (0.9941 vs BERT's 0.9937) while
being 25x faster to train and 69.7x smaller. Under 43.5x class imbalance, macro-averaged F1 is the
metric that refuses to let the three large intents mask the small ones. The BiLSTM was selected for
deployment on that basis plus artefact size: 9.7 MiB of ONNX ships inside the function bundle,
where a 178M-parameter model would not.

**Caveat, stated plainly.** The size and F1-macro arguments are sound; the *latency* argument in
the thesis is not measured correctly. `avg_latency_ms` is computed as full-batch inference wall
time divided by N (`train_svm.py:99`, `train_bilstm.py:265`), i.e. amortised throughput, not
per-request latency in a serverless container. That is why SVM appears to run in 0.0002 ms per
example. **There is no benchmark of the deployed ONNX session, and no cold-start measurement.** The
correct experiment is per-request `onnxruntime-node` latency on a 512 MiB instance, warm and cold.
It was not run, so the claim that the smaller artefact helps cold start is reasoning, not data.

---

## 6. Deployment

### Export

```python
torch.onnx.export(
    model, dummy_input, "models/bilstm_model.onnx",
    input_names=["input_ids"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14,
)
```
`convert_to_onnx.py:82-93`

Result: `input_ids` INT64 `[batch, 50]` → `logits` FLOAT `[batch, 12]`, a 39-node graph.
**10,207,804 bytes (9.7 MiB)** of ONNX plus a **22 KB** vocabulary JSON (1,384 entries including
`<PAD>`/`<UNK>`, 12 labels, `max_seq_len: 50`). Both files are committed under
`deployment/` and read with `fs.readFileSync` from `__dirname`, so the function downloads
nothing at cold start.

### Serving

```js
exports.chatbot = onCall({
  region: "europe-west1",
  enforceAppCheck: true,
  secrets: [anthropicApiKey],
  memory: "512MiB",
  timeoutSeconds: 30,
}, async (request) => { ... });
```
`functions/index.js:4071-4076`

- `onnxruntime-node ^1.21.0` on Node 24, Firebase Functions v2 callable.
- The `InferenceSession` is held in a module-scope variable and created lazily
  (`functions/index.js:3463-3471`), so only the first invocation on a container pays session setup.
- App Check enforced, plus an explicit `request.auth` check. The caller's `currentHomeId` is read
  once (`functions/index.js:4119`) and every subsequent read and write is scoped under
  `homes/{homeId}/...`.
- Softmax is computed in JS with the standard max-subtraction for numerical stability
  (`functions/index.js:3493-3496`), and the top-3 alternatives are logged to Firestore.

### Routing: rules first, model second, override third

The model is not trusted to trigger writes on its own. `deployment/router.js` (234 lines) is
a pure module with no `require` of Firebase, ONNX or the network, holding four confidence
thresholds (`router.js:3-8`):

| Threshold | Value | Meaning |
|---|---:|---|
| `rule` | 0.98 | A deterministic rule matched |
| `override` | 0.95 | A rule corrected the model's output |
| `modelDirect` | 0.85 | Below this, a model-only intent will **not** execute an action |
| `lowFallback` | 0.70 | Below this, hand the sentence to Claude to ask a clarifying question |

Flow (`functions/index.js:4220-4301`): `routeChatbotIntentByRules()` runs first; only if it returns
`null` does the ONNX model run, followed by `applyChatbotIntentOverride()`, which holds four
corrections for known confusion pairs (status query vs. mark-as-taken, medication vs. shopping,
help vs. greeting). A model-only intent between 0.70 and 0.85 returns *"No estoy seguro de si
quieres X. ¿Puedes decírmelo de otra forma?"* instead of writing to Firestore
(`functions/index.js:4290-4300`). For an app that records medication doses, refusing to act on a
medium-confidence guess is the correct default.

One rule is defensive by design: `"¿ha tomado el paracetamol?"` is a **question**, so it routes to
`check_medication_status`, never to `mark_medication_taken` (`router.js:123-127`, applied at
`router.js:181-183`). A classifier that marks a dose as taken because the user *asked* whether it
was taken is a patient-safety bug.

The rules cover 9 of the 12 intents. `add_reminder`, `add_contact` and `list_reminders` have no
rule and always reach the model.

### Evaluation harness

```js
const {
  routeChatbotIntentByRules,
} = require("../../deployment/router");
```
`tfm/evaluation/evaluate_chatbot_router.js:5-7`

The harness imports **the exact module the deployed function loads** — not a re-implementation. If
the rules drift, the harness drifts with them. 26 hand-written cases (accent-less input, colloquial
phrasing, shopping-vs-medication ambiguity, questions vs. commands):

```
Casos totales: 26
Cubiertos por reglas: 24
Delegados al modelo: 2
Errores de reglas: 0
```

The two delegated cases are the `add_reminder` and `add_contact` sentences, which no rule claims.

**Limitations:** those 26 cases were written by the same person who wrote the rules they test, so
this is a regression guard, not an unbiased evaluation. It also covers the rules layer only — it
never loads the ONNX model.

### Status

The Cloud Function, the router, the ONNX artefacts, `lib/services/chatbot_service.dart` (81 lines)
and a Flutter chat screen (`lib/screens/home/dashboard/chatbot_screen.dart`, 448 lines) all live on
the `tfm` branch, and were manually exercised end to end against a live Firebase project (the test
log is in `tfm/TFM_PROGRESS.md:273-281`). **They are not merged into `master`, and the chatbot is
not present in the published build of the app.** Section 7 is the main reason.

---

## 7. Known issues

### 7.1 Train/serve tokenisation skew (real bug, unfixed)

Training tokenises with:

```python
def tokenize(text):
    return text.lower().split()
```
`train_bilstm.py:77-78` — lowercase and split on whitespace. **Punctuation and accents are kept**,
so the learned vocabulary contains entries like `¿cuántas`, `dra.`, `mañana`, `medicación`.

Production tokenises with:

```js
function normalizeChatbotTextForModel(text) {
  return compactSpaces(String(text || "")
      .toLowerCase()
      .replace(/[`´'"]/g, " ")
      .replace(/[¿?¡!,.;()[\]{}]/g, " "));
}
```
`deployment/router.js:71-76` — **strips punctuation**, keeps accents.

Running that function over every entry of the shipped `bilstm_vocab.json` (1,382 entries excluding
`<PAD>`/`<UNK>`):

| Effect | Count |
|---|---:|
| Vocabulary entries the JS normaliser would alter | 587 / 1,382 (42.5%) |
| Entries thereby made **unreachable** (altered form absent from vocab) | 45 |
| Entries containing accents | 237 |
| …of which have no accent-free twin in vocab | **221** |

Two failure modes coexist. First, 45 vocabulary entries (`¿cuántas`, `dra.`, `2.5mg`, …) can never
be produced by the production tokeniser — the model has weights it can no longer reach. Second, and
worse in practice: Spanish users routinely type without accents, and the same repo's *rules*
normaliser strips accents on that assumption (`router.js:43-45, 51-55`). The *model* normaliser
keeps them, and 221 accented entries have no accent-free equivalent — so `"medicacion"` typed
without the accent hits `<UNK>` while `"medicación"` resolves. The training data's own typo pass
(`build_dataset.py:217`) deliberately generates accent-less variants, which partially masks this,
but does not close it.

**There is no Python↔JS tokenisation parity test.** That is the missing artefact. The fix is a
shared normalisation spec plus a fixture file of, say, 200 strings whose token id sequences must be
identical in both runtimes, run in CI. Finding this while auditing my own deployment is the reason
the chatbot was not merged.

### 7.2 The one confusion the model makes is unguarded

The rules return `null` for `"¿Qué le toca a mamá hoy?"` and for its accent-less form, and no
override covers the `list_medications` / `list_reminders` pair. The exact ambiguity that produces
52 of the model's 53 errors reaches the model with no deterministic layer in front of it and no
correction behind it.

### 7.3 Unmeasured serving latency

Covered in section 5: no per-request ONNX benchmark, no cold-start measurement. The
BiLSTM-over-BERT decision is well supported on artefact size and F1-macro, and unsupported on
latency.

### 7.4 Entity extraction is outside the thesis

Every action requires entities, and every entity comes from a Claude Haiku call
(`functions/index.js:3565-3607`). That means a per-request external API dependency inside a 30 s
timeout, a cost per interaction, and a component with no accuracy measurement of any kind. A NER
head trained alongside the classifier — using the entity spans the generator already knows — is the
obvious next step and is not done.

### 7.5 Single language, single register

Spanish only. The dataset's register is peninsular and family-colloquial (`la yaya`, `Mercadona`,
`el estanco`). `bert-base-multilingual-cased` was chosen partly to make a later Catalan extension
cheap, but no multilingual evaluation was run.

---

## 8. Reproducing this

```bash
git checkout tfm
cd tfm

# 1. Dataset - deterministic, no API keys, ~1.5 s
python build_dataset.py          # -> dataset/{train,val,test}.jsonl

# 2. Classical models (CPU, seconds)
pip install -r requirements.txt
python train_naive_bayes.py
python train_random_forest.py
python train_svm.py

# 3. Neural models (GPU strongly recommended)
python train_bilstm.py           # 131 s on an RTX 4060
python train_bert.py             # 3,276 s on an RTX 4060

# 4. Comparison table and plots
python compare_models.py         # -> results/comparison_summary.json, results/plots/

# 5. Export the deployment artefact
python convert_to_onnx.py        # -> models/bilstm_model.onnx + bilstm_vocab.json

# 6. Rules-layer regression check (Node, no Firebase/ONNX/API key needed)
node evaluation/evaluate_chatbot_router.js
```

`requirements.txt` pins `torch>=2.0.0` without a CUDA index; install the wheel matching your own
CUDA version first if you want GPU training.

### Reproducing the leakage measurement

The audit in section 4 is the part worth re-running. This is the whole of it:

```python
import json
from collections import defaultdict

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def skeleton(ex):
    """Substitute entity values back into their placeholders."""
    t = ex["text"]
    for k, v in sorted(ex["entities"].items(), key=lambda kv: -len(str(kv[1]))):
        v = str(v)
        if v and v in t:
            t = t.replace(v, "{" + k + "}")
    return (ex["intent"], t)

train, test = load("dataset/train.jsonl"), load("dataset/test.jsonl")

seen = {skeleton(e) for e in train}
hits = sum(1 for e in test if skeleton(e) in seen)
print(f"template leakage: {hits}/{len(test)} = {hits/len(test)*100:.2f}%")
# -> template leakage: 11686/11845 = 98.66%

vocab = {w for e in train for w in e["text"].lower().split()}
toks = [t for e in test for t in e["text"].lower().split()]
oov = [t for t in toks if t not in vocab]
print(f"OOV tokens: {len(oov)}/{len(toks)} = {len(oov)/len(toks)*100:.4f}%")
# -> OOV tokens: 4/103574 = 0.0039%

labels = defaultdict(set)
for e in train + load("dataset/val.jsonl") + test:
    labels[e["text"].lower()].add(e["intent"])
print(f"contradictory labels: {sum(1 for v in labels.values() if len(v) > 1)}")
# -> contradictory labels: 87
```

---

## Repository layout

```
tfm/                                  (branch: tfm)
  build_dataset.py                    # 969 lines. Template generator, seed 42.
  generate_dataset.py                 # LLM-based alternative. Written, never executed.
  train_{naive_bayes,random_forest,svm,bilstm,bert}.py
  convert_to_onnx.py                  # PyTorch -> ONNX opset 14
  compare_models.py
  results/*.json                      # Per-model metrics, per-intent P/R/F1, confusion matrices
  evaluation/
    chatbot_intent_cases.jsonl        # 26 hand-written routing cases
    evaluate_chatbot_router.js        # Imports the deployed router module directly

functions/
  chatbot/router.js                   # 234 lines. Pure. Rules + normalisers + thresholds.
  chatbot/bilstm_model.onnx           # 9.7 MiB
  chatbot/bilstm_vocab.json           # 22 KB
  index.js:3440-4400                  # ONNX session, classifier, entity extraction, actions
                                      # (of 4,684 lines total)

lib/
  services/chatbot_service.dart       # 81 lines
  screens/home/dashboard/chatbot_screen.dart   # 448 lines
```

---

## Closing note

The number I would defend is not 99.57%. It is this: five very different model families landed
within 21 errors of each other, and rather than reading that uniformity as confirmation, I measured
the benchmark itself until I could account for where the number came from. It came from a flat
shuffle over templated data — 98.66% template leakage — and a template collision I wrote myself,
which accounts for 52 of the winning model's 53 remaining errors.

The same pass over the deployment found a 42.5% mismatch between the training and serving
tokenisers, which is why the chatbot is still on a branch.
