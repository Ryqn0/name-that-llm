# ADR 0003: A hand-written loop for fine-tuning encoders

- **Status:** accepted (2026-10-25, P5)
- **Context:**
  - P5 fine-tunes transformer encoders (DeBERTa-v3, ModernBERT) as 18-way classifiers on
    Kaggle's T4 GPUs, with the same prompt-grouped CV as every other model.
  - Hugging Face's `Trainer` would do the training in a few lines of configuration. But it
    hides each step, needs `accelerate`, and its API has moved between the transformers
    on Kaggle (5.16) and the one installed locally (5.20).
  - This is a learning project: seeing every step of a training loop is part of the goal.
- **Decision:**
  - `encoder.train_encoder` is a short hand-written PyTorch loop:
    - AdamW;
    - a linear warmup, then a linear decay of the learning rate;
    - float16 autocast plus a `GradScaler` on CUDA;
    - gradient clipping at norm 1.0;
    - a `FloatingPointError` as soon as a loss isn't finite.
  - **The number of epochs is fixed** (a tuned hyperparameter). There's no early stopping
    on the validation fold, since choosing the epoch there would leak that fold into its
    own OOF score.
  - **Each fold starts from the pretrained checkpoint.** `fit_predict_fold` reloads
    `base_model` every time.
  - **Test predictions are the mean of the 5 fold models** (`predict_proba_folds`). They're
    the same models that produced the OOF, so blend weights chosen on the OOF carry over.
    The final notebook saves them in float16. The offline submission attaches that
    notebook's output, and `find_fold_dirs` finds the models, refusing anything other
    than exactly 5.
  - A CI job, `torch`, installs CPU-only torch and runs the model tests on tiny random
    models. ADR 0002 anticipated this job "if the LM code grows", and a training loop is
    exactly that.
- **Consequences:**
  - Every step is visible and commented, and the only API surface is long-stable torch:
    `autocast`, `GradScaler`, `AdamW`, `LambdaLR`.
  - We own its correctness, which is what the tests and the `torch` CI job are for.
  - Given up, to add only if a model ever needs them:
    - gradient accumulation (larger effective batches than memory allows);
    - mid-fold checkpoints (resuming a fold after an interrupted session);
    - multi-GPU training.
