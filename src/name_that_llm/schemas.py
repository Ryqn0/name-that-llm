"""Data contracts: what valid train, test and submission tables look like.

A "contract" is a rule the data must satisfy before any code touches it. Checking
at the boundary (right after loading, right before writing a submission) turns a
silent problem 50 steps later into a loud, precise error now. Pandera lets us write
these rules as code, as a class per table, the same way you'd describe a database
table.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from name_that_llm.labels import LABELS


class TrainSchema(pa.DataFrameModel):
    """train.csv: one labelled passage per row."""

    id: Series[str] = pa.Field(unique=True)
    prompt_id: Series[str]
    # Empty text in TRAIN means a broken row: there's nothing to learn from it.
    text: Series[str] = pa.Field(str_length={"min_value": 1})
    model: Series[str] = pa.Field(isin=LABELS)

    class Config:
        strict = True  # exactly these 4 columns, no more and no fewer
        # Each prompt was answered at most once by each model. A duplicate pair
        # would mean corrupted data, and it would break the per-prompt structure we
        # plan to exploit later (spec §6.7, M2).
        unique = ["prompt_id", "model"]


class TestSchema(pa.DataFrameModel):
    """test.csv: unlabelled passages.

    Deliberately LENIENT about the text (empty is allowed). On the hidden ~4,000-row
    rerun, a crash costs a whole submission, while a weird row only costs one
    prediction (Review Focus 2). Only the id must be perfect, because the submission
    is matched on it.
    """

    id: Series[str] = pa.Field(unique=True)
    text: Series[str]

    class Config:
        strict = True


class SubmissionSchema(pa.DataFrameModel):
    """submission.parquet: exactly what Kaggle expects."""

    id: Series[str] = pa.Field(unique=True)
    model: Series[str] = pa.Field(isin=LABELS)  # non-null by default in pandera

    class Config:
        strict = True  # exactly `id` and `model`


def validate_submission(sub: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
    """Check a submission before writing it, and return it unchanged if it's valid.

    The schema checks each row. The set comparison checks the file as a whole: every
    test id must appear exactly once. Running this BEFORE writing turns a wasted
    Kaggle submission (5 per day) into a free local error message.
    """
    sub = SubmissionSchema.validate(sub)
    if len(sub) != len(test_ids) or set(sub["id"]) != set(test_ids):
        raise ValueError("submission ids do not match test ids")
    return sub
