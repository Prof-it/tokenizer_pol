"""KLEJ benchmark task definitions."""

from dataclasses import dataclass
from typing import List, Optional, Literal


@dataclass
class TaskConfig:
    """Configuration for a KLEJ task."""
    name: str
    dir_name: str  # Directory name in data/klej/KLEJ/
    text_columns: List[str]  # Columns to concatenate as input
    target_column: str
    task_type: Literal["classification", "regression"]
    num_labels: Optional[int]  # classification labels
    metric: str  # evaluation metric
    has_dev: bool = True  # Some tasks don't have dev set


TASKS = {
    # named entity recognition
    "nkjp-ner": TaskConfig(
        name="nkjp-ner",
        dir_name="NKJP-NER",
        text_columns=["sentence"],
        target_column="target",
        task_type="classification",
        num_labels=6,  # persName, placeName, orgName, geogName, date, noEntity
        metric="accuracy",
    ),
    #
    "cdsc-e": TaskConfig(
        name="cdsc-e",
        dir_name="CDSC-E",
        text_columns=["sentence_A", "sentence_B"],
        target_column="entailment_judgment",
        task_type="classification",
        num_labels=3,  # ENTAILMENT, NEUTRAL, CONTRADICTION
        metric="accuracy",
    ),
    # sentence similarity
    "cdsc-r": TaskConfig(
        name="cdsc-r",
        dir_name="CDSC-R",
        text_columns=["sentence_A", "sentence_B"],
        target_column="relatedness_score",
        task_type="regression",
        num_labels=1,
        metric="spearman",
    ),
    # classification
    "cbd": TaskConfig(
        name="cbd",
        dir_name="CBD",
        text_columns=["sentence"],
        target_column="target",
        task_type="classification",
        num_labels=2,  # 0, 1
        metric="f1",
        has_dev=False,
    ),
    "polemo-in": TaskConfig(
        name="polemo-in",
        dir_name="POLEMO2.0-IN",
        text_columns=["sentence"],
        target_column="target",
        task_type="classification",
        num_labels=4,  # meta_plus_m, meta_minus_m, meta_zero, meta_amb
        metric="accuracy",
    ),
    "polemo-out": TaskConfig(
        name="polemo-out",
        dir_name="POLEMO2.0-OUT",
        text_columns=["sentence"],
        target_column="target",
        task_type="classification",
        num_labels=4,
        metric="accuracy",
    ),
    "dyk": TaskConfig(
        name="dyk",
        dir_name="DYK",
        text_columns=["question", "answer"],
        target_column="target",
        task_type="classification",
        num_labels=2,  # 0, 1
        metric="f1",
        has_dev=False,
    ),
    "psc": TaskConfig(
        name="psc",
        dir_name="PSC",
        text_columns=["extract_text", "summary_text"],
        target_column="label",
        task_type="classification",
        num_labels=2,  # 0, 1
        metric="f1",
        has_dev=False,
    ),
    "ar": TaskConfig(
        name="ar",
        dir_name="ECR",  # Note: directory is ECR, not AR
        text_columns=["text"],
        target_column="rating",
        task_type="regression",
        num_labels=1,
        metric="wmae",
    ),
}