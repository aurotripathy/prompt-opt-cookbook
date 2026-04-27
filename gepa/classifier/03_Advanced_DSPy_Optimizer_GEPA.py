# This version is adapted from the original version at:
# https://medium.com/@AI-on-Databricks/prompt-optimizing-with-gepa-and-databricks-for-90x-cheaper-inference-0068a2909d86
# Yet, it is free from the databricks notebook environment and is more standalone. No Databricks secrets are needed.
# # DSPy GEPA prompt optimization (text classification)
#
# This walkthrough focuses on one topic: using the GEPA optimizer to improve a smaller
# Gemini model on a supervised task. You will:
#
# 1. Load the PubMed text classification dataset (Hugging Face) with a train/test split
# 2. Define a DSPy signature and TextClassifier module (Google Gemini via LiteLLM)
# 3. Run GEPA with a larger Gemini as the reflection / teacher LM
# 4. Compare accuracy before and after optimization (small Gemini vs large Gemini)
#
# Prerequisites: GOOGLE_API_KEY in the environment (see repo .env), plus MLflow if you want traces.


# #DSPy Prompt Optimizers 
#
# Iterating through prompts manually is tedious. Without an automated and grounded/objective way of iterating development of prompts, 
# it becomes nearly impossible to maintain prompts over time, especially post production. 
#
# **Automated improvement** - Instead of manually tweaking prompts through trial-and-error, 
# DSPy systematically optimizes them based on your metrics and training examples. 
# This can save significant time and often discovers better prompts than manual engineering.
#
# **Data-driven optimization** - The optimizers learn from your specific examples and use cases, 
# tailoring prompts to your actual needs rather than generic best practices.
# Complex pipeline optimization - When you have multi-step LLM workflows (retrieval → reasoning → generation), 
# DSPy can optimize the entire pipeline together, which is much harder to do manually.
#
# **Reproducible and systematic** - Unlike ad-hoc prompt engineering, DSPy provides a programmatic, 
# repeatable process for improving your LLM applications.
#
# **Handling prompt brittleness** - Optimizers can find more robust prompts that work across different examples, 
# reducing the brittleness common with hand-crafted prompts. 
#
# DSPy optimizers are particularly useful when you:
#
# 1. Have clear metrics and evaluation data
# 2. Need to optimize complex, multi-step LLM pipelines
# 3. Want to adapt prompts for different models (DSPy can re-optimize when you switch models)
# 4. Have spent significant time manually tweaking prompts without great results
# 5. Need consistent performance across diverse inputs
#
# ##Cost Value of Prompt Optimizers 
#
# The Databricks Mosaic AI Research Team released a blog post highlighting how they achieve 90x cost savings by using GEPA, 
# a prompt optimizer on their AI workflows. It highlights how we can find significant performance gains just from optimizing 
# prompts on smaller LLMs. 
# Check out the blog here: 
# https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization
#
# If costs are stopping you from going to production, it is essentially mandatory to do prompt optimization so that you are enabled to use smaller LLMs. 
#
# For example, you can compare cost/latency of a flash-tier Gemini against a pro-tier Gemini.


# #Prompt Optimization Demo: GEPA 
#
# We use the same GEPA idea as in the Mosaic AI research blog post: optimize a smaller student model using a stronger
# reflection LM. Here the student is Gemini Flash-Lite and the teacher/reflection LM is Gemini Pro (both via the
# Google AI API). We compare small vs large accuracy before optimization, then measure the student again after GEPA.


# %pip install --upgrade dspy mlflow python-dotenv certifi
# Restart the Python kernel after installing packages. (`certifi` fixes HTTPS CSV download on some macOS Python builds.)


# ## Set up data
# The following downloads the
#  [pubmed text classification cased](https://huggingface.co/datasets/ml4pubmed/pubmed-text-classification-cased/resolve/main/{}.csv) 
# dataset from Huggingface and writes a utility to ensure that your train and test split has the same labels.
# The labels are: 'CONCLUSIONS', 'RESULTS', 'METHODS', 'OBJECTIVE', 'BACKGROUND'


import dataclasses
import io
import json
import os
import ssl
import urllib.request

import certifi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from dspy.datasets.dataset import Dataset
from pandas import StringDtype
from tqdm.auto import tqdm

from utils import (
    LARGE_MODEL_CANDIDATES,
    SMALL_MODEL_CANDIDATES,
    resolve_gemini_model,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Gemini model ids (LiteLLM). Student = smaller; large = stronger baseline; reflection = GEPA teacher LM.
# Preview models (e.g. "gemini-3.1-pro-preview") may 404 on v1alpha if your API key lacks preview
# access. The helper below probes each candidate and returns the first that actually responds.
small_model = resolve_gemini_model(SMALL_MODEL_CANDIDATES, role="small_model")
large_model = resolve_gemini_model(LARGE_MODEL_CANDIDATES, role="large_model")
reflection_model = large_model


def _to_json_serializable(obj):
    """Recursively convert LM history to JSON-safe data (e.g. LiteLLM ModelResponse objects)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, set):
        return [_to_json_serializable(x) for x in sorted(obj, key=str)]
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_json_serializable(model_dump())
        except Exception:
            pass
    dict_fn = getattr(obj, "dict", None)
    if callable(dict_fn):
        try:
            return _to_json_serializable(dict_fn())
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return _to_json_serializable(dataclasses.asdict(obj))
        except Exception:
            pass
    return str(obj)


def _read_csv_from_url(url: str) -> pd.DataFrame:
    """Load a CSV over HTTPS using certifi's CA bundle (avoids SSL verify failures on some systems)."""
    print(f"Reading CSV from URL: {url}")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
        return pd.read_csv(io.BytesIO(resp.read()))
    print(f"CSV read successfully from URL: {url}")


def _read_csv_from_disk(filename: str, dataset_dir: str = "./dspy_hackathon/dataset") -> pd.DataFrame:
    """Load a CSV from the local dataset directory (default: ./dataset).

    Run `dspy_hackathon/download_pubmed_dataset.py` once to populate the folder
    with `train.csv` and `test.csv` before calling this function.
    """
    path = os.path.join(dataset_dir, filename)
    print(f"Reading CSV from disk: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found at {path}. Run `python dspy_hackathon/download_pubmed_dataset.py` "
            f"to download the PubMed train/test splits into {dataset_dir}/ first."
        )
    return pd.read_csv(path)


def read_data_and_subset_to_categories() -> tuple[pd.DataFrame]:
    """
    Read the pubmed-text-classification-cased dataset. Docs can be found in the url below:
    https://huggingface.co/datasets/ml4pubmed/pubmed-text-classification-cased/resolve/main/{}.csv
    """

    # Read train/test split from local disk (run download_pubmed_dataset.py first).
    # Remote URL fallback retained for reference:
    # file_path = "https://huggingface.co/datasets/ml4pubmed/pubmed-text-classification-cased/resolve/main/{}.csv"
    # train = _read_csv_from_url(file_path.format("train"))
    # test = _read_csv_from_url(file_path.format("test"))

    # Previously downloaded to ./dspy_hackathon/dataset/train.csv and ./dspy_hackathon/dataset/
    train = _read_csv_from_disk("train.csv")
    test = _read_csv_from_disk("test.csv")

    train.drop('description_cln', axis=1, inplace=True)
    test.drop('description_cln', axis=1, inplace=True)

    return train, test


class CSVDataset(Dataset):
    """
    A dataset class for the pubmed text classification cased dataset.
    
    Args:
        n_train_per_label: The number of training samples per label.
        n_test_per_label: The number of test samples per label.
    """
    def __init__(
        self, n_train_per_label: int = 40, n_test_per_label: int = 20, *args, **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        self.n_train_per_label = n_train_per_label
        self.n_test_per_label = n_test_per_label

        self._create_train_test_split_and_ensure_labels()

    def _create_train_test_split_and_ensure_labels(self) -> None:
        """Perform a train/test split that ensure labels in `test` are also in `train`."""
        # Read the data
        train_df, test_df = read_data_and_subset_to_categories()

        train_df = train_df.astype(StringDtype())
        test_df = test_df.astype(StringDtype())

        # Sample for each label
        train_samples_df = pd.concat([
            group.sample(n=self.n_train_per_label, random_state=1) 
            for _, group in train_df.groupby('target')
        ])
        test_samples_df = pd.concat([
            group.sample(n=self.n_test_per_label, random_state=1) 
            for _, group in test_df.groupby('target')
        ])

        # Set DSPy class variables
        self._train = train_samples_df.to_dict(orient="records")
        self._test = test_samples_df.to_dict(orient="records")


# Sample a train/test split from the pubmed-text-classification-cased dataset
dataset = CSVDataset(n_train_per_label=3, n_test_per_label=10)

# Create train and test sets containing DSPy examples
train_dataset = [example.with_inputs("description") for example in dataset.train]
test_dataset = [example.with_inputs("description") for example in dataset.test]

print(f"train dataset size: \n {len(train_dataset)}")
print(f"test dataset size: \n {len(test_dataset)}")
print(f"Train labels: \n {set([example.target for example in dataset.train])}")

# print the first 5 samples
print("********* train dataset sample entries (first 5): *********")
for ex in train_dataset[:5]:
    print(ex)
print("********* test dataset sample entries (first 5): *********")
for ex in test_dataset[:5]:
    print(ex)
print("*************************************************************")

# #Set up the DSPy module and signature for testing 

from typing import Literal
import warnings

import dspy

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Pydantic V1 functionality isn't compatible with Python 3\.\d+.*",
)
import mlflow


def gemini_llm(model: str, cache: bool = False, **kwargs) -> dspy.LM:
    return dspy.LM(model, api_key=os.environ["GOOGLE_API_KEY"], cache=cache, **kwargs)


# turning on autologging traces
mlflow.dspy.autolog(
    log_evals=True,
    log_compiles=True,
    log_traces_from_compile=True
)

# Create a signature for the DSPy module
class TextClassificationSignature(dspy.Signature):
    description: str = dspy.InputField()
    target: Literal[
        'CONCLUSIONS', 'RESULTS', 'METHODS', 'OBJECTIVE', 'BACKGROUND'
        ] = dspy.OutputField()


class TextClassifier(dspy.Module):
    """
    Classifies medical texts into a previously defined set of categories.
    """
    def __init__(self, model: str):
        super().__init__()
        self.lm = gemini_llm(model, cache=False, max_tokens=25000)
        self.generate_classification = dspy.Predict(TextClassificationSignature)

    def forward(self, description: str):
        """Returns the predcited category of the description text provided"""
        with dspy.context(lm=self.lm):
            return self.generate_classification(description=description)


# #Let's test that it works

# Initialize classifier with the small and large Gemini models
small_text_classifier = TextClassifier(model=small_model)
large_text_classifier = TextClassifier(model=large_model)
description = "This study is designed as a randomised controlled trial in which men living with HIV in Australia will be assigned to either an intervention group or usual care control group ."
print(f"Using {small_model} as the model to classify the text:\n{description}")
print(f"{small_text_classifier(description=description)}")
print(f"Using {large_model} as the model to classify the text:\n{description}")
print(f"{large_text_classifier(description=description)}")

# #Make an Evaluation Function
#
# GEPA needs a metric that returns a score plus optional text feedback. Here we use gold labels from the dataset:
# exact match on the structured label, with short feedback for GEPA (no Databricks AI Judges required).

def validate_classification_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Score 1 if predicted label matches gold, else 0; feedback string helps GEPA refine prompts.
    """
    if example.target == prediction.target:
        return dspy.Prediction(score=1, feedback="Correct: prediction matches the gold label.")
    return dspy.Prediction(
        score=0,
        feedback=(
            f"Incorrect: expected '{example.target}' but model predicted '{prediction.target}'."
        ),
    )

def check_accuracy_on_test_dataset(classifier, test_data: pd.DataFrame = test_dataset, desc: str = "Evaluating") -> float:
    """
    Checks the accuracy of the classifier on the test data.
    """
    scores = []
    progress = tqdm(test_data, desc=desc, unit="ex")
    for example in progress:
        prediction = classifier(description=example["description"])
        score = validate_classification_with_feedback(example, prediction).score
        scores.append(score)
        progress.set_postfix(acc=f"{np.mean(scores):.3f}")

    return np.mean(scores)


# #Baseline: small vs large Gemini (before GEPA optimization is applied)

# the original prompt for the small model
baseline_classifier = TextClassifier(model=small_model)
_ = baseline_classifier(description=description)  # or any example string
original_prompt = baseline_classifier.lm.history[-1]["messages"][0]["content"]
print(f"Original prompt for {small_model}: \n{original_prompt}")

uncompiled_small_lm_accuracy = check_accuracy_on_test_dataset(
    TextClassifier(model=small_model), desc=f"Uncompiled {small_model}"
)
print(f"Uncompiled {small_model} accuracy on test dataset: {uncompiled_small_lm_accuracy}")

uncompiled_large_lm_accuracy = check_accuracy_on_test_dataset(
    TextClassifier(model=large_model), desc=f"Uncompiled {large_model}"
)
print(f"Uncompiled {large_model} accuracy on test dataset: {uncompiled_large_lm_accuracy}")


# ### The smaller model is usually weaker than the larger one; GEPA tries to close that gap on the student model.
# #Time to run GEPA
#
# With baselines in place, we optimize the student (small Gemini) using the large Gemini as the reflection LM.
#
# If you need to read more about GEPA, check out the resources here: 
# 1. GEPA Paper: https://arxiv.org/pdf/2507.19457 
# 2. DSPy GEPA Tutorials: https://dspy.ai/api/optimizers/GEPA/overview/ 

print("Starting GEPA optimization...")
import uuid

# defining an UUID to identify the optimized module in the MLflow run
id = str(uuid.uuid4())
print(f"id: {id}")

gepa = dspy.GEPA(
    metric=validate_classification_with_feedback,
    auto="light",
    reflection_minibatch_size=15,
    reflection_lm=gemini_llm(reflection_model, max_tokens=100000),
    num_threads=16,
    seed=1
)

with mlflow.start_run(run_name=f"gepa_{id}"):
    compiled_gepa = gepa.compile(
        TextClassifier(model=small_model),
        trainset=train_dataset, #reminder: Only passing 15 training sets! 
    )

compiled_gepa.save(f"compiled_gepa_{id}.json")
print(f"GEPA optimization completed and saved to compiled_gepa_{id}.json")

# #Let's try it again
#
# The optimized program is saved as JSON. Reload it on the same small Gemini student and re-evaluate.

print("Loading the optimized model and reevaluating accuracy on test dataset...")
text_classifier_gepa = TextClassifier(model=small_model)
text_classifier_gepa.load(f"compiled_gepa_{id}.json")

compiled_small_lm_accuracy = check_accuracy_on_test_dataset(
    text_classifier_gepa, desc=f"GEPA-compiled {small_model}"
)
print(f"Compiled {small_model} accuracy: {compiled_small_lm_accuracy}")


# #Look at the score
#
# You may see the compiled small model move closer to (or past) the uncompiled large model on this tiny train slice—
# results depend on labels, sample size, and API variance. Compare numbers above and inspect the evolved prompt below.


print("You can inspect the prompt below! ")
print(text_classifier_gepa.lm.history[-1]["messages"][0]["content"])
abs_gain = compiled_small_lm_accuracy - uncompiled_small_lm_accuracy
rel_pct = (
    (abs_gain / uncompiled_small_lm_accuracy) * 100.0 if uncompiled_small_lm_accuracy else float("nan")
)
print(
    f"GEPA vs uncompiled student ({small_model}): "
    f"accuracy {uncompiled_small_lm_accuracy:.4f} → {compiled_small_lm_accuracy:.4f} "
    f"(Δ = {abs_gain:+.4f}; {rel_pct:+.2f}% relative vs. uncompiled baseline)"
)

# dump all the history to a json file (history may contain LiteLLM ModelResponse; not JSON-serializable by default)
with open(f"gepa_{id}_history.json", "w") as f:
    json.dump(_to_json_serializable(text_classifier_gepa.lm.history), f, indent=2)
    f.write("\n")


print(f"History dumped to gepa_{id}_history.json")