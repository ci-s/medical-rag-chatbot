# %%
import sys
import os
import pandas as pd

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import get_document
from settings import get_page_types, config, settings, VIGNETTE_COLLECTION

# %% [markdown]
# # Create evaluation sheet

# %%
file_path = os.path.join(settings.data_path, settings.file_name)
pages, _, _, _ = get_page_types()
document = get_document(file_path, pages)

# %%
import json

best_file_path = "mlartifacts/822308634021697303/0ea73b5800624805b24c7442fe36b249/artifacts/generation_eval_1740485574_new_chunking_None.json"  # loud-gull-340
not_best_file_path = "mlartifacts/822308634021697303/4ecac81d28464c428dc8a2469df0415d/artifacts/generation_eval_1740487137_new_chunking_None.json"  # debonair-jay-351
not_best_file_path = "mlartifacts/822308634021697303/0302b00427ff40b2ab728766fbff223e/artifacts/generation_eval_1740489565_new_chunking_None.json"  # unruly-ape-22

with open(best_file_path, "r") as file:
    best_experiment_data = json.load(file)

with open(not_best_file_path, "r") as file:
    not_best_experiment_data = json.load(file)

# %%
text_only_qids = [0, 9, 12, 14, 18, 22, 23, 24, 26, 30, 32, 34, 35, 42, 43, 44, 48, 51, 52, 65, 72, 80, 81]

# %%
rows = []
for vignette in VIGNETTE_COLLECTION.vignettes:
    for question in vignette.questions:
        if question.id in text_only_qids:
            rows.append(
                {
                    "Vignette ID": vignette.get_id(),
                    "Question ID": question.get_id(),
                    "Background": vignette.get_background(),
                    "Question": question.get_question(),
                    "Ground Answer": question.get_answer(),
                    "Reference Pages": question.get_reference(),  # ", ".join(map(str, question.get_reference())) if question.get_reference() else "N/A"
                }
            )

df = pd.DataFrame(rows)
df

# %%
rows = []
for entry in best_experiment_data["all_feedbacks"]:
    rows.append(
        {
            "Question ID": entry["question_id"],
            "Generated Answer": entry["generated_answer"]["answer"],
            "Feedback": entry["feedback"],
            "Score": entry["score"],
        }
    )

# Create DataFrame
df_generated_best = pd.DataFrame(rows)

rows = []
for entry in not_best_experiment_data["all_feedbacks"]:
    rows.append(
        {
            "Question ID": entry["question_id"],
            "Generated Answer": entry["generated_answer"]["answer"],
            "Feedback": entry["feedback"],
            "Score": entry["score"],
        }
    )

# Create DataFrame
df_generated_not_best = pd.DataFrame(rows)

df_generated = df_generated_best.merge(df_generated_not_best, on="Question ID", suffixes=("_best", "_not_best"))
df_generated

# %%
import random

# Extract generated answers for both experiments
rows_best = []
for entry in best_experiment_data["all_feedbacks"]:
    rows_best.append(
        {
            "Question ID": entry["question_id"],
            "Generated Answer": entry["generated_answer"]["answer"],
            "Source": "best",
            "Feedback": entry["feedback"],
            "Score": entry["score"],
        }
    )

df_generated_best = pd.DataFrame(rows_best)

rows_not_best = []
for entry in not_best_experiment_data["all_feedbacks"]:
    rows_not_best.append(
        {
            "Question ID": entry["question_id"],
            "Generated Answer": entry["generated_answer"]["answer"],
            "Source": "not_best",
            "Feedback": entry["feedback"],
            "Score": entry["score"],
        }
    )

df_generated_not_best = pd.DataFrame(rows_not_best)

# Merge by Question ID
df_generated = df_generated_best.merge(df_generated_not_best, on="Question ID", suffixes=("_best", "_not_best"))


# Randomly assign answers to "Generated Answer 1" and "Generated Answer 2", while keeping track of sources
def shuffle_answers(row):
    answers = [(row["Generated Answer_best"], "best"), (row["Generated Answer_not_best"], "not_best")]
    random.shuffle(answers)
    return pd.Series(
        [answers[0][0], answers[1][0], answers[0][1], answers[1][1]],
        index=["Generated Answer 1", "Generated Answer 2", "Generated Answer 1 Source", "Generated Answer 2 Source"],
    )


df_generated[["Generated Answer 1", "Generated Answer 2", "Generated Answer 1 Source", "Generated Answer 2 Source"]] = (
    df_generated.apply(shuffle_answers, axis=1)
)

df_final = df.merge(df_generated, on="Question ID")
df_final["Reference Pages Text"] = df_final.apply(
    lambda row: "/pagebreak".join(
        [document.get_processed_content(page_number) for page_number in row["Reference Pages"]]
    ),
    axis=1,
)
# Clinician sheet: Drop source columns before presenting
df_clinicians = df_final.drop(columns=["Generated Answer 1 Source", "Generated Answer 2 Source"])

# Internal sheet: Save the full mapping for later reference
df_internal = df_final.copy()

# %%
df_clinicians.head()

# %%
df_clinicians.to_csv(os.path.join(settings.results_path, "llm_as_a_judge_clinicians.csv"), index=False)

# %%
df_internal.to_csv(os.path.join(settings.results_path, "llm_as_a_judge_clinicians_internal.csv"), index=False)

# %% [markdown]
# # Clinician feedback

# %%
df_internal = pd.read_csv(os.path.join(settings.results_path, "llm_as_a_judge_clinicians_internal.csv"))

df_clinician_feedback = pd.read_excel(
    os.path.join(settings.results_path, "Evaluation by Clinicians_Georg.xlsx"), header=1
)
df_clinician_feedback_2 = pd.read_csv(os.path.join(settings.results_path, "Evaluation by Clinicians_SK.csv"), header=1)

# %%
df_clinician_feedback_2.head()

# %%
df_internal.head()

# %%
df_clinician_feedback.columns

# %%
df_clinician_feedback_2.columns

# %%
df_clinician_feedback_2.head()

# %%
# Merge with internal mapping using 'Question ID'
df_merged = df_internal.merge(
    df_clinician_feedback[["Question ID", "Generated Answer 1", "Generated Answer 2", "Score 1", "Score 2"]],
    on="Question ID",
    suffixes=("_internal", "_clinician1"),
)
df_merged = df_merged.merge(
    df_clinician_feedback_2[["Question ID", "Generated Answer 1", "Generated Answer 2", "Score 1", "Score 2"]],
    on="Question ID",
    suffixes=("", "_clinician2"),
)

# %%
df_merged.head()

# %%
len(df_merged)

# %%
df_merged.columns


# %%
def match_llm_scores(row):
    llm_score_1 = row["Score_best"] if row["Generated Answer 1 Source"] == "best" else row["Score_not_best"]
    llm_score_2 = row["Score_best"] if row["Generated Answer 2 Source"] == "best" else row["Score_not_best"]
    return pd.Series([llm_score_1, llm_score_2], index=["llm_score_1", "llm_score_2"])


# Apply function to create new columns
df_merged[["llm_score_1", "llm_score_2"]] = df_merged.apply(match_llm_scores, axis=1)

# %%
df_view = df_merged[
    [
        "Question ID",
        "Background",
        "Question",
        "Ground Answer",
        "Generated Answer 1 Source",
        "Generated Answer 2 Source",
        "Generated Answer 1_internal",
        "Generated Answer 2_internal",
        "llm_score_1",
        "llm_score_2",
        "Score 1",
        "Score 2",
        "Score 1_clinician2",
        "Score 2_clinician2",
    ]
]

# %% [markdown]
# # Overall

# %%
import numpy as np
from scipy.stats import pearsonr, wilcoxon

# Assuming df_merged is the final dataset
df = df_merged.copy()

# %%
# Clnician SK
llm_scores = np.concatenate([df["llm_score_1"], df["llm_score_2"]])
clinician_scores = np.concatenate([df["Score 1"], df["Score 2"]])  # Georg
clinician_scores_2 = np.concatenate([df["Score 1_clinician2"], df["Score 2_clinician2"]])  # SK


def calculate_overall_scores(llm_scores, clinician_scores):
    # 1️⃣ Overall Correlation between LLM and Clinician Scores
    correlation, _ = pearsonr(llm_scores, clinician_scores)

    # 2️⃣ Overall Mean Absolute Error (MAE)
    mae = np.mean(np.abs(llm_scores - clinician_scores))

    # 3️⃣ Overall Exact Score Match Rate
    match_rate = np.mean(llm_scores == clinician_scores)

    # 4️⃣ Overall Directional Consistency (Ranking Agreement)
    def ranking_consistency(row):
        llm_diff = row["llm_score_1"] - row["llm_score_2"]
        clinician_diff = row["Score 1"] - row["Score 2"]
        return np.sign(llm_diff) == np.sign(clinician_diff)

    ranking_agreement = np.mean(df.apply(ranking_consistency, axis=1))

    # 5️⃣ Overall Wilcoxon Signed-Rank Test (Checking if LLM scores differ significantly from clinician scores)
    wilcoxon_p_value = wilcoxon(llm_scores, clinician_scores).pvalue

    # Compile results
    evaluation_results_overall = pd.DataFrame(
        {
            "Metric": [
                "Pearson Correlation (Overall)",
                "Mean Absolute Error (Overall)",
                "Exact Match Rate (Overall)",
                "Ranking Agreement",
                "Wilcoxon p-value (Overall)",
            ],
            "Value": [correlation, mae, match_rate, ranking_agreement, wilcoxon_p_value],
        }
    )
    return evaluation_results_overall


print(calculate_overall_scores(llm_scores, clinician_scores))
print(calculate_overall_scores(llm_scores, clinician_scores_2))

# %%
best_scores = (
    df.loc[df["Generated Answer 1 Source"] == "best", "llm_score_1"].tolist()
    + df.loc[df["Generated Answer 2 Source"] == "best", "llm_score_2"].tolist()
)
clinician_best_scores = (
    df.loc[df["Generated Answer 1 Source"] == "best", "Score 1"].tolist()
    + df.loc[df["Generated Answer 2 Source"] == "best", "Score 2"].tolist()
)
clinician2_best_scores = (
    df.loc[df["Generated Answer 1 Source"] == "best", "Score 1_clinician2"].tolist()
    + df.loc[df["Generated Answer 2 Source"] == "best", "Score 2_clinician2"].tolist()
)

not_best_scores = (
    df.loc[df["Generated Answer 1 Source"] == "not_best", "llm_score_1"].tolist()
    + df.loc[df["Generated Answer 2 Source"] == "not_best", "llm_score_2"].tolist()
)
clinician_not_best_scores = (
    df.loc[df["Generated Answer 1 Source"] == "not_best", "Score 1"].tolist()
    + df.loc[df["Generated Answer 2 Source"] == "not_best", "Score 2"].tolist()
)
clinician2_not_best_scores = (
    df.loc[df["Generated Answer 1 Source"] == "not_best", "Score 1_clinician2"].tolist()
    + df.loc[df["Generated Answer 2 Source"] == "not_best", "Score 2_clinician2"].tolist()
)


def calculate_experiment_based_scores(best_scores, not_best_scores, clinician_best_scores, clinician_not_best_scores):
    # 1️⃣ Correlation
    correlation_best, _ = pearsonr(best_scores, clinician_best_scores)
    correlation_not_best, _ = pearsonr(not_best_scores, clinician_not_best_scores)

    # 2️⃣ Mean Absolute Error (MAE)
    mae_best = np.mean(np.abs(np.array(best_scores) - np.array(clinician_best_scores)))
    mae_not_best = np.mean(np.abs(np.array(not_best_scores) - np.array(clinician_not_best_scores)))

    # 3️⃣ Exact Score Match Rate
    match_rate_best = np.mean(np.array(best_scores) == np.array(clinician_best_scores))
    match_rate_not_best = np.mean(np.array(not_best_scores) == np.array(clinician_not_best_scores))

    # 5️⃣ Wilcoxon Signed-Rank Test
    wilcoxon_p_value_best = wilcoxon(best_scores, clinician_best_scores).pvalue
    wilcoxon_p_value_not_best = wilcoxon(not_best_scores, clinician_not_best_scores).pvalue

    # Compile results
    evaluation_results = pd.DataFrame(
        {
            "Metric": [
                "Pearson Correlation (Best)",
                "Pearson Correlation (Not Best)",
                "Mean Absolute Error (Best)",
                "Mean Absolute Error (Not Best)",
                "Exact Match Rate (Best)",
                "Exact Match Rate (Not Best)",
                "Wilcoxon p-value (Best)",
                "Wilcoxon p-value (Not Best)",
            ],
            "Value": [
                correlation_best,
                correlation_not_best,
                mae_best,
                mae_not_best,
                match_rate_best,
                match_rate_not_best,
                wilcoxon_p_value_best,
                wilcoxon_p_value_not_best,
            ],
        }
    )
    return evaluation_results


print(calculate_experiment_based_scores(best_scores, not_best_scores, clinician_best_scores, clinician_not_best_scores))
print(
    calculate_experiment_based_scores(best_scores, not_best_scores, clinician2_best_scores, clinician2_not_best_scores)
)


# %%
def print_summary(best_scores, not_best_scores, clinician_best_scores, clinician_not_best_scores):
    print("**Sums**")
    print(f"Best experiment, LLM score sum: {sum(best_scores)}")
    print(f"Best experiment, Clinician score sum: {sum(clinician_best_scores)}")
    print(f"Not best experiment, LLM score sum: {sum(not_best_scores)}")
    print(f"Not best experiment, Clinician score sum: {sum(clinician_not_best_scores)}")
    print("**Averages**")

    print(f"Best experiment, LLM score avg: {np.mean(best_scores):.2f}")
    print(f"Best experiment, Clinician score avg: {np.mean(clinician_best_scores):.2f}")

    print(f"Not best experiment, LLM score avg: {np.mean(not_best_scores):.2f}")
    print(f"Not best experiment, Clinician score avg: {np.mean(clinician_not_best_scores):.2f}")


# print_summary(best_scores, not_best_scores, clinician_best_scores, clinician_not_best_scores)
# print("\n\n")
# print_summary(best_scores, not_best_scores, clinician2_best_scores, clinician2_not_best_scores)


def summary_in_df(
    best_scores,
    not_best_scores,
    clinician_best_scores,
    clinician_not_best_scores,
    clinician2_best_scores,
    clinician2_not_best_scores,
):
    summary_data = {
        "Metric": [
            "Best experiment, LLM score sum",
            "Best experiment, Clinician score sum",
            "Best experiment, Clinician 2 score sum",
            "Not best experiment, LLM score sum",
            "Not best experiment, Clinician score sum",
            "Not best experiment, Clinician 2 score sum",
            "Best experiment, LLM score avg",
            "Best experiment, Clinician score avg",
            "Best experiment, Clinician 2 score avg",
            "Not best experiment, LLM score avg",
            "Not best experiment, Clinician score avg",
            "Not best experiment, Clinician 2 score avg",
        ],
        "Value": [
            sum(best_scores),
            sum(clinician_best_scores),
            sum(clinician2_best_scores),
            sum(not_best_scores),
            sum(clinician_not_best_scores),
            sum(clinician2_not_best_scores),
            np.mean(best_scores),
            np.mean(clinician_best_scores),
            np.mean(clinician2_best_scores),
            np.mean(not_best_scores),
            np.mean(clinician_not_best_scores),
            np.mean(clinician2_not_best_scores),
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)


summary_in_df(
    best_scores,
    not_best_scores,
    clinician_best_scores,
    clinician_not_best_scores,
    clinician2_best_scores,
    clinician2_not_best_scores,
)


# %%
best_scores

# %%
clinician_best_scores

# %%
print([int(score) for score in clinician2_best_scores])

# %%
not_best_scores

# %%
clinician_not_best_scores

# %% [markdown]
# ## Examine in detail

# %%
df_view = df_merged[
    [
        "Question ID",
        "Background",
        "Question",
        "Ground Answer",
        "Generated Answer 1 Source",
        "Generated Answer 2 Source",
        "Generated Answer 1_internal",
        "Generated Answer 2_internal",
        "llm_score_1",
        "llm_score_2",
        "Score 1",
        "Score 2",
        "Unnamed: 11",
    ]
]

# %%
df_view["Best_experiment_answer"] = df_view.apply(
    lambda row: row["Generated Answer 1_internal"]
    if row["Generated Answer 1 Source"] == "best"
    else row["Generated Answer 2_internal"],
    axis=1,
)
df_view["Not_best_experiment_answer"] = df_view.apply(
    lambda row: row["Generated Answer 1_internal"]
    if row["Generated Answer 1 Source"] == "not_best"
    else row["Generated Answer 2_internal"],
    axis=1,
)

df_view["Best_experiment_score"] = df_view.apply(
    lambda row: row["llm_score_1"] if row["Generated Answer 1 Source"] == "best" else row["llm_score_2"], axis=1
)
df_view["Not_best_experiment_score"] = df_view.apply(
    lambda row: row["llm_score_1"] if row["Generated Answer 1 Source"] == "not_best" else row["llm_score_2"], axis=1
)

df_view["Best_experiment_clinician_score"] = df_view.apply(
    lambda row: row["Score 1"] if row["Generated Answer 1 Source"] == "best" else row["Score 2"], axis=1
)
df_view["Not_best_experiment_clinician_score"] = df_view.apply(
    lambda row: row["Score 1"] if row["Generated Answer 1 Source"] == "not_best" else row["Score 2"], axis=1
)


# %%
df_view.drop(
    columns=[
        "Generated Answer 1_internal",
        "Generated Answer 2_internal",
        "llm_score_1",
        "llm_score_2",
        "Score 1",
        "Score 2",
    ],
    inplace=True,
)

# %%
df_view

# %%
best_experiment_llm_score_sum = df_view["Best_experiment_score"].sum()
best_experiment_clinician_score_sum = df_view["Best_experiment_clinician_score"].sum()
not_best_experiment_llm_score_sum = df_view["Not_best_experiment_score"].sum()
not_best_experiment_clinician_score_sum = df_view["Not_best_experiment_clinician_score"].sum()

# Calculate averages
best_experiment_llm_score_avg = df_view["Best_experiment_score"].mean()
best_experiment_clinician_score_avg = df_view["Best_experiment_clinician_score"].mean()
not_best_experiment_llm_score_avg = df_view["Not_best_experiment_score"].mean()
not_best_experiment_clinician_score_avg = df_view["Not_best_experiment_clinician_score"].mean()

# Compile results
evaluation_sums_averages = pd.DataFrame(
    {
        "Metric": [
            "Best experiment, LLM score sum",
            "Best experiment, Clinician score sum",
            "Not best experiment, LLM score sum",
            "Not best experiment, Clinician score sum",
            "Best experiment, LLM score avg",
            "Best experiment, Clinician score avg",
            "Not best experiment, LLM score avg",
            "Not best experiment, Clinician score avg",
        ],
        "Value": [
            best_experiment_llm_score_sum,
            best_experiment_clinician_score_sum,
            not_best_experiment_llm_score_sum,
            not_best_experiment_clinician_score_sum,
            best_experiment_llm_score_avg,
            best_experiment_clinician_score_avg,
            not_best_experiment_llm_score_avg,
            not_best_experiment_clinician_score_avg,
        ],
    }
)
evaluation_sums_averages

# %%
df_final["Generated Answer 1 Source"].value_counts()

# %%
df_final["Generated Answer 2 Source"].value_counts()

# %%
df_final

# %%
