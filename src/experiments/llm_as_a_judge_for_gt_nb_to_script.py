# %%
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Phase 2 Generation")

# %%
import sys
import os
import json
import time

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import get_document
from services.retrieval import FaissService, retrieve, reorder_flowchart_chunks
from core.chunking import chunk_document, load_saved_chunks

from settings.settings import settings
from settings import get_page_types, config
from eval.generation_metrics import llm_as_a_judge
from eval.generation import RAGASResult
from core.generation import create_question_prompt_w_docs
from core.model import generate_response
from settings import VIGNETTE_COLLECTION

file_path = os.path.join(settings.data_path, settings.file_name)

pages, _, _, _ = get_page_types()
print(f"Number of pages: {len(pages)}")
# toc_pages=[2,3]

document = get_document(file_path, pages)
# toc = get_document(file_path, toc_pages)
chunks_saved = True
if chunks_saved:
    _all_chunks = load_saved_chunks(config.saved_chunks_path)
    chunks = reorder_flowchart_chunks(_all_chunks)
else:
    chunks = chunk_document(method="size", document=document, pages=pages, chunk_size=512)

faiss_service = FaissService()
faiss_service.create_index(chunks)

# %%
from domain.vignette import Vignette, Question
from parsing import ParaphrasedGroundTruth, parse_with_retry

# %% [markdown]
# # Paraphrase ground truth answers and calculate llm_as_a_judge score for paraphrased sentences

# %%
# I tried providing example within the prompt but all paraphrased sentences in the responses were the same with example. So, I removed the example from the prompt.


def paraphrase_ground_truth_answer(vignette: Vignette, question: Question):
    system_prompt = """
        You are an AI language model assistant. Your task is to generate three different versions of the given ground truth answer, expanding it into slightly longer sentences while preserving its meaning. These variations will help in evaluating an LLM’s ability to assess similarity between responses.  

        The answers are related to the given background and question. When paraphrasing, make sure that the reworded responses remain consistent with the context provided in the background. You may use relevant details from the background to enhance the paraphrases, but do not introduce new information. Your goal is to rephrase the answer in the given format while adding minor elaborations or alternative phrasing, but without changing its core information. Do not deviate from the specified format.

        **Example Format:**
        Background: {{background}}
        Question: {{question}}
        Ground truth answer: {{answer}}

        Output:
        {{
            "paraphrased": [
                "<paraphrased version 1>",
                "<paraphrased version 2>",
                "<paraphrased version 3>"
            ]
        }}
        
        Do not say anything else. Make sure the response is a valid JSON.
    """

    user_prompt = """
        Background:{background}
        Question: {question}
        Ground truth answer: {answer}
    """

    response = generate_response(
        system_prompt,
        user_prompt.format(background=vignette.background, question=question.question, answer=question.answer),
    )
    response = parse_with_retry(ParaphrasedGroundTruth, response)
    return response.paraphrased


# %%
from domain.document import Document


# %%
def evaluate_single_gt(
    vignette_id: int,
    question_id: int,
    faiss_service: FaissService,
    document: Document,
) -> RAGASResult:
    vignette = VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id)
    questions = vignette.get_questions()
    print(f"Questions in vignette {vignette_id}: {len(questions)}")
    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    question = vignette.get_question(question_id)

    retrieved_documents = retrieve(vignette, question, faiss_service)

    paraphrased_answers = paraphrase_ground_truth_answer(vignette, question)

    return [
        RAGASResult(
            question_id,
            retrieved_documents,
            paraphrased_answer,
            llm_as_a_judge(vignette, question, paraphrased_answer, document),
            None,
            None,
            None,
        )
        for paraphrased_answer in paraphrased_answers
    ]


from statistics import mean


def compute_average_scores(all_feedbacks: list[list[RAGASResult]], score_keys: list[str]) -> dict:
    """
    Computes the average score per ground truth by first averaging the paraphrased scores,
    then averaging across all ground truths.

    Parameters:
        all_feedbacks (list[list[RAGASResult]]): List of lists, where each sublist contains the
                                                 paraphrased results of a single ground truth.
        score_keys (list[str]): List of score keys to compute averages for.

    Returns:
        dict: Dictionary with final average scores for each key.
    """
    avg_scores = {}

    # Compute per-ground-truth average
    gt_averages = {key: [] for key in score_keys}  # Store averages per ground truth

    for feedback_set in all_feedbacks:  # Each feedback_set corresponds to one ground truth
        if not feedback_set:
            continue

        for key in score_keys:
            scores = [
                float(getattr(ragas_result, key).score)
                for ragas_result in feedback_set
                if getattr(ragas_result, key) is not None
            ]
            if scores:  # Avoid empty lists
                gt_averages[key].append(mean(scores))  # Store per-ground-truth mean

    # Compute final average across all ground truths
    for key in score_keys:
        avg_scores[key] = mean(gt_averages[key]) if gt_averages[key] else 0.0  # Avoid empty dataset

    return avg_scores


def evaluate_all(
    faiss_service: FaissService,
    document: Document,
) -> tuple[int, list[RAGASResult]]:
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_source() != "Handbuch":
                continue
            all_feedbacks.append(evaluate_single_gt(vignette.get_id(), question.get_id(), faiss_service, document))

    score_keys = ["llm_as_judge"]
    avg_scores = compute_average_scores(all_feedbacks, score_keys)
    # Add validation to score for integer between 1 and 5
    return avg_scores, all_feedbacks


# %%
config.top_k

# %%
# evaluate using llm as a judge
# comment out ragas

avg_scores, all_feedbacks = evaluate_all(faiss_service, document)

# %%
import time

result_dicts = {
    "config": config.model_dump(),
    "all_feedbacks": [fb.to_dict() for fb_set in all_feedbacks for fb in fb_set],
    "avg_score": avg_scores,
}
output_file = f"eval_generation_eval_{int(time.time())}.json"
output_path = os.path.join(settings.results_path, output_file)
with open(output_path, "w") as file:
    json.dump(result_dicts, file, indent=4, ensure_ascii=False)

# eval_generation_eval_1738680803.json

# %%
avg_scores

# %%
with mlflow.start_run():
    mlflow.log_params(config.model_dump())
    mlflow.log_params(settings.model_dump(mode="json"))
    mlflow.log_params({"num_questions": len(all_feedbacks)})
    mlflow.log_metric("avg_score", avg_scores["llm_as_judge"])
    mlflow.set_tag("name", "GT All")

    mlflow.log_artifact(output_path)

# %%
print(f"Results saved to {output_path}")

# %% [markdown]
#
# ## question
# 'Kann eine rekanalisierende Therapie erfolgen?'
# ## original answer
# 'Prinzipiell kann bei klarem Zeitfenster nach Ausschluss von Kontraindikation auch bei einem bereits weitgehend demarkierten Infarkt lysiert werden, da dies keine Kontraindikation darstellt. Zunächst muss eine Blutdrucksenkung erzielt werden.'
#
# ### paraphrased
# 'In principle, a recanalizing therapy can be performed even if the infarct is largely demarcated, as long as there are no contraindications and a clear time window is present. However, initially, a reduction in blood pressure must be achieved.',
#
# ### feedback
# "The response is mostly correct, accurate, and factual. It correctly states that a recanalizing therapy can be performed even if the infarct is largely demarcated, as long as there are no contraindications and a clear time window is present. However, it does not explicitly mention the need to exclude contraindications, which is a crucial aspect of the reference answer. Additionally, the phrase 'a reduction in blood pressure must be achieved' is not as precise as the reference answer's 'eine Blutdrucksenkung erzielt werden'.",
#
#
# ### paraphrased
# 'Given a clear time frame and the absence of contraindications, a recanalizing therapy is feasible even for a mostly demarcated infarct, which does not itself constitute a contraindication. Lowering the blood pressure is the first step.',
#
# ### feedback
# 'The response is mostly correct, accurate, and factual. It correctly states that a recanalizing therapy is feasible in the given scenario. However, it does not explicitly mention the importance of excluding contraindications, which is a crucial aspect of the reference answer. Additionally, the response mentions lowering blood pressure as the first step, but does not specify that this is done to achieve a clear time frame.',
#
# ### paraphrased
# "Despite the infarct being largely demarcated, a recanalizing therapy is still possible within a clear time window and without contraindications. The first necessary action is to decrease the patient's blood pressure.",
#
# ### feedback
# "The response is mostly correct, accurate, and factual. It correctly states that a recanalizing therapy is possible within a clear time window and without contraindications, even with a largely demarcated infarct. However, it does not explicitly mention the importance of excluding contraindications before proceeding with the therapy, which is a crucial step. Additionally, the response mentions decreasing the patient's blood pressure as the first necessary action, which is correct but not explicitly stated in the reference answer.",
#
#
# ### All three scores = 4

# %% [markdown]
# # Provide LLM retrieved docs, background and question along with either generated answer or ground truth answer. Ask LLM if the provided answer is correct, compare generated results with ground truth results

# %% [markdown]
#

# %%
import json

filepath = "/Users/cisemaltan/workspace/thesis/medical-rag-chatbot/results/generation_eval_1738062339.json"

# Read json file
with open(filepath, "r", encoding="utf-8") as file:
    data = json.load(file)

# %%
feedbacks = data[0]["all_feedbacks"]

# %%
for f in feedbacks:
    if f["llm_as_judge"]["score"] in ("3", 3):
        print(f["question_id"])

# %%
from collections import Counter

scores = [f["llm_as_judge"]["score"] for f in feedbacks]

# Count occurrences of each score
score_counts = Counter(scores)

# Print the count for each score
for score, count in score_counts.items():
    print(f"Score {score}: {count} feedbacks")

# %%
for f in feedbacks:
    if f["question_id"] == 14:
        print("Generated answer: ", f["generated_answer"])
        print("*" * 10)
        print("LLM as a judge", f["llm_as_judge"])
        print("*" * 10)
        print("-" * 10 + "Retrieved docs" + "-" * 10)
        for doc in f["retrieved_documents"]:
            print(doc["start_page"], doc["end_page"], doc["text"])
            print("-" * 10)

# %%
import sys
import os
import json
import time

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from settings.settings import settings
from settings import get_page_types, config

file_path = os.path.join(settings.data_path, settings.file_name)
# pages, _, _, _ = get_page_types()
pages = list(range(7, 109))
from core.document import get_document

document = get_document(file_path, pages)

# %%
document.get_page(15).processed_content

# %%
"7. Spezielle Krankheitsentititäten\n7.1 Symptomatische hochgradige Stenosen hirnversorgender Gefäße\nBasistherapie :engmaschiges Monitoring (beihämodynamischer Wirksamkeit und/\noder starken RR-Schwankungen incl.invasiver Blutdruckmessung), Ziel-RRbei\nhochgradigen Stenosen inderAkutphase 140-210mmHg systolisch präoperativ /\npräinterventionell .\n7.1.1 Symptomatische extrakranielle ACI-Stenosen\nBasistherapie :frühe Sekundärprophylaxe mitASS 100 mg/d (alternativ Clopidgrel\nbeiASS-Unverträglichkeit) undAtorvastatin 80mg/d .\nIndikation operative bzw.endovaskuläre Therapie beiStenosegrad ≥50%NASCET\n(starke Empfehlung beiStenosegrad 70-99%NASCET )→MNL_VA_Carotisstenose\n7.1.2 Symptomatische intrakranielle Stenosen\nPrimär aggressiv -medikamentöse Therapie (AMT) für 3 Monate, dann  Reevaluation\n(nach SAMMPRIS -Studie )\nEine endovaskuläre Therapie kann erwogen werden bei (Quelle: FDA, G -BA Urteil 2016)\n-2 oder mehr Schlaganfällen im Versorgungsgebiet der Stenose trotz AMT und\n-mind. 70%iger Stenose der intrakraniellen Gefäße und\n-mRS ≤ 3 Punkten.\nGgf.beihämodynamischen Infarkten beieingeschränkter Kollateralisierung zu\ndiskutieren, frühestens 7Tage nach vaskulärem Ereignis (Blutungsrisiko!\nAusnahme :Akutstenting während Rekanalisationsbehandlung ).1. Doppelte TAH mit ASS 100 mg/d und Clopidogrel 75 mg/d für 3 Monate\n2. Statintherapie mit Ziel -LDL < 70 mg/dl\n3. Antihypertensive Therapie mit Ziel -RR < 140/90 mmHg (bei Diabetikern < 130/80 mmHg )\n4. Lifestyle -Modifikation: BMI < 25 bzw. Reduktion des Körpergewichts um mind. 10%, \nregelmäßige körperliche Aktivität (mind. 3xwöchentlich 30 min)\n5. Coaching im Hinblick auf optimale Einstellung der vaskulären Risikofaktoren\n       "

# %%
from eval.generation import evaluate_single

toc = None
chunks = chunk_document(method="section", document=document, pages=pages, toc=toc)

faiss_service = FaissService()
faiss_service.create_index(chunks)

# %%
response = """\n\n{\n    \"relevant_sentences\": [\n        \"Die Thrombektomie ist-sofern keine Kontraindikationen vorliegen -miteiner systemischen Thrombolyse ingleicher Gesamtdosis wiebeialleiniger Therapie (d.h. 0,9mg/kgKG,max.90mg)zukombinieren (Bridging -Konzept ).\",\n        \"Imvorderen Kreislauf (MCA, ACI) innerhalb von6Stunden nach Symptom - beginn, gemäß →MNL_VA_Rekanalisationsmanagement akuter Schlaganfall .\",\n        \"Imvorderen Kreislauf (MCA, ACI) kann auch beierweitertem Zeitfenster >6h und <24hnach Symptombeginn oder unklarem Symptombeginn /Wake -Up Stroke imEinzelfall eine Rekanalisation sinnvoll sein (DAWN, NEJM 2017 ; DEFUSE 3,NEJM 2018 ).\",\n        \"Imhinteren Kreislauf (BA-Thrombose) :insbesondere beifluktuierendem Beginn kein festes Zeitfenster ;Thrombektomie sinnvoll beiKoma <6Stunden, keine klinischen Zeichen irreversibler Hirstammschädigung (Pupillen anisocor undweit, LR erloschen, CR erloschen, Atemstörung, Kreislaufdysregulation), kein ausgedehnter Hirnstamminfarkt ;gemäß → MNL_VA_Rekanalisationsmanagement akuter Schlaganfall .\"\n    ],\n    \"irrelevant_sentences\": [\n        \"Diemechanische Thrombektomie akuter intrakranieller Gefäßverschlüsse (proxi- male A.cerebri media ,A.carotis interna ,A.basilaris )insbesondere mitStent - Retrievern führt zueiner signifikanten Verbesserung derfrühen Rekanalisations - rate, derklinische Nutzen konnte mittlerweile inmehreren Studien bewiesen werden (Metaanalyse der„bigfive“,HERMES Kollaboration, Lancet 2016 ).\",\n        \"Wird eingroßer Gefäßverschluss ineinem auswärtigen Klinikum diagnostiziert, wird beifehlenden Kontraindikationen derBeginn dersystemischen Lysetherapie extern erfolgen .\",\n        \"Die Vorgehensweise beiderAnnahme erfolgt nach →MNL_VA_SOP Rekanalisation Annahme .\",\n        \"BeiZuverlegung vonextern istvorder interventionellen Behandlung insbesondere beierfolgter systemischer Thrombolyse ein ExperCT inder Angio -Suite zum Blutungsausschluss durchzuführen .\",\n        \"Postinterventionell gelten diegleichen Überwachungsmaßnahmen wiein→Kapitel 4beschrieben .\",\n        \"Sollte einAkutstenting einer symptomatischen extra -oder intrakraniellen Stenose im Rahmen derRekanalisationsbehandlung erforderlich sein, erfolgt nach Maßgabe der Kollegen der Neuroradiologie dieTherapie mitTirofiban (Dosierung und weiteres Management → MNL_CL_Tirofiban ).\",\n        \"Zudem istauf eine strengere Blutdruckeinstellung (Ziel-RR120-140mmHg systolisch) zuachten ..\",\n        \"Trotz derFortschritte inderakuten Schlaganfalltherapien gibtesweiterhin eine nicht unerhebliche Anzahl Patienten, beidenen derSchlaganfall zum Tode oder zum Überleben mitschwerer Behinderung führt.\",\n        \"Beiallen schwer betroffenen Schlaganfallpatienten istderPatientenwille (inForm einer Patientenverfügung oder mutmaßlicher Wille imGespräch mitAngehörigen / Betreuern) zuermitteln .\",\n        \"Eine endgültige Einschätzung des Krankheitsverlaufs istinden ersten Stunden oftmas nicht möglich, sodass einprimär kuratives Therapiekonzept inAbsprache mitPatient /Angehörigen zuverfolgen ist.\",\n        \"ImFalle eines schweren neurologischen Defizits auch imVerlauf istinAbsprache mitPatient /Angehörigen eine Therapiezieländerung mitsymptomorientierter Therapie zudiskutieren (i.d.R.keine Notfallents"""
error_message = """
1 validation error for ContextRelevanceResultResponse
  Input should be a valid dictionary or instance of ContextRelevanceResultResponse [type=model_type, input_value='"\\n\\n{\\n    \\"releva....d.R.keine Notfallents"', input_type=str]
    For further information visit https://errors.pydantic.dev/2.9/v/model_type
"""
from parsing.parse_try_fix import get_format_instructions
from parsing.models import ContextRelevanceResultResponse

SYSTEM_PROMPT = """You are a helpful AI assistant that assists with fixing erraneous completion formats. You will go step by step. First you'll identify the problem with the completion and then you'll fix it. You don't need to let me know what's wrong about the format, just fix it and return a valid JSON object. Don't change the content of the completion, just the format."""

FIX_OUTPUT_PROMPT = (
    "Completion:\n--------------\n{completion}\n--------------\n"
    "Instructions:\n--------------\n{instructions}\n--------------\n"
    "\nAbove, the Completion did not satisfy the constraints given in the Instructions.\n"
    "nYour task is to fix it. Respond with an answer that satisfies the constraints laid out in the Instructions and provide a valid JSON object. Say nothing else.\n"
)

user_prompt = FIX_OUTPUT_PROMPT.format(
    instructions=get_format_instructions(ContextRelevanceResultResponse), completion=response, error=error_message
)
response = generate_response(SYSTEM_PROMPT, user_prompt)
response

# %%
'"\\n\\n{\\n    \\"relevant_sentences\\": [\\n        \\"Die Thrombektomie ist-sofern keine Kontraindikationen vorliegen -miteiner systemischen Thrombolyse ingleicher Gesamtdosis wiebeialleiniger Therapie (d.h. 0,9mg/kgKG,max.90mg)zukombinieren (Bridging -Konzept ).\\",\\n        \\"Imvorderen Kreislauf (MCA, ACI) innerhalb von6Stunden nach Symptom - beginn, gemäß →MNL_VA_Rekanalisationsmanagement akuter Schlaganfall .\\",\\n        \\"Imvorderen Kreislauf (MCA, ACI) kann auch beierweitertem Zeitfenster >6h und <24hnach Symptombeginn oder unklarem Symptombeginn /Wake -Up Stroke imEinzelfall eine Rekanalisation sinnvoll sein (DAWN, NEJM 2017 ; DEFUSE 3,NEJM 2018 ).\\",\\n        \\"Imhinteren Kreislauf (BA-Thrombose) :insbesondere beifluktuierendem Beginn kein festes Zeitfenster ;Thrombektomie sinnvoll beiKoma <6Stunden, keine klinischen Zeichen irreversibler Hirstammschädigung (Pupillen anisocor undweit, LR erloschen, CR erloschen, Atemstörung, Kreislaufdysregulation), kein ausgedehnter Hirnstamminfarkt ;gemäß → MNL_VA_Rekanalisationsmanagement akuter Schlaganfall .\\"\\n    ],\\n    \\"irrelevant_sentences\\": [\\n        \\"Diemechanische Thrombektomie akuter intrakranieller Gefäßverschlüsse (proxi- male A.cerebri media ,A.carotis interna ,A.basilaris )insbesondere mitStent - Retrievern führt zueiner signifikanten Verbesserung derfrühen Rekanalisations - rate, derklinische Nutzen konnte mittlerweile inmehreren Studien bewiesen werden (Metaanalyse der„bigfive“,HERMES Kollaboration, Lancet 2016 ).\\",\\n        \\"Wird eingroßer Gefäßverschluss ineinem auswärtigen Klinikum diagnostiziert, wird beifehlenden Kontraindikationen derBeginn dersystemischen Lysetherapie extern erfolgen .\\",\\n        \\"Die Vorgehensweise beiderAnnahme erfolgt nach →MNL_VA_SOP Rekanalisation Annahme .\\",\\n        \\"BeiZuverlegung vonextern istvorder interventionellen Behandlung insbesondere beierfolgter systemischer Thrombolyse ein ExperCT inder Angio -Suite zum Blutungsausschluss durchzuführen .\\",\\n        \\"Postinterventionell gelten diegleichen Überwachungsmaßnahmen wiein→Kapitel 4beschrieben .\\",\\n        \\"Sollte einAkutstenting einer symptomatischen extra -oder intrakraniellen Stenose im Rahmen derRekanalisationsbehandlung erforderlich sein, erfolgt nach Maßgabe der Kollegen der Neuroradiologie dieTherapie mitTirofiban (Dosierung und weiteres Management → MNL_CL_Tirofiban ).\\",\\n        \\"Zudem istauf eine strengere Blutdruckeinstellung (Ziel-RR120-140mmHg systolisch) zuachten ..\\",\\n        \\"Trotz derFortschritte inderakuten Schlaganfalltherapien gibtesweiterhin eine nicht unerhebliche Anzahl Patienten, beidenen derSchlaganfall zum Tode oder zum Überleben mitschwerer Behinderung führt.\\",\\n        \\"Beiallen schwer betroffenen Schlaganfallpatienten istderPatientenwille (inForm einer Patientenverfügung oder mutmaßlicher Wille imGespräch mitAngehörigen / Betreuern) zuermitteln .\\",\\n        \\"Eine endgültige Einschätzung des Krankheitsverlaufs istinden ersten Stunden oftmas nicht möglich, sodass einprimär kuratives Therapiekonzept inAbsprache mitPatient /Angehörigen zuverfolgen ist.\\",\\n        \\"ImFalle eines schweren neurologischen Defizits auch imVerlauf istinAbsprache mitPatient /Angehörigen eine Therapiezieländerung mitsymptomorientierter Therapie zudiskutieren (i.d.R.keine Notfallentscheidung).\\"\\n    ]\\n}"'

# %%
f = evaluate_single(0, 0, faiss_service)

# %%
from parsing.answer import Answer

Answer.schema_json()

# %%
