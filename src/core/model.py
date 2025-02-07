import requests
import json

from settings.settings import config
from settings import LLM
from prompts import QUESTION_PROMPT, VIGNETTE_PROMPT
from domain.vignette import Vignette, Question
from domain.document import Chunk
from .utils import replace_abbreviations


def summarize_documents(retrieved_documents: list[Chunk]) -> str:
    system_prompt = """
        Consider 5 given texts and write a concise summary. Texts might start or end with an incomplete sentence, do no try to complete them.  Do not deviate from the specified format and respond strictly in the following JSON format:

        {
            "summary": "<Your summary here>"
        }

        Example:
        Text 1:
        einem ICPvon10mmHg beieiner Abklemmzeit von5Minuten), Ventil nach ICP-\nMessung wieder öffnen\n•Dokumentation derLiquorfördermengen mind .1x/Schicht (d.h.alle8Stunden)\n•Liquorfördermenge ca.5–10ml/h,max.300ml/24hzurVermeidung einer\nÜberdrainage (ggf.Anfpassung derAblaufhöhe) (Ausnahme beiinfratentorieller\nRaumforderung :max.Liquorfördermenge 150 ml/24hwegen Risiko der\nHerniation nach oben)\n•Liquordiagnostik (Zellzahl, Glukose, Eiweiß, Laktat ;mikrobiologisch Mikroskopie
        Text 2:
        undKultur) 2x/Woche bzw.beiklinischen Zeichen einer Meningitis /Ventrikulitis\n(Liquorentnahme über Bakterienfilter unter streng sterilen Kautelen)\n•während Transport /Lagerungsmaßnahmen Drainagesystem wenn möglich\n(stabiler ICP) geschlossen halten, aufjeden Fall Filter des Ablaufsystems\nschliessen\n•Abtrainieren nach →Workflow Abtrainieren derEVD\n•im Einzelfall kann eine intraventrikuläre Lysetherapie erwogen werden \n(individueller Heilversuch, siehe → MNL_VA_ICB_intraventrikuläre Lysetherapie )
        Text 3:
        •Liquordrainage bei liegender EVD (max. Fördermenge 250 ml / 24h, bei Abweichen bzw. insuffizienter \nLiquorförderung RS AVD Neurochirurgie)\n•Osmotherapie nur kurzfristig bei akuten Hirndruckkrisen, keine prophylaktische Gabe: Mannit 20% (25 \n–50 g) oder hypertone NaCl 7.5% (3 ml/kg KG)\nhierunter Kontrolle von Elektrolyten (Ziel Na < 160 mmol/l) und Serumosmolalität (Bestimmung der
        Text 4:
        Mögliche Probleme imUmgang mitder EVD:fehlende Liquorförderung bzw.\ngedämpfte ICP-Kurve →Ursachen :▪EVD disloziert, Katheterschlauch abgeknickt\n▪Drei-Wege -Hahn nicht inrichtiger Messposition\n▪Fehlposition desDruckabnehmers, feuchter Tropfkammerfilter\n▪Leitungssystem verstopft (insbes .Blutkoagel ):Durchgängigkeit desEVD-Systems\ndurch Tiefhalten derTropfkammer überprüfbar →wenn auch hier keine Liquor -\nförderung vorsichtiges Aspirieren undAnspülen desVentrikelkatheters mitmax.1,0
        Text 5:
        Patientensicherheit gewährleisten\nBei motorischer Unruhe Sitzw achen einsetzen, w enn möglich Angehörige [→ MNL_FB_Sitzwachenanforderung ] □\nReorientierung\nFrühmobilisation unterstützen/Physiotherapie einbinden; 2x/Schicht Mobilisation entsprechend\nfunktioneller Ressourcen des Patienten ermöglichen □\nSchlafrhythmus unterstützen (Schlafrituale des Patienten erheben und ermöglichen;\nNachtruhe nicht durch Geräusche oder verschiebbare Tätigkeiten stören) □
        Output:
        {
            "summary": "ICP überwachen, Liquorförderung dokumentieren (5–10 ml/h, max. 300 ml/24h; 150 ml/24h bei infratentoriellen Läsionen), Drainagesystem während des Transports geschlossen halten. EVD-Probleme wie Dislokation, Knicke oder Verstopfungen beheben; ggf. aspirieren oder spülen. EVD: max. 250 ml/24h. Osmotherapie nur bei akuten ICP-Krisen, z. B. Mannitol oder NaCl, mit Kontrolle von Elektrolyten/Osmolalität. Patientensicherheit gewährleisten, Reorientierung fördern, Frühmobilisation (2x/Schicht) unterstützen und Schlafrhythmus durch Reduzierung von Lärm und Beachtung von Patientenritualen fördern."
        }

        Do not say anything else. Make sure the response is a valid JSON.\n
    """

    user_prompt = "\n".join(f"Text {i + 1}:\n{doc.text}" for i, doc in enumerate(retrieved_documents))
    response = generate_response(system_prompt, user_prompt)
    print("Response within summarization: ", response)
    try:
        return response["summary"]
    except Exception as e:
        print("summary key not found in response or bad parsing:", e)
        raise e


def create_user_question_prompt(vignette: Vignette, question: Question) -> str:
    if config.include_preceding_question_answers:
        preceding_questions = vignette.get_preceding_questions(question.id)
        preceding_qa_str = ""
        for q in preceding_questions:
            preceding_qa_str += f"Question: {q.get_question()}\nAnswer: {q.get_answer()}\n\n"
    else:
        preceding_qa_str = ""

    if config.include_context:
        context = "Context:\n" + vignette.context
    else:
        context = ""

    print("Question: ", question.get_question())
    return VIGNETTE_PROMPT.format(
        background=vignette.background,
        context=context,
        preceding_question_answer_pairs=preceding_qa_str,
        query=question.get_question(),
    )


def create_question_prompt_w_docs(retrieved_documents, vignette: Vignette, question: Question) -> str:
    if config.summarize_retrieved_documents:
        documents_str = summarize_documents(retrieved_documents)
    else:
        if config.most_relevant_chunk_first:
            documents_str = "".join([f"{document}\n" for document in retrieved_documents])
        else:
            documents_str = "".join([f"{document}\n" for document in retrieved_documents[::-1]])

    user_prompt = create_user_question_prompt(vignette, question)

    user_prompt = QUESTION_PROMPT.format(
        retrieved_documents=documents_str,
        user_prompt=user_prompt,
    )
    prompt, replaced_count = replace_abbreviations(user_prompt)

    # print("Prompt:", prompt)  # TODO: Some logging for experimenting
    print(f"Number of abbreviations replaced: {replaced_count}")
    return prompt


class PromptFormat_llama3:
    description = "Llama3-instruct models"

    def __init__(self):
        pass

    def default_system_prompt(self):
        return """Assist users with tasks and answer questions to the best of your knowledge. Provide helpful and informative responses.Make sure to return a valid JSON."""

    def first_prompt(self):
        return (
            """<|start_header_id|>system<|end_header_id|>\n\n"""
            + """<|system_prompt|><|eot_id|>"""
            + """<|start_header_id|>user<|end_header_id|>\n\n"""
            + """<|user_prompt|><|eot_id|>"""
            + """<|start_header_id|>assistant<|end_header_id|>"""
        )

    def subs_prompt(self):
        return (
            """<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""
            + """<|user_prompt|><|eot_id|>"""
            + """<|start_header_id|>assistant<|end_header_id|>"""
        )

    def stop_conditions(self, tokenizer):
        return [tokenizer.eos_token_id, tokenizer.single_id("<|eot_id|>"), tokenizer.single_id("<|start_header_id|>")]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


def format_prompt(prompt_format: PromptFormat_llama3, user_prompt, system_prompt=None, first=True):
    if system_prompt is None:
        system_prompt = prompt_format.default_system_prompt()
    if first:
        return (
            prompt_format.first_prompt()
            .replace("<|system_prompt|>", system_prompt)
            .replace("<|user_prompt|>", user_prompt)
        )
    else:
        return prompt_format.subs_prompt().replace("<|user_prompt|>", user_prompt)


prompt_format = PromptFormat_llama3()


def generate_response(user_prompt: str, system_prompt: str = None) -> str:
    if config.inference_location == "local":
        if config.inference_type == "exllama":
            formatted_prompt = format_prompt(prompt_format, user_prompt, system_prompt, first=True)
            return LLM.generate(prompt=formatted_prompt, max_new_tokens=config.max_new_tokens, add_bos=True)
        elif config.inference_type == "ollama":
            # url = "http://localhost:11434/api/generate"
            # model_name = "llama3.1"  #:8b-instruct-q4_0
            # stream = False
            # data = {
            #     "model": model_name,
            #     "prompt": prompt,
            #     "stream": stream,
            #     "parameter": ["temperature", 0],
            #     # "options": {"seed": 42},
            #     # "format": "json",
            #     # "raw": True,
            # }  # "raw": True, "seed", 123
            # response = requests.post(url, json=data)

            # return response.json()["response"]
            return LLM.invoke(user_prompt)["output"]  # TODO: messages?
        else:
            raise ValueError("Invalid inference type")
    elif config.inference_location == "remote":
        url = "http://localhost:8082/generate"
        headers = {"Content-Type": "application/json"}

        formatted_prompt = format_prompt(prompt_format, user_prompt, system_prompt, first=True)
        data = {"prompt": formatted_prompt, "max_new_tokens": config.max_new_tokens}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_text = response.text.encode("utf-8").decode("utf-8").strip()
        try:
            parsed_response = json.loads(response_text)  # Parse JSON
            if isinstance(parsed_response, str):
                return json.loads(parsed_response.strip())  # Handle double-encoded JSON
            return parsed_response
        except json.JSONDecodeError:
            print("Failed to parse response as JSON")
            return response_text
        # try:
        # print("will try to parse response: ", response)
        # parsed_response = response.json()  # Automatically parses JSON

        # print("parsed response: ", parsed_response)
        # # Check if the parsed response contains another JSON-encoded string
        # if isinstance(parsed_response, str):
        #     print("Response is a string")
        #     return json.loads(parsed_response.strip())  # Parse inner JSON
        # return parsed_response

        # except json.JSONDecodeError:
        #     # If the response isn't valid JSON, return raw text
        #     print("Failed to parse response as JSON in generate_response")
        #     return response.text.strip()
