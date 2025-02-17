from prompts import QUESTION_PROMPT, VIGNETTE_PROMPT, RAG_USER_PROMPT
from domain.vignette import Vignette, Question
from domain.document import Chunk
from .utils import replace_abbreviations
from parsing import parse_with_retry, Summary
from settings.settings import config
from .model import generate_response
from parsing import get_format_instructions, Answer


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
    try:
        response = parse_with_retry(Summary, response)
        print("Response within summarization: ", response)
        return response.summary
    except Exception as e:
        print("Problematic parsing:", e)
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

    return VIGNETTE_PROMPT.format(
        background=vignette.background,
        context=context,
        preceding_question_answer_pairs=preceding_qa_str,
        query=question.get_question(),
    )


def create_question_prompt_w_docs(retrieved_documents: Chunk, vignette: Vignette, question: Question) -> str:
    if config.summarize_retrieved_documents:
        documents_str = summarize_documents(retrieved_documents)
    else:
        if config.most_relevant_chunk_first:
            documents_str = "".join([f"{document.text}\n" for document in retrieved_documents])
        else:
            documents_str = "".join([f"{document.text}\n" for document in retrieved_documents[::-1]])

    user_prompt = create_user_question_prompt(vignette, question)

    user_prompt = RAG_USER_PROMPT.format(
        retrieved_documents=documents_str,
        user_prompt=user_prompt,
    )
    # user_prompt, replaced_count = replace_abbreviations(user_prompt)

    system_prompt = QUESTION_PROMPT.format(format_instructions=get_format_instructions(Answer))
    return system_prompt, user_prompt


def create_question_prompt_w_docs_prod(retrieved_documents: Chunk, question: str) -> str:
    if config.summarize_retrieved_documents:
        documents_str = summarize_documents(retrieved_documents)
    else:
        if config.most_relevant_chunk_first:
            documents_str = "".join([f"{document.text}\n" for document in retrieved_documents])
        else:
            documents_str = "".join([f"{document.text}\n" for document in retrieved_documents[::-1]])

    user_prompt = f"Question: {question}\n"

    user_prompt = RAG_USER_PROMPT.format(
        retrieved_documents=documents_str,
        user_prompt=user_prompt,
    )
    # user_prompt, replaced_count = replace_abbreviations(user_prompt)

    system_prompt = QUESTION_PROMPT.format(format_instructions=get_format_instructions(Answer))
    return system_prompt, user_prompt
