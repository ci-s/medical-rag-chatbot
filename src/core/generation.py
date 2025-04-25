from prompts import (
    QUESTION_PROMPT,
    VIGNETTE_PROMPT,
    RAG_USER_PROMPT,
    QUESTION_PROMPT_w_REASONING,
    QUESTION_PROMPT_w_THINKING,
)
from domain.vignette import Vignette, Question
from domain.document import Chunk, Document
from .utils import replace_abbreviations
from settings.settings import config
from .model import generate_response
from parsing import (
    parse_with_retry,
    Summary,
    get_format_instructions,
    Answer,
    ReasoningAnswer,
    ThinkingAnswer,
    TableDescription,
    TableMarkdown,
)


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


def create_question_prompt_w_docs(retrieved_documents: list[Chunk], vignette: Vignette, question: Question) -> str:
    if config.summarize_retrieved_documents:
        documents_str = summarize_documents(retrieved_documents)
    else:
        documents_str = ""

        for idx, document in enumerate(
            retrieved_documents if config.most_relevant_chunk_first else retrieved_documents[::-1]
        ):
            page_info = (
                f"Page {document.start_page}"
                if document.start_page == document.end_page
                else f"Pages {document.start_page}-{document.end_page}"
            )

            if document.section_heading:
                documents_str += f"Document {idx + 1} (Source: [{page_info}, Section: {document.section_heading}]):\n"
                documents_str += f'"{document.text}"\n\n'
            else:
                documents_str += f"Document {idx + 1} (Source: [{page_info}]):\n"
                documents_str += f"{document.text}\n\n"

    user_prompt = create_user_question_prompt(vignette, question)

    user_prompt = RAG_USER_PROMPT.format(
        retrieved_documents=documents_str,
        user_prompt=user_prompt,
    )
    user_prompt, replaced_count = replace_abbreviations(user_prompt)

    if config.reasoning:
        system_prompt = QUESTION_PROMPT_w_REASONING.format(format_instructions=get_format_instructions(ReasoningAnswer))
    elif config.thinking:
        system_prompt = QUESTION_PROMPT_w_THINKING.format(format_instructions=get_format_instructions(ThinkingAnswer))
    else:
        system_prompt = QUESTION_PROMPT.format(format_instructions=get_format_instructions(Answer))
    return system_prompt, user_prompt


def create_question_prompt_w_docs_prod(retrieved_documents: Chunk, question: str) -> str:
    # TODO: Add support for summarization and max format changes from create_question_prompt_w_docs
    raise NotImplementedError
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
    user_prompt, replaced_count = replace_abbreviations(user_prompt)

    system_prompt = QUESTION_PROMPT.format(format_instructions=get_format_instructions(Answer))
    return system_prompt, user_prompt


def describe_table_for_generation(table: Chunk, document: Document):
    system_prompt = """
    You'll be given a table along with the context from a medical document that clinicians use to make decisions.

    Given the table in text format and its context, you'll write a detailed description in German. Description requires:
    - provide a summary first
    - then convert the table into a paragraph

    Summary should provide an general idea what the table is about and the paragraph should cover all the information in the table.

    Do not deviate from the specified format and respond strictly in the following JSON format:

    {
        "description": "<Your summary and table in text paragraph here in German>"
    }

    Do not say anything else. Make sure the response is a valid JSON.\n
    """

    user_prompt = f"""
        The context:\n{
        "\n".join(
            [
                document.get_processed_content(page_number)
                for page_number in range(table.start_page - 1, table.end_page + 1)
                if document.get_processed_content(page_number) is not None
            ]
        )
    }
        
        The table content:\n{table.text}
        """  ## start and end page are the same for tables
    response = generate_response(system_prompt, user_prompt)
    try:
        response = parse_with_retry(TableDescription, response)
        print("Response within summarization: ", response)
        return response.description
    except Exception as e:
        print("Problematic parsing:", e)
        raise e


def markdown_table_for_generation(table: Chunk, document: Document):
    system_prompt = """
    You'll be given a table from a medical document that clinicians use to make decisions. The table can contain footer notes, headers, and other formatting elements.

    Given the table in text format, you'll convert it into markdown format so that it is easier to read and understand. Don't change anything in the table, just convert it into markdown format. Keep the footer notes if there are any.

    Do not deviate from the specified format and respond strictly in the following JSON format:

    {
        "markdown": "<Table in markdown format here along with footer notes if there are any>"
    }

    Do not say anything else. Make sure the response is a valid JSON.\n
    """

    user_prompt = f"""
        The table content:\n{table.text}
        """  ## start and end page are the same for tables
    response = generate_response(system_prompt, user_prompt)
    try:
        response = parse_with_retry(TableMarkdown, response)
        print("Response within summarization: ", response)
        return response.markdown
    except Exception as e:
        print("Problematic parsing:", e)
        raise e
