import faiss
from typing import Callable
import os
import base64
import requests
import json
import numpy as np
from collections import defaultdict

from core.embedding import embed_chunks
from settings.settings import config, settings
from core.utils import replace_abbreviations
from domain.vignette import Vignette, Question
from domain.document import Chunk, ChunkType, Document
from core.model import generate_response
from core.generation import create_user_question_prompt
from prompts import (
    HYPOTHETICAL_DOCUMENT_PROMPT,
    STEPBACK_PROMPT,
    DECOMPOSING_PROMPT,
    PARAPHRASING_PROMPT,
    TABLES_RETRIEVAL_PROMPT,
    FLOWCHART_DESCRIPTION_PROMPT,
)
from parsing import parse_with_retry, TableDescription, FlowchartDescription

from langchain_core.output_parsers import BaseOutputParser


class FaissService:
    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = None
        self.retrieval_strings: list[str] = None

    def create_index(
        self,
        chunks: list[Chunk] | list[tuple[str, Chunk]],
        retrieve_by: Callable[[Chunk], str] = lambda chunk: chunk.section_heading + " " + chunk.text
        if chunk.section_heading
        else chunk.text,
    ):
        if chunks is None:
            raise ValueError("Image representations cannot be None")

        if isinstance(chunks[0], Chunk):
            self.retrieval_strings = [retrieve_by(chunk) for chunk in chunks]
        elif isinstance(chunks[0], tuple):
            print("Ignoring retrieve_by function for tuple input")
            self.retrieval_strings = [text for text, _ in chunks]
            chunks = [chunk for _, chunk in chunks]
        else:
            raise ValueError("Invalid input type for chunks")

        embeddings = embed_chunks(
            self.retrieval_strings,
            task_type="search_document",
        )

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # TODO: parametrize
        index.add(embeddings)
        print("Index created with {} chunks".format(index.ntotal))
        self.index = index
        self.chunks = chunks
        self.set_chunk_indices()

    def search_index(self, query_embedding, k: int) -> tuple[list[float], list[Chunk]]:
        D, I = self.index.search(query_embedding, k)

        retrieved_indices = {idx: sim for idx, sim in zip(I[0], D[0])}

        if config.surrounding_chunk_length > 0:
            print("Expanding retrieved indices with surrounding chunks")
            original_indices = list(retrieved_indices.items())
            print("Original indices: ", original_indices)
            for idx, similarity_score in original_indices:
                original_section = self.chunks[idx].section_heading
                for offset in range(1, config.surrounding_chunk_length + 1):
                    for new_idx in (idx - offset, idx + offset):
                        if 0 <= new_idx < len(self.chunks) and self.chunks[new_idx].section_heading == original_section:
                            # Add only if index doesn't exist or has a lower similarity score
                            if new_idx not in retrieved_indices:
                                retrieved_indices[new_idx] = 0
                            elif retrieved_indices[new_idx] < similarity_score:
                                retrieved_indices[new_idx] = similarity_score

            print("Expanded indices: ", retrieved_indices)
        expanded_indices = sorted(retrieved_indices.items())
        expanded_indices.sort(key=lambda tup: tup[0])
        sorted_retrieved_chunks = [(self.chunks[idx], sim) for idx, sim in expanded_indices]
        # print("Sorted retrieved chunks as str: ", [(score, str(c)) for c, score in sorted_retrieved_chunks])
        scores, returned_chunks = self.merge_chunks_if_consecutive(sorted_retrieved_chunks)
        return scores, returned_chunks

    def set_chunk_indices(self):
        for i, chunk in enumerate(self.chunks):
            chunk.index = i

    def merge_chunks_if_consecutive(self, sorted_chunks: list[tuple[Chunk, float]]) -> list[Chunk]:
        merged_chunks = []

        scores = [score for _, score in sorted_chunks]
        sorted_chunks = [chunk for chunk, _ in sorted_chunks]
        current_chunk = sorted_chunks[0].copy()
        current_scores = [scores[0]]

        for i in range(1, len(sorted_chunks)):
            next_chunk = sorted_chunks[i].copy()
            next_score = scores[i]

            # No merge for tables and flowcharts
            # If type is none, might be problematic
            if (
                next_chunk.type not in (ChunkType.TABLE, ChunkType.FLOWCHART)
                and next_chunk.index == (current_chunk.index + 1)
                and next_chunk.section_heading == current_chunk.section_heading
            ):
                current_chunk.text += " " + next_chunk.text
                current_chunk.end_page = max(current_chunk.end_page, next_chunk.end_page)
                current_chunk.index = next_chunk.index
                current_scores.append(next_score)
            else:
                merged_chunks.append((current_chunk, max(current_scores)))
                current_chunk = next_chunk
                current_scores = [next_score]

        merged_chunks.append((current_chunk, max(current_scores)))

        # Sort by similarity score
        merged_chunks.sort(key=lambda x: x[1], reverse=True)
        retrieved_chunks = [chunk for chunk, _ in merged_chunks]
        scores = [score for _, score in merged_chunks]

        for chunk in retrieved_chunks:
            chunk.index = None

        return scores, retrieved_chunks


def _retrieve(query: str, faiss_service: FaissService) -> list[Chunk]:
    print("Retrieving with query: ", query)
    query, _ = replace_abbreviations(query)
    query_embedding = embed_chunks(query, task_type="search_query")

    sims, retrieved_documents = faiss_service.search_index(query_embedding, config.top_k)
    print(sims)
    return retrieved_documents


class LineListOutputParser(BaseOutputParser[list[str]]):
    """Robust output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        if "\\n" in text:
            text = text.replace("\\n", "\n")
        text = text.replace("\n\n", "\n")
        lines = text.strip().split("\n")
        cleaned_lines = [line.strip().strip('"').strip("'") for line in lines if line.strip()]

        return list(filter(None, cleaned_lines))


output_parser = LineListOutputParser()  # TODO: Move


def parse_optimized_query(response: str) -> str:
    if config.optimization_method is None:
        return response
    elif config.optimization_method == "hypothetical_document":
        return response
    elif config.optimization_method == "stepback":
        return response

    try:
        return output_parser.parse(response)
    except Exception as e:
        raise ValueError(f"Failed to parse optimized query: {e}") from e


def get_optimization_prompt() -> str:
    method = config.optimization_method

    if method == "hypothetical_document":
        return HYPOTHETICAL_DOCUMENT_PROMPT
    elif method == "stepback":
        return STEPBACK_PROMPT
    elif method == "decomposing":
        return DECOMPOSING_PROMPT
    elif method == "paraphrasing":
        return PARAPHRASING_PROMPT
    else:
        raise ValueError("Invalid optimization method")


def _retrieve_and_rank(queries: list[str], faiss_service: FaissService) -> list[Chunk]:
    # Current implementation returns top k documents with the highest scores from all queries
    # Not exactly sure if this is the best way to combine results from multiple queries
    # Because of the length of the queries, does it make sense to compare scores?
    all_retrieved_documents = []

    for query in queries:
        query, _ = replace_abbreviations(query)
        query_embedding = embed_chunks(query, task_type="search_query")

        similarities, retrieved_documents = faiss_service.search_index(query_embedding, config.top_k)
        all_retrieved_documents.extend(zip(retrieved_documents, similarities))

    all_retrieved_documents.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in all_retrieved_documents[: config.top_k]]


def retrieve(
    vignette: Vignette | None,
    question: Question | str,
    faiss_service: FaissService,
    production: bool = False,
) -> list[Chunk]:
    if isinstance(question, Question):
        query = question.get_question()
    else:
        query = question
    if production:
        if config.use_original_query_only:
            return _retrieve(query, faiss_service)
    else:
        if config.use_original_query_only:
            return _retrieve(query, faiss_service)

        print("Using optimized query with method: ", config.optimization_method)

        system_prompt = get_optimization_prompt()
        user_prompt = create_user_question_prompt(vignette, question)
        response = generate_response(user_prompt, system_prompt)
        new_query = parse_optimized_query(response)
        # add parser because decompose will return two queries
        if isinstance(new_query, list):
            retrieved_documents = _retrieve_and_rank(new_query, faiss_service)
        else:
            retrieved_documents = _retrieve(new_query, faiss_service)

        return retrieved_documents


def retrieve_and_return_optimized_query(
    vignette: Vignette | None,
    question: Question | str,
    faiss_service: FaissService,
    production: bool = False,
) -> tuple[list[Chunk], str | list[str]]:
    if isinstance(question, Question):
        query = question.get_question()
    else:
        query = question
    if production:
        if config.use_original_query_only:
            return _retrieve(query, faiss_service)
    else:
        if config.use_original_query_only:
            return _retrieve(query, faiss_service)

        print("Using optimized query with method: ", config.optimization_method)

        system_prompt = get_optimization_prompt()
        user_prompt = create_user_question_prompt(vignette, question)
        response = generate_response(user_prompt, system_prompt)
        new_query = parse_optimized_query(response)
        # add parser because decompose will return two queries
        if isinstance(new_query, list):
            retrieved_documents = _retrieve_and_rank(new_query, faiss_service)
        else:
            retrieved_documents = _retrieve(new_query, faiss_service)

        return retrieved_documents, new_query


def describe_table_for_retrieval(table: Chunk, document: Document):
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
    response = generate_response(user_prompt, TABLES_RETRIEVAL_PROMPT)
    try:
        response = parse_with_retry(TableDescription, response)
        print("Response within summarization: ", response)
        return response.description
    except Exception as e:
        print("Problematic parsing:", e)
        raise e


def gather_chunks_orderly(sorted_text_chunks: list[Chunk], sorted_table_chunks: list[Chunk]) -> list[Chunk]:
    """Expects already sorted text and table chunks and merges them in the right order.

    Args:
        sorted_text_chunks (list[Chunk]): _description_
        sorted_table_chunks (list[Chunk]): _description_

    Returns:
        list[Chunk]: _description_
    """
    merged_chunks = []
    table_index = 0  # Track position in table_chunks

    for text_chunk in sorted_text_chunks:
        # Insert all table chunks that belong *before* this text chunk
        while (
            table_index < len(sorted_table_chunks)
            and sorted_table_chunks[table_index].start_page <= text_chunk.start_page
        ):
            merged_chunks.append(sorted_table_chunks[table_index])
            table_index += 1

        # Insert the text chunk (maintaining its order)
        merged_chunks.append(text_chunk)

    # Add any remaining table chunks at the end
    while table_index < len(sorted_table_chunks):
        merged_chunks.append(sorted_table_chunks[table_index])
        table_index += 1

    return merged_chunks


def create_flowchart_chunks(flowchart_directory) -> list[Chunk]:
    # Configured for Qwen 2.5
    url = f"http://0.0.0.0:{config.vlm_port}/generate"
    headers = {"Content-Type": "application/json"}
    flowchart_directory = os.path.join(settings.data_path, "flowcharts")

    flowchart_paths = []
    for file_name in os.listdir(flowchart_directory):
        if file_name.endswith(".png"):
            flowchart_paths.append(os.path.join(flowchart_directory, file_name))

    fchunks = []
    for flowchart_path in flowchart_paths:
        try:
            page_number = int(flowchart_path.split("/")[-1].split(".")[0].replace("page", ""))
        except Exception as e:
            raise ValueError(f"Could not parse page number from {flowchart_path}: {e}")
        print(f"Processing page {page_number}")
        with open(flowchart_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        data = {"prompt": FLOWCHART_DESCRIPTION_PROMPT, "max_new_tokens": 1024, "image_input": img_base64}

        response = requests.post(url, headers=headers, data=json.dumps(data))
        print("Response:", response.text)
        try:
            parsed_response = parse_with_retry(FlowchartDescription, response.text)
            print("Response within summarization: ", parsed_response)
            fchunks.append(
                Chunk(
                    text=parsed_response.description,
                    start_page=page_number,
                    end_page=page_number,
                    type=ChunkType.FLOWCHART,
                )
            )
        except Exception as e:
            print("Problematic parsing:", e)
            raise e
    return fchunks


def reorder_flowchart_chunks(all_items: list[tuple[str, Chunk]]) -> list[tuple[str, Chunk]]:
    # Separate into flowcharts and others
    flowcharts = [item for item in all_items if item[1].type == ChunkType.FLOWCHART]
    others = [item for item in all_items if item[1].type != ChunkType.FLOWCHART]

    # Sort flowcharts by start_page
    flowcharts.sort(key=lambda x: x[1].start_page)

    reordered = []
    fc_index = 0

    for text, chunk in others:
        # Insert all flowcharts that belong before this chunk
        while fc_index < len(flowcharts) and flowcharts[fc_index][1].start_page <= chunk.start_page:
            reordered.append(flowcharts[fc_index])
            fc_index += 1

        reordered.append((text, chunk))

    # Add any remaining flowcharts
    while fc_index < len(flowcharts):
        reordered.append(flowcharts[fc_index])
        fc_index += 1

    return reordered


class HierarchicalFaissService:
    def __init__(self):
        self.layer1_index = None  # Index for section summaries/representations
        self.layer2_indices = {}  # Dictionary: section_id -> FAISS index for chunks in that section
        self.section_map = {}  # Dictionary: layer1_index_id -> section_id
        # Store tuples of (retrieval_string, Chunk) grouped by section
        self.chunks_by_section = defaultdict(list[tuple[str, Chunk]])
        self.section_retrieval_strings = []  # List of strings used for layer 1 retrieval (section IDs/headings)

    def _get_section_id(self, chunk: Chunk) -> str:
        """Determines the section identifier for a chunk."""
        # Using section_heading as the identifier. Handle cases where it might be None.
        return chunk.section_heading if chunk.section_heading else "__DEFAULT_SECTION__"

    def create_index(
        self,
        chunks: list[tuple[str, Chunk]],
    ):
        """
        Builds a hierarchical FAISS index.
        Layer 1 indexes section representations (using section headings).
        Layer 2 indexes chunks within each section (using provided retrieval strings).
        """
        if not chunks:
            raise ValueError("Chunks cannot be None or empty")

        # --- Group chunks by section ---
        for retrieval_string, chunk in chunks:  # Iterate through tuples
            section_id = self._get_section_id(chunk)
            # Store the tuple (retrieval_string, chunk)
            self.chunks_by_section[section_id].append((retrieval_string, chunk))

        print(f"Grouped chunks into {len(self.chunks_by_section)} sections.")

        # --- Build Layer 1 Index (Sections) ---
        section_ids = list(self.chunks_by_section.keys())
        # Use section_ids (headings) directly as retrieval strings for Layer 1
        # Below summaries are generated for a quick check
        self.section_retrieval_strings = [
            "Dieses Kapitel beschreibt die Zielsetzung des Handbuchs als strukturierte Behandlungsanleitung für vaskuläre neurologische Erkrankungen auf der Stroke Unit am Klinikum rechts der Isar. Es basiert auf nationalen (DGN) und internationalen (ESO, AHA) Leitlinien und ergänzt diese durch lokale SOPs. Ziel ist es, die Versorgungsqualität zu verbessern, standardisierte Abläufe zu gewährleisten und Mitarbeitenden eine praxisorientierte Grundlage zu bieten.",
            "Akutdiagnostik umfasst CCT, CTA, CTP, cMRT zur Differenzierung von Ischämie vs. Blutung, Beurteilung von Thrombolyse-/Thrombektomiekandidaten und zur Ursachensuche (TOAST-Kriterien). Weitere Maßnahmen: EKG, TTE, TEE, Duplexsonographie, Labor (inkl. Koagulation, kardiale Marker, Infektparameter), strukturierte Anamnese, Vigilanzkontrolle und standardisierte Aufnahmeprozeduren inkl. NIHSS-Score.",
            "Kontinuierliches Monitoring von RR, HF, SpO2, EKG, Temperatur, Vigilanzzustand und neurologischem Status. Tägliche Rhythmusvisite, SRAclinic zur Arrhythmieerkennung, strukturierte Dokumentation (Vitalparameter, Infusionsraten, Perfusorlaufzeit), standardisierte SOPs zur Überwachung, Protokolle zur Delirprävention und -erkennung (DOS, CAM), Checklisten für Pflege und ärztliche Visite.",
            "Grundmaßnahmen wie Frühmobilisation, enterale Ernährung (nach Schluckversuch), Atemtherapie (inkl. EzPAP), Thromboseprophylaxe, Stuhlregulation. Blutdruckmanagement mit Urapidil, Clonidin, Enalapril oder Clevidipin bei hypertensiven Werten, ggf. Akrinor, Noradrenalin, Dobutamin bei Hypotonie. Glukosekontrolle (Ziel: 100–160 mg%), Insulingabe s.c. oder i.v. inkl. Perfusorschema. Sauerstoffgabe bei SpO2 < 95 %, regelmäßige Blutgaskontrolle.",
            "Systemische Thrombolyse mit rtPA (0,9 mg/kg), mechanische Thrombektomie, Behandlung der intrazerebralen Blutung (ICB) inkl. Spot-Sign-Erkennung, engmaschige RR-Kontrollen, Anlage externer Ventrikeldrainage (EVD), intraventrikuläre Lysetherapie mit rtPA (1 mg alle 8h über EVD), Liquordiagnostik 2x/Woche bei EVD (Zellzahl, Glukose, Eiweiß, Laktat, Mikrobiologie). Subarachnoidalblutung: Aneurysmaanalyse, Nimodipin, Vasospasmusprophylaxe, ICP-Kontrolle.",
            "Rezidivvermeidung über antithrombotische Therapie (ASS, Clopidogrel, orale Antikoagulation bei VHF), Statine, Blutdrucksenkung (<130/85 mmHg), Blutzuckerkontrolle, Lifestylemodifikation. Entscheidungen gemäß TOAST-Klassifikation, Sekundärprophylaxe differenziert nach kardioembolisch, lakunär, atherothrombotisch. Anpassung der Medikation mit Blick auf Bildgebung, Laborparameter und individuelle Risikofaktoren.",
            "Behandlung von PFO, Dissektion ACI/AV, Sinusvenenthrombose, Karotisstenose. Diagnostik mit Duplexsonographie, MRT/MRA, TEE, ggf. genetische Tests. Therapieentscheidungen basieren auf Alter, Klinik, Bildgebung und Embolierisiko. Interventionelle Verfahren wie Stenting oder PFO-Verschluss, antithrombotische Langzeittherapie individuell festgelegt.",
            "Prophylaxe und Therapie von Pneumonie, HWI, tiefer Beinvenenthrombose (TVT), Lungenembolie (LE), epileptischen Anfällen, Delir. Einsatz von Präventionsbögen, strukturierte Vigilanzüberwachung, Osmotherapie bei Hirndruck (z.B. Mannitol, Hyperton-NaCl), neurochirurgische Konsile bei Raumforderungen, Fixierungsprotokolle, Protokoll für Antiepileptikagabe (Levetiracetam, Valproat, Lorazepam, Midazolam).",
            "Therapiezielklärung, Patientenwille, Dokumentation (DNR/DNI). Symptomorientierte Behandlung von Dyspnoe (Morphin, Benzodiazepine), Schmerzen (Metamizol, Morphin, Fentanyl), terminalem Rasseln (Buscopan, Scopolamin), Delir (Haloperidol, Quetiapin), Anfällen. Verzicht auf nicht lebensverlängernde Maßnahmen, individuelle Sedierung. Hygienemaßnahmen, Kommunikation mit Angehörigen, Einsatz von Seelsorge.",
            "Strukturierte Einarbeitung über SOPs, Checklisten, ärztliche und pflegerische Schulungen. Dokumentation inkl. Verlaufsbögen, Übergaben, Arztbriefvorlagen, Qualitätsmanagement mit definierten Standards (Monitoring, Mobilisation, Therapieumstellung). Interdisziplinäre Kommunikation, strukturierte Visiten, Delegation von Aufgaben, strukturierter Wissenstransfer an neue Mitarbeitende.",
        ]
        # section_ids

        if not self.section_retrieval_strings:
            raise ValueError("Could not determine any section IDs for Layer 1.")

        # Embed section headings/IDs
        section_embeddings = embed_chunks(
            self.section_retrieval_strings,
            task_type="search_document",  # Appropriate for representing document sections
        )

        if section_embeddings is None or section_embeddings.shape[0] == 0:
            raise ValueError("Failed to generate embeddings for sections.")

        d = section_embeddings.shape[1]
        self.layer1_index = faiss.IndexFlatIP(d)
        self.layer1_index.add(section_embeddings)

        # Map layer 1 index ID back to section ID
        for i, section_id in enumerate(section_ids):
            self.section_map[i] = section_id

        print(f"Layer 1 index created with {self.layer1_index.ntotal} sections.")

        # --- Build Layer 2 Indices (Chunks within Sections) ---
        for section_id, section_chunk_tuples in self.chunks_by_section.items():
            # Extract pre-computed retrieval strings from the tuples
            chunk_retrieval_strings = [rs for rs, chunk in section_chunk_tuples]

            if not chunk_retrieval_strings:
                print(f"Warning: No retrieval strings found for section '{section_id}'. Skipping L2 index creation.")
                continue

            # Embed the provided chunk retrieval strings
            chunk_embeddings = embed_chunks(
                chunk_retrieval_strings,
                task_type="search_document",
            )

            if chunk_embeddings is None or chunk_embeddings.shape[0] == 0:
                print(
                    f"Warning: Failed to generate embeddings for chunks in section '{section_id}'. Skipping L2 index creation."
                )
                continue

            d_chunk = chunk_embeddings.shape[1]
            layer2_index = faiss.IndexFlatIP(d_chunk)
            layer2_index.add(chunk_embeddings)
            self.layer2_indices[section_id] = layer2_index
            # No need to store l2_index in metadata, rely on list order within section

        print(f"Layer 2 indices created for {len(self.layer2_indices)} sections.")

    def search_index(
        self, query_embedding: np.ndarray, k: int = 5, k1_sections: int = 2
    ) -> tuple[list[float], list[Chunk]]:
        """
        Searches the hierarchical index.
        1. Search Layer 1 for relevant sections.
        2. Search Layer 2 within those sections for relevant chunks.
        3. Combine and rank results.
        """
        if self.layer1_index is None:
            raise ValueError("Index has not been created yet.")
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)  # FAISS expects 2D array

        # --- Search Layer 1 ---
        # Search L1 using the query embedding to find relevant sections
        D1, I1 = self.layer1_index.search(query_embedding, k1_sections)
        top_section_indices = I1[0]

        all_results = []

        # --- Search Layer 2 for each top section ---
        for l1_index_id in top_section_indices:
            if l1_index_id == -1:  # Faiss returns -1 for no results
                continue
            section_id = self.section_map.get(l1_index_id)
            if section_id is None or section_id not in self.layer2_indices:
                print(f"Warning: Section ID for L1 index {l1_index_id} not found or has no L2 index.")
                continue

            layer2_index = self.layer2_indices[section_id]
            print(f"Retrieved section: {section_id}")
            # Get the list of (retrieval_string, Chunk) tuples for this section
            section_chunk_tuples = self.chunks_by_section[section_id]
            # Extract just the Chunk objects for retrieval mapping
            section_chunks = [chunk for rs, chunk in section_chunk_tuples]

            if not section_chunks:
                print(f"Warning: No chunks found for section '{section_id}' during search.")
                continue

            # Search within the section's L2 index using the query embedding
            k2_chunks = max(k, 5)  # Retrieve enough chunks from relevant sections
            D2, I2 = layer2_index.search(query_embedding, k2_chunks)

            for l2_index_id, score in zip(I2[0], D2[0]):
                if l2_index_id == -1:
                    continue
                # Find the original chunk using the index within the section's chunk list
                if 0 <= l2_index_id < len(section_chunks):
                    original_chunk = section_chunks[l2_index_id]
                    all_results.append((score, original_chunk))
                else:
                    print(
                        f"Warning: Invalid L2 index {l2_index_id} for section '{section_id}' with size {len(section_chunks)}."
                    )
                    # Skip this result as the index is out of bounds

        # --- Combine and Rank Results ---
        # Sort all collected chunks by similarity score (descending)
        all_results.sort(key=lambda x: x[0], reverse=True)

        # Return top k overall chunks and their scores
        top_k_results = all_results[:k]
        final_scores = [score for score, chunk in top_k_results]
        final_chunks = [chunk for score, chunk in top_k_results]

        return final_scores, final_chunks
