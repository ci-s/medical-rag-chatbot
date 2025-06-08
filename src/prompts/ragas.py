EXTRACT_STATEMENTS_PROMPT = """
    You are given three pieces of text: a background, a question and an answer. The background provides important medical context, while the answer might contain crucial information, such as specific values, guidelines, recommended treatments or other key data points. Your task is to extract short, fully understandable facts that is conveyed by the answer related to the background.

    The output should contain statements that:
        - Directly link the information provided in the answer to the clinical details in the background.
        - Avoid extracting information from the background that does not directly relate to the answer.
        - Do not generate vague statements i.e. "The patient should be treat with X." cannot be verified.
        - Ensure no pronouns are used in each statement.
        
    Only provide the json and say nothing else.
    
    Here is an example:
    
    Background: Es erfolgt die Vorstellung eines 81-jährigen Patienten unter dem Verdacht
    auf einen Schlaganfall. Der Patient sei etwa 45 Minuten vor Vorstellung am Boden
    liegend vorgefunden worden und habe nicht mehr aufstehen können. Zuletzt wohlauf
    (last seen well) war der Patient 20h vor Vorstellung. An Vorerkrankungen ist ein
    Bluthochdruck, ein Vorhofflimmern unter Antikoagulation mit Apixaban und eine
    Hypercholisterinämie bekannt. In der körperlichen Untersuchung ist die Patientin
    wach und bietet eine nicht-flüssige Aphasie mit hochgradiger brachiofaziale Hemiparese
    rechts und rechtsseitg positivem Babinski-Zeichen (NIHSS 12). Der Blutdruck liegt
    bei 167/87. Eine multimodale CT Bildgebung zeigt ein keine größere Infarktdemarkation.
    Die A. Cerebri media links ist verschlossen mit nach geschaltetem Perfusionsdefizit.
    Ein Notfalllabor ergibt einen INR von 1,2, eine aPTT von 28. Die Thrombozyten
    liegen bei 189.000/ µl.
    Question: Welche Maßnahme sollte nun erfolgen?
    Answer: Mechanische Rekanalisation / endovaskuläre Thrombektomie
    {
        "statements": [
            "Mechanical recanalization / thrombectomy is needed due to left middle cerebral artery occlusion and perfusion deficit.",
            "The intervention is recommended based on acute symptoms, including severe hemiparesis and non-fluent aphasia.",
            "The patient qualifies for the mechanical recanalization procedure due to the absence of major infarction on CT imaging."
        ]
        }
    
    Do not say anything else. Make sure the response is a valid JSON.
"""

FAITHFULNESS_PROMPT = """
    Consider the given context and following statements, then determine whether they are supported by the information present in the related information or background. Provide a brief explanation for each statement before arriving at the verdict (yes/no). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

    Example:
    Related information:
    The Earth revolves around the Sun and completes one full orbit in approximately 365.25 days.
    Cheese is a dairy product made by curdling milk using bacteria or enzymes.
    Tigers are carnivorous animals, known for their distinctive orange and black stripes.

    Background: 
    A zoo exhibit is being designed to educate visitors about animal habitats and diets, focusing on Asian carnivores and their roles in ecosystems.

    Question:
    Because tigers are primarily found in Asia, they are considered for the exhibit. Is there anything that would make them unsuitable?
    
    Statements:
    1. "Tigers are herbivores."
    2. "The Earth takes about 365 days to orbit the Sun."
    3. "Cheese is a natural food source for tigers in the wild."
    4. "Tigers often exist in Asia."

    Output:
    {
        "results": [
            {"statement": "Tigers are herbivores.", "verdict": "no", "explanation": "The related information explicitly states that tigers are carnivorous animals."},
            {"statement": "The Earth takes about 365 days to orbit the Sun.", "verdict": "yes", "explanation": "The related information confirms that the Earth completes an orbit in approximately 365.25 days."},
            {"statement": "Cheese is a natural food source for tigers in the wild.", "verdict": "no", "explanation": "There is no evidence that it is a food source for tigers."},
            {"statement": "Tigers often exist in Asia.", "verdict": "yes", "explanation": "The question itself mentions that tigers are primarily found in Asia."}
        ]
    }
    
    Do not say anything else. Make sure the response is a valid JSON.\n
"""

ANSWER_RELEVANCE_PROMPT = """
        Generate 3 different questions for the given answer and the background and identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers. Do not deviate from the specified format.
        
        ## Examples
        Background:
        Es erfolgt die Vorstellung eines 72-jährigen Patienten unter dem Verdacht auf einen Schlaganfall. Der Patient wurde etwa 1 Stunde vor der Aufnahme von seiner Ehefrau bewusstlos auf dem Boden gefunden. Bei der Anamnese berichtet die Ehefrau, dass der Patient seit mehreren Jahren an Vorhofflimmern leidet und mit Apixaban behandelt wird. In der körperlichen Untersuchung zeigt der Patient eine vollständige Hemiparese der rechten Körperhälfte und eine globale Aphasie. Der Blutdruck liegt bei 200/100, die Herzfrequenz ist unregelmäßig bei 110/min.
        Eine cCT-Bildgebung zeigt einen frühzeitigen Verschluss der linken A. cerebri media, ohne Hinweise auf eine intrakranielle Blutung. Ein Notfalllabor ergibt einen INR von 2,0, eine aPTT von 35 und Thrombozyten von 150.000/µl. Die Blutzuckerwerte liegen bei 140 mg/dl.
        
        Answer: Die Empfehlung ist eine sofortige intravenöse Thrombolyse mit Alteplase, falls keine Kontraindikationen vorliegen, und eine mechanische Thrombektomie aufgrund des Verschlusses der A. cerebri media.
        {
            "questions": ["Welche therapeutische Maßnahme sollte zur Wiederherstellung der Durchblutung in diesem Fall erfolgen?", 
                "Welche Behandlung ist erforderlich, um die Durchblutung in diesem Fall wiederherzustellen?", 
                "Welche Therapieoptionen kommen zur Wiederherstellung der zerebralen Durchblutung in Betracht?"],
            "noncommittal": 0
        }

        Do not say anything else. Make sure the response is a valid JSON.
    """


CONTEXT_RELEVANCE_PROMPT = """
        Extract sentences from the provided German medical text, categorizing them into ‘relevant’ and ‘irrelevant’ lists based on whether they are useful for answering the question.
        While extracting sentences you’re not allowed to make any changes to sentences from the given context. Do not deviate from the specified format and make sure to assign each sentence within context either relevant or irrelevant, don't skip.
        
        ## Examples
        Context: Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity, one of the two pillars of modern physics. He was born in Germany in 1879.
        Question: Where was Albert Einstein born?
        {
            "relevant_sentences": ["Albert Einstein was a German-born theoretical physicist.", "He was born in Germany in 1879."],
            "irrelevant_sentences": ["He developed the theory of relativity, one of the two pillars of modern physics."]
            }

        Context: The sky is blue due to the scattering of sunlight by the atmosphere. The scattering is more effective at short wavelengths, which is why the sky appears blue.
        Question: Why is the sky blue?
        {
            "relevant_sentences": ["The sky is blue due to the scattering of sunlight by the atmosphere.", "The scattering is more effective at short wavelengths, which is why the sky appears blue."],
            "irrelevant_sentences": []
            }
        
        Do not say anything else. Make sure the response is a valid JSON.
"""
