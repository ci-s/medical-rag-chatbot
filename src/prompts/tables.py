RETRIEVAL_PROMPT = """
        You'll be given a table along with the context from a medical document that clinicians use to make decisions.

        Your task is to generate a detailed, structured, and information-rich description in German that maximizes retrieval effectiveness. Your response should include:

        1. A Clear Summary (2-3 sentences):  
        - Provide a concise yet informative overview of what the table represents.  
        - Include key medical concepts and terms clinicians might search for.  
        - Use synonyms and alternative phrasing to capture diverse query formulations.  

        2. A Detailed Table-to-Text Description:  
        - Convert the table into a well-structured, coherent paragraph.
        - Group related information together logically instead of listing data row by row.  
        - Include clear relationships between values (e.g., comparisons, trends, categories).  
        - Avoid overly mechanical repetition; use descriptive wording and natural transitions. 

        Your response must follow this JSON format strictly:  

        {
            "description": "<Your summary and table-to-text conversion in German>"
        }
        
        Do not say anything else. Make sure the response is a valid JSON.\n
"""

DESCRIPTION_GENERATION_PROMPT = """
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

MARKDOWN_GENERATION_PROMPT = """
    You'll be given a table from a medical document that clinicians use to make decisions. The table can contain footer notes, headers, and other formatting elements.

    Given the table in text format, you'll convert it into markdown format so that it is easier to read and understand. Don't change anything in the table, just convert it into markdown format. Keep the footer notes if there are any.

    Do not deviate from the specified format and respond strictly in the following JSON format:

    {
        "markdown": "<Table in markdown format here along with footer notes if there are any>"
    }

    Do not say anything else. Make sure the response is a valid JSON.\n
    """
