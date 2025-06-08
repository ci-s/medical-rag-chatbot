DESCRIPTION_PROMPT = """You'll be given a page containing a flowchart from a medical document that clinicians use to make decisions.

        Your task is to generate a detailed, structured, and information-rich description in German that maximizes retrieval and generation effectiveness. Your response should follow these guidelines:

        - Convert the flowchart into a well-structured, coherent paragraph.
        - Group related information together logically instead of listing steps.  
        - Include clear relationships and the order between steps and decisions.
        - Avoid overly mechanical repetition; use descriptive wording and natural transitions. 

        Your response must follow this JSON format strictly:  

        {
            "description": "<Your flowchart-to-text conversion in German in one single string>"
        } <END OF JSON>
        
        Do not say anything else. Make sure the response is a valid JSON. Stop immediately at <END OF JSON>.\n
"""
