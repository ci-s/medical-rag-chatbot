HEADINGS_PROMPT = """
Read the PDF page containing table of content below and provide the list of headings in JSON format as follows:
[
{
    "number": "4",
    "heading": "Allgemeine Behandlungsmaßnahmen / Basistherapie"
},
{
    "number": "4.1",
    "heading": "Oxygenierung"
}
]

for a content as follows:
    "Inhalt
    4. Allgemeine Behandlungsmaßnahmen / Basistherapie ………………………………………....
    4.1 Oxygenierung ….............................
    88
    92"

Please ensure the list starts from the very first heading in the content and continues up to the third level of subheadings. Only include headings from the table of contents, excluding "Inhalt." Do not include subheadings beyond the third level (e.g., no 4.1.1.2). Do not say anything else and don't use markdown. Make sure the response is a valid JSON.\n

"""
