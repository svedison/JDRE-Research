# JDRE_Research

UMLS Tagging (umls_expand):

A tiny helper that turns plain clinical text into a UMLS-tagged string using spaCy
 + scispaCy.
 It expands abbreviations (e.g., “AF” → atrial fibrillation) and links entities to UMLS CUIs.

 ---

What it does:

Finds biomedical entities in text

Expands abbreviations when possible

Links entities to UMLS and keeps the top matches

Returns a readable string with inline tags (CUI + canonical name)

---

An example output that I got for "The patient has atrial fibrillation (AF) and was given ASA." :
The patient{C0030705|Patients} has atrial fibrillation{C0004238|Atrial Fibrillation} (AF{C0004238|Atrial Fibrillation}) and was given ASA{C0004057|aspirin}.
