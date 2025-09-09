from Bio import Entrez
import pandas as pd
import scispacy
import spacy

# # search PubMed
# handle = Entrez.esearch(db="pubmed", term="cancer", retmax=5)
# record = Entrez.read(handle)
# pmids = record["IdList"]
# print(f"Found PMIDs: {pmids}")

# # fetch article data
# handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
# raw_data = handle.read()

# # save to file
# with open("pubmed_raw.txt", "w") as f:
#     f.write(raw_data)
# print("Results saved to pubmed_raw.txt")


def query_pubmed(query, max_results=10, email="edisonzjy@gmail.com", api_key='58e310aceb6f4fd74996b4a1fb71eb3e9709'):
    """
    Query PubMed for papers related to the query string.
    Returns a list of dicts with PMID, title, and abstract.

    Parameters
    ---------
    query: str
        String to query from PubMed
    max_results: int
        Number of results to retrieve
    email: str
        Email to access PubMed API
    api_key: str
        API key to access PubMed API

    Returns
    -------
    dict
    dictionary of papers containing article ID, title, and abstract
    """
    Entrez.email = email
    Entrez.api_key = api_key
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    papers = []
    if not pmids: return papers
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
    papers_data = Entrez.read(handle)
    handle.close()
    for article in papers_data["PubmedArticle"]:
        medline = article["MedlineCitation"]
        article_info = medline["Article"]
        title = article_info.get("ArticleTitle", "")
        abstract_list = article_info.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(x) for x in abstract_list)
        papers.append({
            "pmid": medline["PMID"],
            "title": title,
            "abstract": abstract
        })
    return papers

def extract_umls_concepts(texts):
    """
    Extract UMLS concepts from a list of texts using SciSpaCy's UMLS linker.
    Returns a list of dicts containing extracted concepts and CUIs.

    Parameters
    ----------
    texts: dict
        JSON-formatted papers retrieved

    Returns
    -------
    dict
        Dictionary of results
    """
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    
    linker = nlp.get_pipe("scispacy_linker")
    results = []

    for text in texts:
        doc = nlp(text)
        concepts = []
        for ent in doc.ents:
            for umls_ent in ent._.kb_ents:
                cui = umls_ent[0]
                score = umls_ent[1]
                umls_entity = linker.kb.cui_to_entity[cui]
                concepts.append({
                    "text": ent.text,
                    "cui": cui,
                    "umls_name": umls_entity.canonical_name,
                    "definition": umls_entity.definition,
                    "score": score
                })
        results.append(concepts)
    return results

papers = query_pubmed("histopathology deep learning", max_results=5)
for p in papers:
    print(f"PMID: {p['pmid']}")
    print(f"Title: {p['title']}")
    print(f"Abstract: {p['abstract']}\n")

abstracts = [p["abstract"] for p in papers if p["abstract"]]
umls_results = extract_umls_concepts(abstracts)

for i, concepts in enumerate(umls_results):
    print(f"\nAbstract {i+1}:")
    for c in concepts:
        print(f"  - {c['text']} ({c['umls_name']}) [CUI={c['cui']}] Score={c['score']:.3f}")

# pipeline function
def pubmed_to_umls(query, max_results=10000):
    papers = query_pubmed(query, max_results=max_results)
    abstracts = [p["abstract"] for p in papers if p["abstract"]]
    umls = extract_umls_concepts(abstracts)
    return papers, umls