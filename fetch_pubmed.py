from Bio import Entrez
import pandas as pd

Entrez.email = "edisonzjy@gmail.com"
Entrez.api_key = "58e310aceb6f4fd74996b4a1fb71eb3e9709"

# search PubMed
handle = Entrez.esearch(db="pubmed", term="cancer", retmax=5)
record = Entrez.read(handle)
pmids = record["IdList"]
print(f"Found PMIDs: {pmids}")

# fetch article data
handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
raw_data = handle.read()

# save to file
with open("pubmed_raw.txt", "w") as f:
    f.write(raw_data)
print("Results saved to pubmed_raw.txt")