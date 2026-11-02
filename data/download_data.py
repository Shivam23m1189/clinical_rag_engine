import os
import time
import json
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse

OUTPUT_FILE = "pmc_clinical_subset.jsonl"
MAX_ARTICLES = 5000
QUERY = 'pathology OR histopathology OR epithelial OR "immunohistochemistry" OR neoplasm OR tumor'
BATCH_SIZE = 100

def search_pmc_ids(query, retmax=1000):
    params = {
        "db": "pmc",
        "term": f"{query} AND open access[filter]",
        "retmax": retmax,
        "retmode": "json",
        "sort": "pub_date"
    }
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ClinicalRAGEngine/1.0 (contact: test@example.com)"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data["esearchresult"]["idlist"]

def fetch_and_parse_pmc_batch(id_list):
    params = {
        "db": "pmc",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ClinicalRAGEngine/1.0 (contact: test@example.com)"})
    with urllib.request.urlopen(req) as resp:
        xml_content = resp.read()

    articles = []
    root = ET.fromstring(xml_content)
    
    for article in root.findall(".//article"):
        # Extract title
        title_node = article.find(".//article-title")
        title = "".join(title_node.itertext()).strip() if title_node is not None else "Unknown Title"

        # Extract PMC/PMID
        pmid_node = article.find(".//article-id[@pub-id-type='pmid']")
        pmid = pmid_node.text.strip() if pmid_node is not None else None

        # Extract abstract and body text
        abstract_nodes = article.findall(".//abstract//p")
        body_nodes = article.findall(".//body//p")

        abstract_text = "\n".join("".join(p.itertext()).strip() for p in abstract_nodes)
        body_text = "\n".join("".join(p.itertext()).strip() for p in body_nodes)
        full_text = f"{abstract_text}\n\n{body_text}".strip()

        if len(full_text) > 500:  # Skip empty or truncated articles
            articles.append({
                "pmid": pmid,
                "title": title,
                "text": full_text
            })
    return articles

def main():
    print("Searching for relevant PMC article IDs...")
    pmc_ids = search_pmc_ids(QUERY, retmax=MAX_ARTICLES)
    print(f"Found {len(pmc_ids)} candidate articles.")

    saved_count = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(0, len(pmc_ids), BATCH_SIZE):
            batch_ids = pmc_ids[i:i + BATCH_SIZE]
            try:
                records = fetch_and_parse_pmc_batch(batch_ids)
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    saved_count += 1
                    if saved_count >= MAX_ARTICLES:
                        break
                print(f"Fetched and saved: {saved_count}/{MAX_ARTICLES}")
            except Exception as e:
                print(f"Error fetching batch {i}-{i+BATCH_SIZE}: {e}")
            
            if saved_count >= MAX_ARTICLES:
                break
            time.sleep(0.4)  # Adhere to NCBI rate-limit (<3 requests/sec without API key)

    print(f"Finished. Total saved in {OUTPUT_FILE}: {saved_count}")

if __name__ == "__main__":
    main()