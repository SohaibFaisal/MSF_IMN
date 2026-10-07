from pathlib import Path

bib_file = Path("newnewnew.txt")

with open(bib_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

papers = {}
current_entry = []

entry_id = 0

for line in lines:
    # Blank line means end of one reference
    if line.strip() == "":
        if current_entry:
            papers[entry_id] = current_entry.copy()
            entry_id += 1
            current_entry = []
    else:
        current_entry.append(line.rstrip("\n"))

# Add final entry if file does not end with a blank line
if current_entry:
    papers[entry_id] = current_entry.copy()

print(f"Found {len(papers)} references.")

for key, value in papers.items():
    print(f"\n--- Paper {key} ---")
    for line in value:
        print(line)



import re
import time
import requests
from urllib.parse import quote


# ---------------------------------------------------------
# Extract information from your existing BibTeX entry
# ---------------------------------------------------------

def get_bibtex_key(entry_lines):
    """
    Extract:
        Wu.2025
    from:
        @article{Wu.2025,
    """

    first_line = entry_lines[0]

    match = re.search(r'@\w+\s*\{\s*([^,]+)', first_line)

    if match:
        return match.group(1).strip()

    return None


def get_field(entry_lines, field_name):
    """
    Extract a BibTeX field such as title or author.
    Works with fields spanning multiple lines.
    """

    text = "\n".join(entry_lines)

    pattern = rf'{field_name}\s*=\s*[\{{"](.+?)[\}}"]\s*,?\s*(?=\n\s*\w+\s*=|\n\s*\}})'

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        value = match.group(1)

        # Clean whitespace
        value = re.sub(r'\s+', ' ', value)

        # Remove common LaTeX braces used around titles
        value = value.replace("{", "").replace("}", "")

        return value.strip()

    return None


# ---------------------------------------------------------
# Search Crossref
# ---------------------------------------------------------

def search_crossref(title, authors=None):

    query = title

    if authors:
        query += " " + authors

    url = (
        "https://api.crossref.org/works"
        f"?query.bibliographic={quote(query)}"
        "&rows=1"
    )

    headers = {
        "User-Agent": "BibTeXFixer/1.0"
    }

    r = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    r.raise_for_status()

    data = r.json()

    items = data["message"]["items"]

    if not items:
        return None

    return items[0]


# ---------------------------------------------------------
# Retrieve BibTeX using DOI
# ---------------------------------------------------------

def get_bibtex_from_doi(doi):

    url = f"https://doi.org/{doi}"

    headers = {
        "Accept": "application/x-bibtex",
        "User-Agent": "BibTeXFixer/1.0"
    }

    r = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    r.raise_for_status()

    return r.text.strip()


# ---------------------------------------------------------
# Replace citation key
# ---------------------------------------------------------

def replace_bibtex_key(bibtex, original_key):

    # Example:
    #
    # @article{NewKey,
    #
    # becomes
    #
    # @article{OriginalKey,

    bibtex = re.sub(
        r'^(@\w+\s*\{)\s*[^,]+,',
        rf'\1{original_key},',
        bibtex,
        count=1,
        flags=re.IGNORECASE
    )

    return bibtex


# ---------------------------------------------------------
# Process dictionary
# ---------------------------------------------------------

def update_papers(papers):

    updated_papers = {}

    total = len(papers)

    for i, (dict_key, entry_lines) in enumerate(papers.items(), start=1):

        print()
        print("=" * 80)
        print(f"Processing {i}/{total}")
        print("=" * 80)

        original_key = get_bibtex_key(entry_lines)
        title = get_field(entry_lines, "title")
        authors = get_field(entry_lines, "author")

        print("Key:    ", original_key)
        print("Title:  ", title)
        print("Authors:", authors)

        # Safety check
        if original_key is None or title is None:

            print("!!! Could not extract key/title")
            print("Keeping original entry.")

            updated_papers[dict_key] = entry_lines

            continue

        try:

            # ---------------------------------
            # Search paper
            # ---------------------------------

            result = search_crossref(
                title,
                authors
            )

            if result is None:

                print("!!! No result found.")
                updated_papers[dict_key] = entry_lines
                continue

            doi = result.get("DOI")

            found_title = result.get(
                "title",
                ["UNKNOWN"]
            )[0]

            print()
            print("Found:")
            print(found_title)
            print("DOI:", doi)

            if doi is None:

                print("!!! No DOI available.")
                updated_papers[dict_key] = entry_lines
                continue

            # ---------------------------------
            # Download BibTeX
            # ---------------------------------

            new_bibtex = get_bibtex_from_doi(doi)

            # ---------------------------------
            # Restore original key
            # ---------------------------------

            new_bibtex = replace_bibtex_key(
                new_bibtex,
                original_key
            )

            # Store as list of lines like your
            # original dictionary
            updated_papers[dict_key] = new_bibtex.splitlines()

            print()
            print("UPDATED:")
            print(new_bibtex)

        except Exception as e:

            print()
            print("!!! ERROR:")
            print(e)

            print("Keeping original entry.")

            updated_papers[dict_key] = entry_lines

        # Be nice to the API
        time.sleep(0.5)

    return updated_papers

updated_papers = update_papers(papers)


output_file = "references_fixed.bib"

with open(output_file, "w", encoding="utf-8") as f:

    for entry in updated_papers.values():

        for line in entry:
            f.write(line + "\n")

        f.write("\n\n")

print(f"\nSaved to: {output_file}")