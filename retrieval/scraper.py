"""
MIT Course Catalog Scraper
Scrapes student.mit.edu/catalog and outputs data/courses.json
in the schema expected by retrieval/indexer.py

Usage:
    pip install requests beautifulsoup4
    python scraper.py
"""
import json
import re
import time
import os

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL   = "https://student.mit.edu/catalog/"
INDEX_URL  = "https://student.mit.edu/catalog/index.cgi"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "courses.json")

# Map icon filenames (stem only, lowercased) → canonical schema values
DIST_REQ_ICONS = {
    "rest":         "REST",
    "cihw":         "CI-HW",
    "cih":          "CI-HW",
    "hassa":        "HASS-A",
    "hassh":        "HASS-H",
    "hasss":        "HASS-S",
    "lab":          "Institute Lab",
}
# Fallback: map alt-text substrings → canonical values
DIST_REQ_ALT = {
    "rest elec":            "REST",
    "communication intens": "CI-HW",
    "hass arts":            "HASS-A",
    "hass human":           "HASS-H",
    "hass social":          "HASS-S",
    "institute lab":        "Institute Lab",
}

LEVEL_ICONS = {
    "under": "Undergraduate",
    "grad":  "Graduate",
}
SEMESTER_ICONS = {
    "fall":   "Fall",
    "spring": "Spring",
    "iap":    "IAP",
    "summer": "Summer",
}

# Icons that are GIR markers or decorative — ignore for dist req purposes
IGNORE_ICONS = {"calc1", "calc2", "calc3", "hr", "purple1", "purple2"}


def _icon_stem(src: str) -> str:
    """'/icns/rest.gif' → 'rest'"""
    return src.split("/")[-1].rsplit(".", 1)[0].lower()


def fetch(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            r.encoding = "iso-8859-1"
            return r.text
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def get_catalog_pages() -> list[str]:
    """
    Return all m*.html department page URLs.
    Step 1: fetch index.cgi to get the first page of each department.
    Step 2: for each page, follow any pagination links (e.g. m18a → m18b → m18c)
            that appear in the in-page navigation table.
    """
    # Step 1: seed from the index
    html = fetch(INDEX_URL)
    soup = BeautifulSoup(html, "html.parser")
    pages: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if re.match(r"^m\w+\.html$", href):
            pages.add(BASE_URL + href)

    # Step 2: follow pagination links within each page
    # The nav table at the top links to sibling pages (m18b.html, m18c.html, etc.)
    to_visit = list(pages)
    while to_visit:
        url = to_visit.pop()
        try:
            page_html = fetch(url)
        except Exception:
            continue
        page_soup = BeautifulSoup(page_html, "html.parser")
        for a in page_soup.find_all("a", href=True):
            href = a["href"].strip()
            if re.match(r"^m\w+\.html$", href):
                full = BASE_URL + href
                if full not in pages:
                    pages.add(full)
                    to_visit.append(full)
        time.sleep(0.3)

    return sorted(pages)


def _parse_icons(p_tag) -> tuple[list, list, list]:
    """Return (level, semesters_offered, distribution_requirements) from <img> tags."""
    level, semesters, dist = [], [], []
    for img in p_tag.find_all("img"):
        src  = img.get("src", "")
        alt  = (img.get("alt") or "").strip()
        stem = _icon_stem(src)

        if stem in IGNORE_ICONS or alt == "______":
            continue

        if stem in LEVEL_ICONS:
            v = LEVEL_ICONS[stem]
            if v not in level:
                level.append(v)
        elif stem in SEMESTER_ICONS:
            v = SEMESTER_ICONS[stem]
            if v not in semesters:
                semesters.append(v)
        elif stem in DIST_REQ_ICONS:
            v = DIST_REQ_ICONS[stem]
            if v not in dist:
                dist.append(v)
        else:
            # Fallback: check alt text
            alt_lower = alt.lower()
            for key, val in DIST_REQ_ALT.items():
                if key in alt_lower:
                    if val not in dist:
                        dist.append(val)
                    break

    return level, semesters, dist


def _parse_br_fields(p_tag) -> tuple[str | None, str | None, list[str]]:
    """Return (prereqs, units, credit_not_for) from <br>-separated lines."""
    raw_html = str(p_tag)
    parts = re.split(r"<br\s*/?>", raw_html, flags=re.I)

    prereqs        = None
    units          = None
    credit_not_for = []

    for part in parts:
        text = BeautifulSoup(part, "html.parser").get_text(" ").strip()
        text = re.sub(r"\s+", " ", text)
        lower = text.lower()

        if lower.startswith("prereq:"):
            prereqs = text[len("prereq:"):].strip()
        elif lower.startswith("units:"):
            units = text[len("units:"):].strip()
        elif lower.startswith("credit cannot"):
            credit_not_for = re.findall(r"\b\d+[\w.]+\b", text)

    return prereqs, units, credit_not_for


def _parse_schedule(p_tag) -> dict[str, str]:
    """Extract {Lecture: time, Recitation: time, ...} from bold/italic patterns."""
    schedule = {}
    raw = str(p_tag)
    for m in re.finditer(r"<b>([^<:]+):</b>\s*<i>([^<]+)</i>", raw):
        key = m.group(1).strip()
        val = m.group(2).strip()
        if key and val:
            schedule[key] = val
    return schedule


def _parse_description(p_tag) -> str | None:
    """Text after the second horizontal-rule image."""
    hrs = p_tag.find_all("img", alt="______")
    if len(hrs) < 2:
        return None

    desc_parts = []
    for sibling in hrs[1].next_siblings:
        if hasattr(sibling, "get_text"):
            t = sibling.get_text(" ").strip()
        else:
            t = str(sibling).strip()
        if t:
            desc_parts.append(t)

    text = " ".join(desc_parts)
    # Strip instructor/textbook lines at the end
    text = re.sub(r"(Fall|Spring|IAP|Summer|Year):\s*[A-Z][^<\n]*", "", text)
    text = re.sub(r"No textbook information available\.?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def parse_course(anchor, page_url: str, departments: list[str]) -> dict | None:
    course_number = anchor.get("name", "").strip()
    if not course_number:
        return None

    p = anchor.find_next_sibling("p")
    if not p:
        return None

    # Title
    h3 = p.find("h3")
    if not h3:
        return None
    h3_text = h3.get_text(separator=" ").strip()
    h3_text = re.sub(r"\s+", " ", h3_text)
    title   = re.sub(rf"^{re.escape(course_number)}\s*", "", h3_text).strip()
    is_new  = bool(re.search(r"\(New\)", title, re.I))
    title   = re.sub(r"\s*\(New\)\s*", "", title).strip()

    level, semesters, dist = _parse_icons(p)
    prereqs, units, credit_not_for = _parse_br_fields(p)
    schedule    = _parse_schedule(p)
    description = _parse_description(p)

    return {
        "course_number":           course_number,
        "title":                   title,
        "url":                     f"{page_url}#{course_number}",
        "level":                   level,
        "semesters_offered":       semesters,
        "distribution_requirements": dist,
        "is_joint":                False,
        "same_subject_as":         [],
        "meets_with":              [],
        "credit_not_for":          credit_not_for,
        "prerequisites":           prereqs,
        "units":                   units,
        "schedule":                schedule,
        "schedule_notes":          None,
        "description":             description,
        "instructors":             [],
        "is_new":                  is_new,
        "course_url":              None,
        "departments":             departments,
    }


def get_department_name(soup: BeautifulSoup) -> list[str]:
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(" ").strip()
        m = re.search(r"Course\s+[\w.]+:\s*(.+)", text)
        if m:
            dept = m.group(1).split("\n")[0].strip()
            dept = re.sub(r"\s+", " ", dept)
            return [dept]
    title_tag = soup.find("title")
    if title_tag:
        m = re.search(r"Course\s+[\w.]+:\s*(.+)", title_tag.get_text())
        if m:
            return [m.group(1).strip()]
    return []


# Regex for valid course number anchors: "18.01", "6.3900", "STS.001", "WGS.160J"
_COURSE_ANCHOR_RE = re.compile(r"^[A-Z0-9]+\.[A-Z0-9]+$", re.I)


def scrape_page(url: str) -> list[dict]:
    html        = fetch(url)
    soup        = BeautifulSoup(html, "html.parser")
    departments = get_department_name(soup)

    courses = []
    for anchor in soup.find_all("a", attrs={"name": True}):
        name = anchor.get("name", "").strip()
        if not _COURSE_ANCHOR_RE.match(name):
            continue
        course = parse_course(anchor, url, departments)
        if course:
            courses.append(course)

    return courses


def main():
    print("Fetching catalog index...")
    pages = get_catalog_pages()
    print(f"Found {len(pages)} department pages\n")

    all_courses: list[dict] = []
    seen: set[str] = set()

    for page_url in tqdm(pages, desc="Scraping pages"):
        try:
            for course in scrape_page(page_url):
                if course["course_number"] not in seen:
                    all_courses.append(course)
                    seen.add(course["course_number"])
            time.sleep(0.4)   # polite crawl delay
        except Exception as e:
            tqdm.write(f"  ERROR {page_url}: {e}")

    print(f"\nTotal unique courses: {len(all_courses)}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_courses, f, ensure_ascii=False, indent=2)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
