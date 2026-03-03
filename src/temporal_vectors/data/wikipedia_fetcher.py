"""Wikipedia revision fetcher using mwclient.

Fetches article revisions at specified timestamps, extracts lead sections,
and diffs them at sentence level to produce temporal counterfactual pairs.
"""

import difflib
import logging
import re
import time
from pathlib import Path

import mwclient

from temporal_vectors.config import DATA_RAW, WIKI_RATE_LIMIT_SECONDS, WIKI_TARGET_YEARS

logger = logging.getLogger(__name__)


# ── Curated article lists per domain ─────────────────────────────
DOMAIN_ARTICLES: dict[str, list[str]] = {
    "politics": [
        "Joe_Biden",
        "Donald_Trump",
        "Kamala_Harris",
        "Vladimir_Putin",
        "Volodymyr_Zelenskyy",
        "Boris_Johnson",
        "Rishi_Sunak",
        "Emmanuel_Macron",
        "Xi_Jinping",
        "Narendra_Modi",
        "Olaf_Scholz",
        "Lula_da_Silva",
        "United_States_Congress",
        "European_Parliament",
        "NATO",
        "United_Nations_General_Assembly",
        "2020_United_States_presidential_election",
        "2024_United_States_presidential_election",
        "Brexit",
        "Russo-Ukrainian_War",
    ],
    "technology": [
        "ChatGPT",
        "GPT-4",
        "Artificial_intelligence",
        "Large_language_model",
        "Tesla,_Inc.",
        "SpaceX",
        "Cryptocurrency",
        "Bitcoin",
        "Metaverse",
        "5G",
        "Quantum_computing",
        "Neuralink",
        "Self-driving_car",
        "DALL-E",
        "Midjourney",
        "TikTok",
        "Twitter",
        "OpenAI",
        "Apple_Vision_Pro",
        "Starlink",
    ],
    "science": [
        "COVID-19_pandemic",
        "COVID-19_vaccine",
        "CRISPR_gene_editing",
        "James_Webb_Space_Telescope",
        "Monkeypox",
        "Artemis_program",
        "Climate_change",
        "mRNA_vaccine",
        "Nuclear_fusion",
        "SARS-CoV-2",
        "Long_COVID",
        "Omicron_variant",
        "Mars_2020",
        "Perseverance_(rover)",
        "Chandrayaan-3",
        "ITER",
        "Dark_energy",
        "Gravitational_wave",
        "AlphaFold",
        "Genome_editing",
    ],
    "sports": [
        "2020_Summer_Olympics",
        "2022_FIFA_World_Cup",
        "Lionel_Messi",
        "Cristiano_Ronaldo",
        "LeBron_James",
        "Lewis_Hamilton",
        "Novak_Djokovic",
        "Premier_League",
        "UEFA_Champions_League",
        "Super_Bowl",
        "2022_Winter_Olympics",
        "FIFA_World_Cup",
        "Kylian_Mbappé",
        "Erling_Haaland",
        "Max_Verstappen",
        "Simone_Biles",
        "Tokyo_2020_Olympics",
        "NFL",
        "Indian_Premier_League",
        "Formula_One",
    ],
    "geography": [
        "Istanbul",
        "Dubai",
        "Singapore",
        "Tokyo",
        "London",
        "New_York_City",
        "Shanghai",
        "Sydney",
        "Berlin",
        "Mumbai",
        "São_Paulo",
        "Lagos",
        "World_population",
        "Urbanization",
        "Climate_change_and_cities",
        "Delhi",
        "Cairo",
        "Mexico_City",
        "Jakarta",
        "Seoul",
    ],
    "economics": [
        "Inflation",
        "2020_stock_market_crash",
        "Federal_Reserve",
        "Cryptocurrency",
        "Supply_chain_crisis",
        "Silicon_Valley_Bank",
        "Recession",
        "Gross_domestic_product",
        "European_Central_Bank",
        "Quantitative_easing",
        "GameStop_short_squeeze",
        "Non-fungible_token",
        "OPEC",
        "World_Bank",
        "International_Monetary_Fund",
        "Economic_impact_of_the_COVID-19_pandemic",
        "Student_debt",
        "Housing_bubble",
        "Gig_economy",
        "Remote_work",
    ],
}


def _cache_path(article: str, year: int) -> Path:
    """Return the cache file path for a given article and year."""
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", article)
    return DATA_RAW / f"{safe_name}_{year}.txt"


def _extract_lead_section(wikitext: str) -> str:
    """Extract the lead section (text before the first == heading ==).

    Args:
        wikitext: Raw MediaWiki markup.

    Returns:
        Lead section text with basic markup stripped.
    """
    # Split at first section heading
    parts = re.split(r"\n==[^=]", wikitext, maxsplit=1)
    lead = parts[0]

    # Strip common wiki markup
    lead = re.sub(r"\{\{[^}]*\}\}", "", lead)  # templates
    lead = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", lead)  # [[link|text]] -> text
    lead = re.sub(r"<ref[^>]*>.*?</ref>", "", lead, flags=re.DOTALL)  # <ref>...</ref>
    lead = re.sub(r"<ref[^/]*/?>", "", lead)  # <ref .../> self-closing
    lead = re.sub(r"<[^>]+>", "", lead)  # remaining HTML tags
    lead = re.sub(r"'{2,}", "", lead)  # bold/italic markers
    lead = re.sub(r"\s+", " ", lead)  # collapse whitespace

    return lead.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic.

    Args:
        text: Plain text string.

    Returns:
        List of sentence strings.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def fetch_revision_text(
    site: mwclient.Site,
    article: str,
    year: int,
    rate_limit: float = WIKI_RATE_LIMIT_SECONDS,
) -> str | None:
    """Fetch the article text from the closest revision to mid-year.

    Results are cached in data/raw/ to avoid re-fetching.

    Args:
        site: An mwclient.Site instance.
        article: Wikipedia article title.
        year: Target year (fetches revision closest to July 1st).
        rate_limit: Minimum seconds between API requests.

    Returns:
        Raw wikitext string, or None if the article/revision is not found.
    """
    cache = _cache_path(article, year)
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    cache.parent.mkdir(parents=True, exist_ok=True)

    try:
        page = site.pages[article]
        # Target mid-year for a representative snapshot
        timestamp = f"{year}0701000000"

        revisions = list(
            page.revisions(
                start=timestamp,
                limit=1,
                dir="older",
                prop="content|timestamp",
            )
        )
        time.sleep(rate_limit)

        if not revisions:
            logger.warning("No revision found for %s at %d", article, year)
            return None

        content = revisions[0].get("*", "")
        if not content:
            logger.warning("Empty content for %s at %d", article, year)
            return None

        cache.write_text(content, encoding="utf-8")
        logger.debug("Cached %s_%d (%d chars)", article, year, len(content))
        return content

    except Exception as e:
        logger.error("Failed to fetch %s at %d: %s", article, year, e)
        time.sleep(rate_limit)
        return None


def diff_lead_sections(
    text_old: str,
    text_new: str,
) -> list[tuple[str, str]]:
    """Diff two lead sections at sentence level, returning changed pairs.

    Args:
        text_old: Wikitext of the older revision.
        text_new: Wikitext of the newer revision.

    Returns:
        List of (old_sentence, new_sentence) tuples where content changed.
    """
    lead_old = _extract_lead_section(text_old)
    lead_new = _extract_lead_section(text_new)

    sents_old = _split_sentences(lead_old)
    sents_new = _split_sentences(lead_new)

    matcher = difflib.SequenceMatcher(None, sents_old, sents_new)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            # Pair up replaced sentences (zip truncates to shorter side)
            for s_old, s_new in zip(sents_old[i1:i2], sents_new[j1:j2]):
                if s_old.strip() != s_new.strip():
                    pairs.append((s_old.strip(), s_new.strip()))

    return pairs


def fetch_article_pairs(
    site: mwclient.Site,
    article: str,
    domain: str,
    year_old: int,
    year_new: int,
    rate_limit: float = WIKI_RATE_LIMIT_SECONDS,
) -> list[dict]:
    """Fetch revisions for an article at two years and return diff pairs.

    Args:
        site: An mwclient.Site instance.
        article: Wikipedia article title.
        domain: Topic domain label for metadata.
        year_old: Earlier year.
        year_new: Later year.
        rate_limit: Seconds between API requests.

    Returns:
        List of raw pair dicts (before filtering). Each dict has keys:
        text_old, text_new, article, domain, year_old, year_new.
    """
    text_old = fetch_revision_text(site, article, year_old, rate_limit)
    text_new = fetch_revision_text(site, article, year_new, rate_limit)

    if text_old is None or text_new is None:
        return []

    changed = diff_lead_sections(text_old, text_new)
    pairs = []
    for s_old, s_new in changed:
        pairs.append(
            {
                "text_old": s_old,
                "text_new": s_new,
                "article": article,
                "domain": domain,
                "year_old": year_old,
                "year_new": year_new,
            }
        )

    return pairs


def create_site(user_agent: str = "TemporalVectorsBot/0.1") -> mwclient.Site:
    """Create an mwclient Site instance for English Wikipedia.

    Args:
        user_agent: User-Agent string for the API requests.

    Returns:
        Connected mwclient.Site.
    """
    site = mwclient.Site("en.wikipedia.org", clients_useragent=user_agent)
    return site
