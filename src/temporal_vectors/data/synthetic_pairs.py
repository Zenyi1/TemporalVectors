"""Template-based synthetic and control pair generation.

Synthetic pairs capture controlled temporal changes (tense shifts,
year-stamped facts, status changes). Control pairs change non-temporal
attributes to verify temporal vectors are not activated by irrelevant edits.
"""

import itertools
import logging
import random
from typing import Any

from temporal_vectors.config import SEED, WIKI_TARGET_YEARS

logger = logging.getLogger(__name__)

# ── Fact tables for slot-filling ─────────────────────────────────

LEADERS: dict[str, dict[int, str]] = {
    "United States": {
        2020: "Donald Trump",
        2021: "Joe Biden",
        2022: "Joe Biden",
        2023: "Joe Biden",
        2024: "Joe Biden",
    },
    "United Kingdom": {
        2020: "Boris Johnson",
        2021: "Boris Johnson",
        2022: "Rishi Sunak",
        2023: "Rishi Sunak",
        2024: "Keir Starmer",
    },
    "France": {
        2020: "Emmanuel Macron",
        2021: "Emmanuel Macron",
        2022: "Emmanuel Macron",
        2023: "Emmanuel Macron",
        2024: "Emmanuel Macron",
    },
    "Germany": {
        2020: "Angela Merkel",
        2021: "Angela Merkel",
        2022: "Olaf Scholz",
        2023: "Olaf Scholz",
        2024: "Olaf Scholz",
    },
    "Brazil": {
        2020: "Jair Bolsonaro",
        2021: "Jair Bolsonaro",
        2022: "Jair Bolsonaro",
        2023: "Lula da Silva",
        2024: "Lula da Silva",
    },
    "India": {
        2020: "Narendra Modi",
        2021: "Narendra Modi",
        2022: "Narendra Modi",
        2023: "Narendra Modi",
        2024: "Narendra Modi",
    },
    "Japan": {
        2020: "Shinzo Abe",
        2021: "Yoshihide Suga",
        2022: "Fumio Kishida",
        2023: "Fumio Kishida",
        2024: "Shigeru Ishiba",
    },
    "Australia": {
        2020: "Scott Morrison",
        2021: "Scott Morrison",
        2022: "Anthony Albanese",
        2023: "Anthony Albanese",
        2024: "Anthony Albanese",
    },
    "Canada": {
        2020: "Justin Trudeau",
        2021: "Justin Trudeau",
        2022: "Justin Trudeau",
        2023: "Justin Trudeau",
        2024: "Justin Trudeau",
    },
    "South Korea": {
        2020: "Moon Jae-in",
        2021: "Moon Jae-in",
        2022: "Yoon Suk-yeol",
        2023: "Yoon Suk-yeol",
        2024: "Yoon Suk-yeol",
    },
}

POPULATIONS: dict[str, dict[int, str]] = {
    "Tokyo": {2020: "13.96 million", 2022: "13.99 million", 2024: "14.01 million"},
    "Delhi": {2020: "30.29 million", 2022: "31.18 million", 2024: "32.07 million"},
    "Shanghai": {2020: "27.06 million", 2022: "27.80 million", 2024: "28.52 million"},
    "São Paulo": {2020: "22.04 million", 2022: "22.43 million", 2024: "22.81 million"},
    "Mexico City": {2020: "21.78 million", 2022: "22.09 million", 2024: "22.40 million"},
    "Cairo": {2020: "20.90 million", 2022: "21.32 million", 2024: "21.75 million"},
    "Mumbai": {2020: "20.67 million", 2022: "21.30 million", 2024: "21.94 million"},
    "Beijing": {2020: "20.46 million", 2022: "21.00 million", 2024: "21.54 million"},
    "Lagos": {2020: "14.37 million", 2022: "15.39 million", 2024: "16.47 million"},
    "London": {2020: "9.00 million", 2022: "8.80 million", 2024: "8.87 million"},
}

TECH_STATUS: list[dict[str, Any]] = [
    {"tech": "ChatGPT", "old_status": "an upcoming AI chatbot", "new_status": "a widely used AI chatbot", "year_old": 2022, "year_new": 2023},
    {"tech": "5G networks", "old_status": "being rolled out globally", "new_status": "widely deployed globally", "year_old": 2020, "year_new": 2023},
    {"tech": "Self-driving cars", "old_status": "in the testing phase", "new_status": "available in limited areas", "year_old": 2020, "year_new": 2024},
    {"tech": "mRNA vaccines", "old_status": "in clinical trials", "new_status": "approved and widely administered", "year_old": 2020, "year_new": 2021},
    {"tech": "The metaverse", "old_status": "a trending concept in tech", "new_status": "a declining trend in tech", "year_old": 2022, "year_new": 2024},
    {"tech": "NFTs", "old_status": "experiencing a surge in popularity", "new_status": "declining in market value", "year_old": 2021, "year_new": 2023},
    {"tech": "Quantum computing", "old_status": "a largely theoretical field", "new_status": "achieving early practical milestones", "year_old": 2020, "year_new": 2024},
    {"tech": "Electric vehicles", "old_status": "a niche market segment", "new_status": "a mainstream automotive category", "year_old": 2020, "year_new": 2024},
    {"tech": "TikTok", "old_status": "a fast-growing social media platform", "new_status": "one of the most popular social media platforms", "year_old": 2020, "year_new": 2023},
    {"tech": "Large language models", "old_status": "a research topic in AI", "new_status": "a commercial product category", "year_old": 2020, "year_new": 2024},
]

# Events that progressed through time
EVENTS: list[dict[str, str]] = [
    {"event": "The COVID-19 pandemic", "old": "is spreading rapidly around the world", "new": "has been declared endemic in many countries", "domain": "science", "year_old": "2020", "year_new": "2023"},
    {"event": "The James Webb Space Telescope", "old": "is scheduled for launch", "new": "has been successfully deployed and is returning images", "domain": "science", "year_old": "2020", "year_new": "2023"},
    {"event": "The Artemis program", "old": "is in the planning stages", "new": "has completed its first uncrewed mission", "domain": "science", "year_old": "2020", "year_new": "2023"},
    {"event": "The Russo-Ukrainian War", "old": "is a simmering conflict in eastern Ukraine", "new": "is a full-scale war following Russia's invasion", "domain": "politics", "year_old": "2021", "year_new": "2023"},
    {"event": "Brexit", "old": "is being negotiated between the UK and EU", "new": "has been completed with a trade agreement in place", "domain": "politics", "year_old": "2020", "year_new": "2022"},
    {"event": "The global chip shortage", "old": "is disrupting manufacturing worldwide", "new": "has largely been resolved", "domain": "economics", "year_old": "2021", "year_new": "2024"},
    {"event": "Remote work", "old": "is an emergency measure during lockdowns", "new": "has become a permanent option at many companies", "domain": "economics", "year_old": "2020", "year_new": "2023"},
    {"event": "The 2022 FIFA World Cup", "old": "is an upcoming tournament to be held in Qatar", "new": "was won by Argentina in Qatar", "domain": "sports", "year_old": "2021", "year_new": "2023"},
    {"event": "Inflation in the United States", "old": "is at historically low levels", "new": "has surged to multi-decade highs", "domain": "economics", "year_old": "2020", "year_new": "2022"},
    {"event": "The AI safety debate", "old": "is a niche concern among researchers", "new": "is a mainstream policy discussion", "domain": "technology", "year_old": "2020", "year_new": "2024"},
]

# ── Tense shift templates ────────────────────────────────────────

TENSE_TEMPLATES: list[dict[str, str]] = [
    {"old": "The {entity} is expected to {action} in the coming months according to officials.", "new": "The {entity} has officially {action_past} after months of deliberation and planning.", "domain": "politics"},
    {"old": "According to recent reports, the {entity} is currently {state} with several nations involved.", "new": "According to historical records, the {entity} was previously {state} with several nations involved.", "domain": "politics"},
    {"old": "The {entity} will compete in the upcoming {event} alongside other major delegations.", "new": "The {entity} competed in the recent {event} alongside other major delegations.", "domain": "sports"},
    {"old": "Scientists at leading research institutions are developing {thing} that could transform the field.", "new": "Scientists at leading research institutions developed {thing} that has transformed the field.", "domain": "science"},
    {"old": "The {entity} is planning to {action} as part of a broader economic strategy.", "new": "The {entity} has {action_past} as part of a broader economic strategy.", "domain": "economics"},
    {"old": "Officials confirm that the {entity} is preparing to {action} within the next quarter.", "new": "Officials confirmed that the {entity} {action_past} during the previous quarter.", "domain": "politics"},
    {"old": "The {entity} is actively {state} and expects to reach agreement by the end of the year.", "new": "The {entity} was actively {state} and reached agreement by the end of the year.", "domain": "economics"},
    {"old": "Experts predict that the {entity} will soon {action} given the current political climate.", "new": "Experts noted that the {entity} has already {action_past} given the political climate at the time.", "domain": "politics"},
]

TENSE_SLOTS: list[dict[str, str]] = [
    {"entity": "European Union", "action": "impose new sanctions on Russia", "action_past": "imposed new sanctions on Russia", "event": "economic summit", "state": "negotiating a comprehensive trade deal", "thing": "a new mRNA vaccine for respiratory diseases"},
    {"entity": "United Nations", "action": "adopt a landmark climate resolution", "action_past": "adopted a landmark climate resolution", "event": "General Assembly session", "state": "debating the proposed resolution on emissions", "thing": "a framework for international AI governance"},
    {"entity": "World Health Organization", "action": "declare a global health emergency", "action_past": "declared a global health emergency", "event": "annual health assembly meeting", "state": "monitoring the ongoing disease outbreak closely", "thing": "a rapid diagnostic test for emerging pathogens"},
    {"entity": "Federal Reserve", "action": "raise interest rates significantly", "action_past": "raised interest rates significantly", "event": "quarterly policy meeting", "state": "reviewing its monetary policy framework", "thing": "a new digital currency backed by the government"},
    {"entity": "NASA", "action": "launch an ambitious crewed Mars mission", "action_past": "launched an ambitious crewed Mars mission", "event": "space program review session", "state": "testing the next-generation launch system", "thing": "a reusable rocket engine for deep space travel"},
    {"entity": "government", "action": "introduce comprehensive new climate legislation", "action_past": "introduced comprehensive new climate legislation", "event": "parliamentary session on energy policy", "state": "drafting the environmental policy framework", "thing": "a large-scale carbon capture and storage system"},
    {"entity": "International Olympic Committee", "action": "select the host city for the games", "action_past": "selected the host city for the games", "event": "2028 Summer Olympics selection process", "state": "reviewing bids from multiple candidate cities", "thing": "a new standardised scoring system for gymnastics"},
    {"entity": "tech industry", "action": "adopt comprehensive AI safety standards", "action_past": "adopted comprehensive AI safety standards", "event": "global technology governance summit", "state": "developing voluntary guidelines for AI deployment", "thing": "an open-source large language model for research"},
    {"entity": "World Trade Organization", "action": "finalise new global trade agreements", "action_past": "finalised new global trade agreements", "event": "ministerial conference on trade reform", "state": "mediating disputes between member nations", "thing": "a digital platform for cross-border commerce"},
    {"entity": "African Union", "action": "establish a continental free trade zone", "action_past": "established a continental free trade zone", "event": "summit on economic integration", "state": "coordinating infrastructure investment programmes", "thing": "a mobile banking system for underserved regions"},
]

# ── Control pair templates (non-temporal changes) ────────────────

SYNONYM_SWAPS: list[tuple[str, str]] = [
    ("big", "large"),
    ("small", "tiny"),
    ("important", "significant"),
    ("show", "demonstrate"),
    ("use", "utilise"),
    ("help", "assist"),
    ("start", "begin"),
    ("end", "conclude"),
    ("fast", "rapid"),
    ("old", "ancient"),
    ("new", "novel"),
    ("hard", "difficult"),
    ("easy", "simple"),
    ("rich", "wealthy"),
    ("happy", "pleased"),
]

CONTROL_SENTENCES: list[dict[str, Any]] = [
    {"text": "The {adj1} building stands in the center of the city and attracts thousands of tourists every year.", "adj_key": "adj1", "domain": "geography"},
    {"text": "Researchers {verb1} that the experimental results are consistent with the theoretical predictions published last year.", "adj_key": "verb1", "domain": "science"},
    {"text": "The company announced plans to {verb1} its customers with the newly launched digital service platform.", "adj_key": "verb1", "domain": "economics"},
    {"text": "The {adj1} discovery fundamentally changed our understanding of modern physics and cosmology.", "adj_key": "adj1", "domain": "science"},
    {"text": "The veteran athlete made a remarkable and {adj1} comeback during the international tournament.", "adj_key": "adj1", "domain": "sports"},
    {"text": "The {adj1} economy of the region continues to grow steadily despite global uncertainty.", "adj_key": "adj1", "domain": "economics"},
    {"text": "It is {adj1} to fully understand the long-term implications of this government policy.", "adj_key": "adj1", "domain": "politics"},
    {"text": "The {adj1} nation located in Southeast Asia has a remarkably diverse cultural heritage.", "adj_key": "adj1", "domain": "geography"},
    {"text": "The coaching staff will {verb1} an intensive new training program ahead of the next competitive season.", "adj_key": "verb1", "domain": "sports"},
    {"text": "The {adj1} technology developed at the university enables significantly faster data processing.", "adj_key": "adj1", "domain": "technology"},
    {"text": "The {adj1} infrastructure project is expected to benefit millions of residents across the metropolitan area.", "adj_key": "adj1", "domain": "geography"},
    {"text": "The research team plans to {verb1} a series of experiments to validate the initial findings.", "adj_key": "verb1", "domain": "science"},
    {"text": "The {adj1} market conditions have prompted several multinational corporations to revise their strategies.", "adj_key": "adj1", "domain": "economics"},
    {"text": "The {adj1} political alliance between the two parties shaped the legislative agenda for the entire session.", "adj_key": "adj1", "domain": "politics"},
    {"text": "Analysts {verb1} that the data supports a strong correlation between the two economic indicators.", "adj_key": "verb1", "domain": "economics"},
    {"text": "The {adj1} stadium was renovated extensively to host the upcoming international sporting event.", "adj_key": "adj1", "domain": "sports"},
    {"text": "The {adj1} software platform has attracted millions of users across more than fifty countries worldwide.", "adj_key": "adj1", "domain": "technology"},
    {"text": "The panel of experts will {verb1} a comprehensive review of the existing environmental regulations.", "adj_key": "verb1", "domain": "politics"},
    {"text": "The {adj1} bridge connecting the two districts was designed by a renowned architectural firm.", "adj_key": "adj1", "domain": "geography"},
    {"text": "Engineers plan to {verb1} the development of a prototype that meets the latest industry standards.", "adj_key": "verb1", "domain": "technology"},
]


def _generate_leader_pairs(years: list[int]) -> list[dict[str, Any]]:
    """Generate pairs from leader fact table across year combinations."""
    pairs = []
    for country, year_map in LEADERS.items():
        available = sorted(set(years) & set(year_map.keys()))
        for y_old, y_new in itertools.combinations(available, 2):
            leader_old = year_map[y_old]
            leader_new = year_map[y_new]
            if leader_old != leader_new:
                pairs.append({
                    "text_old": f"As of {y_old}, the leader of {country} is {leader_old}.",
                    "text_new": f"As of {y_new}, the leader of {country} is {leader_new}.",
                    "domain": "politics",
                    "year_old": y_old,
                    "year_new": y_new,
                    "article": "",
                    "metadata": {"template": "leader", "country": country},
                })
    return pairs


def _generate_population_pairs(years: list[int]) -> list[dict[str, Any]]:
    """Generate pairs from population fact table."""
    pairs = []
    for city, year_map in POPULATIONS.items():
        available = sorted(set(years) & set(year_map.keys()))
        for y_old, y_new in itertools.combinations(available, 2):
            pop_old = year_map[y_old]
            pop_new = year_map[y_new]
            if pop_old != pop_new:
                pairs.append({
                    "text_old": f"As of {y_old}, the population of {city} is approximately {pop_old}.",
                    "text_new": f"As of {y_new}, the population of {city} is approximately {pop_new}.",
                    "domain": "geography",
                    "year_old": y_old,
                    "year_new": y_new,
                    "article": "",
                    "metadata": {"template": "population", "city": city},
                })
    return pairs


def _generate_tech_status_pairs() -> list[dict[str, Any]]:
    """Generate pairs from technology status changes."""
    pairs = []
    for entry in TECH_STATUS:
        pairs.append({
            "text_old": f"As of {entry['year_old']}, {entry['tech']} is {entry['old_status']}.",
            "text_new": f"As of {entry['year_new']}, {entry['tech']} is {entry['new_status']}.",
            "domain": "technology",
            "year_old": entry["year_old"],
            "year_new": entry["year_new"],
            "article": "",
            "metadata": {"template": "tech_status", "tech": entry["tech"]},
        })
    return pairs


def _generate_event_pairs() -> list[dict[str, Any]]:
    """Generate pairs from event progression facts."""
    pairs = []
    for entry in EVENTS:
        pairs.append({
            "text_old": f"{entry['event']} {entry['old']}.",
            "text_new": f"{entry['event']} {entry['new']}.",
            "domain": entry["domain"],
            "year_old": int(entry["year_old"]),
            "year_new": int(entry["year_new"]),
            "article": "",
            "metadata": {"template": "event", "event": entry["event"]},
        })
    return pairs


def _generate_tense_pairs(years: list[int]) -> list[dict[str, Any]]:
    """Generate pairs from tense shift templates with slot-filling."""
    pairs = []
    for template in TENSE_TEMPLATES:
        for slots in TENSE_SLOTS:
            try:
                old_text = template["old"].format(**slots)
                new_text = template["new"].format(**slots)
            except KeyError:
                continue
            # Assign year pairs from available years
            for y_old, y_new in itertools.combinations(sorted(years)[:3], 2):
                pairs.append({
                    "text_old": old_text,
                    "text_new": new_text,
                    "domain": template["domain"],
                    "year_old": y_old,
                    "year_new": y_new,
                    "article": "",
                    "metadata": {"template": "tense", "slots": slots},
                })
    return pairs


def generate_synthetic_pairs(
    years: list[int] | None = None,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Generate all synthetic temporal counterfactual pairs.

    Args:
        years: Target years for year-dependent templates.
            Defaults to config.WIKI_TARGET_YEARS.
        seed: Random seed for shuffling.

    Returns:
        List of raw pair dicts (before filtering/ID assignment).
    """
    years = years or WIKI_TARGET_YEARS
    rng = random.Random(seed)

    pairs = []
    pairs.extend(_generate_leader_pairs(years))
    pairs.extend(_generate_population_pairs(years))
    pairs.extend(_generate_tech_status_pairs())
    pairs.extend(_generate_event_pairs())
    pairs.extend(_generate_tense_pairs(years))

    rng.shuffle(pairs)
    logger.info("Generated %d raw synthetic pairs", len(pairs))
    return pairs


def _generate_synonym_controls(seed: int) -> list[dict[str, Any]]:
    """Generate control pairs by applying synonym swaps to template sentences."""
    rng = random.Random(seed)
    pairs = []

    for sent_tmpl in CONTROL_SENTENCES:
        for word_old, word_new in SYNONYM_SWAPS:
            text_template = sent_tmpl["text"]
            key = sent_tmpl["adj_key"]

            # Check if the swap type matches the slot type
            is_verb_slot = key.startswith("verb")
            is_verb_swap = word_old in ("show", "use", "help", "start", "end")

            if is_verb_slot and is_verb_swap:
                old_text = text_template.replace(f"{{{key}}}", word_old)
                new_text = text_template.replace(f"{{{key}}}", word_new)
            elif not is_verb_slot and not is_verb_swap:
                old_text = text_template.replace(f"{{{key}}}", word_old)
                new_text = text_template.replace(f"{{{key}}}", word_new)
            else:
                continue

            year = rng.choice([2021, 2022, 2023])
            pairs.append({
                "text_old": old_text,
                "text_new": new_text,
                "domain": sent_tmpl["domain"],
                "year_old": year,
                "year_new": year,  # same year -- non-temporal change
                "article": "",
                "metadata": {
                    "template": "synonym_swap",
                    "word_old": word_old,
                    "word_new": word_new,
                },
            })

    return pairs


def _generate_spelling_controls(seed: int) -> list[dict[str, Any]]:
    """Generate control pairs with British/American spelling differences."""
    rng = random.Random(seed)

    spelling_pairs = [
        ("colour", "color"),
        ("organisation", "organization"),
        ("realise", "realize"),
        ("defence", "defense"),
        ("programme", "program"),
        ("analyse", "analyze"),
        ("centre", "center"),
        ("licence", "license"),
        ("behaviour", "behavior"),
        ("favourite", "favorite"),
    ]

    context_templates = [
        "The {word} of the government's new policy was debated at length in parliament by both parties.",
        "Scientists at the national laboratory decided to {word} all of the data from the multi-year experiment.",
        "The national {word} budget was increased significantly to address growing concerns about security threats.",
        "The {word} of artificial intelligence in education has been a topic of considerable public discussion.",
        "The research {word} developed by the university team was published in a leading peer-reviewed journal.",
        "The minister announced that the {word} of the new healthcare system would be reviewed next quarter.",
        "Several analysts noted that the {word} of the company's operations had improved substantially over time.",
        "The international {word} for sustainable development was formally endorsed by all participating nations.",
        "The committee's {word} of the proposed amendments was thorough and took several weeks to complete.",
        "The {word} of the electoral process remains a key priority for the independent oversight commission.",
    ]

    pairs = []
    for brit, amer in spelling_pairs:
        for tmpl in context_templates:
            try:
                old_text = tmpl.format(word=brit)
                new_text = tmpl.format(word=amer)
            except (KeyError, IndexError):
                continue

            # Skip if the substitution doesn't make grammatical sense
            if old_text == new_text:
                continue

            year = rng.choice([2021, 2022, 2023])
            pairs.append({
                "text_old": old_text,
                "text_new": new_text,
                "domain": "politics",
                "year_old": year,
                "year_new": year,
                "article": "",
                "metadata": {
                    "template": "spelling",
                    "word_old": brit,
                    "word_new": amer,
                },
            })

    return pairs


def generate_control_pairs(seed: int = SEED) -> list[dict[str, Any]]:
    """Generate all non-temporal control counterfactual pairs.

    Args:
        seed: Random seed for shuffling.

    Returns:
        List of raw control pair dicts.
    """
    rng = random.Random(seed)

    pairs = []
    pairs.extend(_generate_synonym_controls(seed))
    pairs.extend(_generate_spelling_controls(seed))

    rng.shuffle(pairs)
    logger.info("Generated %d raw control pairs", len(pairs))
    return pairs
