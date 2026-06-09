"""
Medication Database & Matching Service.
Loads the Moroccan medication reference database (produits.json)
and performs fuzzy text matching.
"""
import json
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    id: int
    name: str
    score: float
    match_type: str  # "exact", "fuzzy", "partial", "token"


class MedicationDatabase:
    """
    In-memory medication database with fast fuzzy search.

    The produits.json file contains 5031 Moroccan medications
    with their IDs and names (mostly in French, some Arabic).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._medications: List[Dict] = []
        self._name_to_med: Dict[str, Dict] = {}
        self._id_to_med: Dict[int, Dict] = {}
        self._normalized_names: List[str] = []
        self._load()

    def _load(self):
        """Load and index the medication database."""
        path = Path(self.db_path)
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._medications = data
        for med in data:
            norm = self._normalize(med["value"])
            self._name_to_med[norm] = med
            self._id_to_med[med["id"]] = med
            self._normalized_names.append(norm)

        logger.info(f"Loaded {len(self._medications)} medications from database.")

    def search(self, query: str, top_n: int = 10, threshold: int = 50) -> List[MatchResult]:
        """
        Search medications by name using multiple fuzzy strategies.
        Returns top_n matches above threshold score.
        """
        if not query or len(query.strip()) < 2:
            return []

        norm_query = self._normalize(query)
        results = []

        # Strategy 1: Exact match
        if norm_query in self._name_to_med:
            med = self._name_to_med[norm_query]
            results.append(MatchResult(id=med["id"], name=med["value"], score=100.0, match_type="exact"))

        # Strategy 2: Prefix match
        for norm, med in self._name_to_med.items():
            if norm.startswith(norm_query[:min(len(norm_query), 8)]) and norm != norm_query:
                score = fuzz.ratio(norm_query, norm)
                if score >= threshold:
                    results.append(MatchResult(id=med["id"], name=med["value"], score=score, match_type="prefix"))

        # Strategy 3: Fuzzy ratio (token set ratio - best for OCR noise)
        matches = process.extract(
            norm_query,
            self._normalized_names,
            scorer=fuzz.token_set_ratio,
            limit=top_n * 2
        )

        seen_ids = {r.id for r in results}
        for match_name, score, idx in matches:
            if score < threshold:
                continue
            med = self._medications[idx]
            if med["id"] not in seen_ids:
                results.append(MatchResult(
                    id=med["id"],
                    name=med["value"],
                    score=float(score),
                    match_type="fuzzy"
                ))
                seen_ids.add(med["id"])

        # Strategy 4: Partial ratio (for substring matches)
        partial_matches = process.extract(
            norm_query,
            self._normalized_names,
            scorer=fuzz.partial_ratio,
            limit=top_n * 2
        )
        for match_name, score, idx in partial_matches:
            if score < threshold:
                continue
            med = self._medications[idx]
            if med["id"] not in seen_ids:
                results.append(MatchResult(
                    id=med["id"],
                    name=med["value"],
                    score=float(score) * 0.9,  # slight penalty for partial
                    match_type="partial"
                ))
                seen_ids.add(med["id"])

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    def search_by_tokens(self, tokens: List[str], top_n: int = 10, threshold: int = 50) -> List[MatchResult]:
        """
        Search using multiple OCR tokens and combine results.
        Useful when OCR produces individual words rather than the full name.
        """
        if not tokens:
            return []

        all_scores: Dict[int, float] = {}

        for token in tokens:
            if len(token) < 3:
                continue
            token_results = self.search(token, top_n=top_n, threshold=threshold)
            for res in token_results:
                # Accumulate scores; a medication matching multiple tokens ranks higher
                if res.id in all_scores:
                    all_scores[res.id] = max(all_scores[res.id], res.score)
                    all_scores[res.id] += res.score * 0.3  # bonus for multi-token hit
                else:
                    all_scores[res.id] = res.score

        # Build results
        results = []
        for med_id, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            med = self._id_to_med.get(med_id)
            if med:
                results.append(MatchResult(
                    id=med["id"],
                    name=med["value"],
                    score=min(score, 100.0),
                    match_type="token"
                ))

        return results

    def get_by_id(self, med_id: int) -> Optional[Dict]:
        return self._id_to_med.get(med_id)

    def list_all(self, page: int = 1, page_size: int = 50, search: Optional[str] = None) -> Dict:
        meds = self._medications
        if search:
            norm_search = self._normalize(search)
            meds = [m for m in meds if norm_search in self._normalize(m["value"])]

        total = len(meds)
        start = (page - 1) * page_size
        end = start + page_size
        items = meds[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "items": items
        }

    def get_stats(self) -> Dict:
        return {
            "total_medications": len(self._medications),
            "database_file": str(self.db_path),
        }

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize medication name for comparison."""
        if not text:
            return ""
        text = text.upper().strip()
        # Normalize spaces and punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
