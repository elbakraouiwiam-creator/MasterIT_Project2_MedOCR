"""
Tests for the medication matching service.
Run with: pytest tests/ -v
"""
import pytest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import MedicationDatabase

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "produits.json")


@pytest.fixture(scope="module")
def db():
    return MedicationDatabase(DB_PATH)


class TestDatabaseLoading:
    def test_loads_successfully(self, db):
        assert len(db._medications) > 0

    def test_loads_5031_products(self, db):
        assert len(db._medications) == 5031

    def test_id_index_built(self, db):
        assert len(db._id_to_med) == 5031

    def test_name_index_built(self, db):
        assert len(db._name_to_med) > 0


class TestExactMatching:
    def test_exact_match_doliprane(self, db):
        results = db.search("DOLIPRANE 500MG 20 CP", top_n=5, threshold=80)
        assert len(results) > 0
        assert results[0].score >= 80

    def test_exact_match_spasfon(self, db):
        results = db.search("SPASFON 80MG B/30 CP", top_n=5, threshold=70)
        assert len(results) > 0

    def test_case_insensitive(self, db):
        r1 = db.search("doliprane 500mg", top_n=3)
        r2 = db.search("DOLIPRANE 500MG", top_n=3)
        assert len(r1) > 0
        assert len(r2) > 0


class TestFuzzyMatching:
    def test_fuzzy_with_typo(self, db):
        # "DOLIPRANE" with a typo → should still find it
        results = db.search("DOLIPRAN 500MG", top_n=5, threshold=50)
        assert len(results) > 0
        names = [r.name for r in results]
        assert any("DOLIPRANE" in n for n in names)

    def test_partial_name(self, db):
        results = db.search("AUGMENTIN 1G", top_n=5, threshold=50)
        assert len(results) > 0

    def test_threshold_filtering(self, db):
        results_low = db.search("DOLIPRANE", top_n=10, threshold=20)
        results_high = db.search("DOLIPRANE", top_n=10, threshold=90)
        assert len(results_low) >= len(results_high)


class TestTokenMatching:
    def test_token_search(self, db):
        tokens = ["DOLIPRANE", "500MG", "16", "CP"]
        results = db.search_by_tokens(tokens, top_n=5, threshold=50)
        assert len(results) > 0

    def test_single_token(self, db):
        results = db.search_by_tokens(["ASPEGIC"], top_n=5, threshold=50)
        assert len(results) > 0


class TestGetById:
    def test_get_existing_id(self, db):
        med = db.get_by_id(59926)  # DOLIPRANE 500MG 20 CP
        assert med is not None
        assert med["id"] == 59926

    def test_get_nonexistent_id(self, db):
        med = db.get_by_id(999999)
        assert med is None


class TestListAll:
    def test_list_first_page(self, db):
        result = db.list_all(page=1, page_size=50)
        assert result["total"] == 5031
        assert len(result["items"]) == 50
        assert result["page"] == 1

    def test_pagination(self, db):
        r1 = db.list_all(page=1, page_size=10)
        r2 = db.list_all(page=2, page_size=10)
        ids1 = [m["id"] for m in r1["items"]]
        ids2 = [m["id"] for m in r2["items"]]
        assert not set(ids1).intersection(set(ids2))

    def test_search_filter(self, db):
        result = db.list_all(search="DOLIPRANE")
        assert result["total"] > 0
        for item in result["items"]:
            assert "DOLIPRANE" in item["value"].upper()


class TestNormalization:
    def test_normalize_removes_punctuation(self):
        norm = MedicationDatabase._normalize("SPASFON 80MG B/30 CP")
        assert "/" not in norm

    def test_normalize_uppercase(self):
        norm = MedicationDatabase._normalize("doliprane")
        assert norm == norm.upper()


class TestStats:
    def test_stats_returns_count(self, db):
        stats = db.get_stats()
        assert stats["total_medications"] == 5031
