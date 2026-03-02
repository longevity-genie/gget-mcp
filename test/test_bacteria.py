"""Integration tests for Ensembl Bacteria tools.

These tests hit the live Ensembl REST API and FTP servers.
Mark them with @pytest.mark.integration so they can be skipped in CI.
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from gget_mcp.ensembl_bacteria import EnsemblBacteriaClient
from gget_mcp.server import GgetMCP


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def client():
    return EnsemblBacteriaClient()


@pytest.fixture
def server():
    return GgetMCP()


# Well-known test data
ECOLI_K12_SPECIES = "escherichia_coli_str_k_12_substr_mg1655_gca_000005845"
ECOLI_DNAA_LOCUS = "b3702"


# ------------------------------------------------------------------ #
# EnsemblBacteriaClient unit-level integration tests
# ------------------------------------------------------------------ #

class TestSpeciesSearch:

    @pytest.mark.integration
    def test_search_ecoli(self, client):
        """E. coli K-12 should appear when searching 'escherichia coli'."""
        results = client.search_species("Escherichia coli", limit=20)
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert any("escherichia_coli" in n for n in names)

    @pytest.mark.integration
    def test_search_bacillus(self, client):
        """Bacillus subtilis should be findable."""
        results = client.search_species("Bacillus subtilis", limit=10)
        assert len(results) > 0
        assert all(r["division"] == "EnsemblBacteria" for r in results)

    @pytest.mark.integration
    def test_search_by_taxonomy_id(self, client):
        """Taxonomy ID 511145 is E. coli K-12 MG1655."""
        results = client.search_species("511145", limit=5)
        assert len(results) > 0
        assert any("escherichia_coli" in r["name"] for r in results)

    @pytest.mark.integration
    def test_search_unknown_returns_empty(self, client):
        """A nonsense query should return an empty list, not error."""
        results = client.search_species("zzzznotaspeciesxxx", limit=5)
        assert results == []

    @pytest.mark.integration
    def test_limit_respected(self, client):
        """Limit parameter should cap results."""
        results = client.search_species("Escherichia", limit=3)
        assert len(results) <= 3


class TestGeneSearch:

    @pytest.mark.integration
    def test_search_dnaa(self, client):
        """dnaA gene should be found in E. coli K-12."""
        result = client.search_genes(["dnaA"], ECOLI_K12_SPECIES, limit=5)
        assert "ensembl_id" in result
        assert len(result["ensembl_id"]) > 0

    @pytest.mark.integration
    def test_search_multiple_genes(self, client):
        """Multiple gene symbols should return multiple results."""
        result = client.search_genes(["dnaA", "gyrB"], ECOLI_K12_SPECIES, limit=5)
        assert "ensembl_id" in result
        # Should have at least 2 results (one per gene)
        assert len(result["ensembl_id"]) >= 2

    @pytest.mark.integration
    def test_search_output_schema(self, client):
        """Output should have gget-compatible keys."""
        result = client.search_genes(["dnaA"], ECOLI_K12_SPECIES, limit=5)
        expected_keys = {"ensembl_id", "gene_name", "ensembl_description", "biotype",
                         "chromosome", "start", "end", "strand", "species"}
        assert expected_keys.issubset(result.keys())

    @pytest.mark.integration
    def test_search_unknown_gene_returns_empty(self, client):
        """Unknown gene symbol should return empty dict, not error."""
        result = client.search_genes(["zzzznotaGenexxx"], ECOLI_K12_SPECIES, limit=5)
        assert result == {} or len(result.get("ensembl_id", {})) == 0

    @pytest.mark.integration
    def test_search_bad_species_returns_empty(self, client):
        """Invalid species name should return empty dict, not crash."""
        result = client.search_genes(["dnaA"], "not_a_real_species_zzz", limit=5)
        assert result == {} or len(result.get("ensembl_id", {})) == 0


class TestReferenceUrls:

    @pytest.mark.integration
    def test_get_reference_all(self, client):
        """Should return URLs for all data types for E. coli K-12."""
        result = client.get_reference_urls(ECOLI_K12_SPECIES, which="all")
        assert "species" in result
        assert "collection" in result
        # Should have at least some data types
        for dtype in ["dna", "pep", "gtf"]:
            assert dtype in result, f"Missing data type: {dtype}"
            assert "url" in result[dtype]
            assert "files" in result[dtype]

    @pytest.mark.integration
    def test_get_reference_specific_type(self, client):
        """Requesting specific type should only return that type."""
        result = client.get_reference_urls(ECOLI_K12_SPECIES, which="dna")
        assert "dna" in result
        # Other types should not be present
        assert "pep" not in result
        assert "gtf" not in result

    @pytest.mark.integration
    def test_get_reference_files_not_empty(self, client):
        """DNA directory should contain at least one .fa.gz file."""
        result = client.get_reference_urls(ECOLI_K12_SPECIES, which="dna")
        files = result["dna"]["files"]
        assert len(files) > 0
        assert any(f.endswith(".fa.gz") for f in files)

    @pytest.mark.integration
    def test_get_reference_unknown_species_raises(self, client):
        """Unknown species should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            client.get_reference_urls("not_a_real_species_zzz")


# ------------------------------------------------------------------ #
# MCP server-level tests (through GgetMCP simplified wrappers)
# ------------------------------------------------------------------ #

class TestMCPBacteriaTools:

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_server_has_bacteria_methods(self, server):
        """Server should have the bacteria tool methods."""
        assert hasattr(server, "bacteria_search_species_simple")
        assert hasattr(server, "bacteria_search_genes_simple")
        assert hasattr(server, "bacteria_get_reference_simple")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_species_search_via_server(self, server):
        """bacteria_species tool should work through the MCP server."""
        results = await server.bacteria_search_species_simple(query="Escherichia coli")
        assert len(results) > 0
        assert any("escherichia_coli" in r["name"] for r in results)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_gene_search_via_server(self, server):
        """bacteria_search tool should work through the MCP server."""
        result = await server.bacteria_search_genes_simple(
            search_terms=["dnaA"],
            species=ECOLI_K12_SPECIES,
        )
        assert "ensembl_id" in result
        assert len(result["ensembl_id"]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reference_via_server(self, server):
        """bacteria_ref tool should work through the MCP server."""
        result = await server.bacteria_get_reference_simple(
            species=ECOLI_K12_SPECIES,
            which="dna",
        )
        assert "dna" in result
        assert "url" in result["dna"]


# ------------------------------------------------------------------ #
# End-to-end workflow test
# ------------------------------------------------------------------ #

class TestEndToEndWorkflow:
    """Test the full bacteria workflow: species → genes → info/seq."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workflow(self, server):
        """End-to-end: find species → search gene → get info with existing tools."""
        # Step 1: Find species
        species_results = await server.bacteria_search_species_simple(
            query="Escherichia coli str. K-12 substr. MG1655"
        )
        assert len(species_results) > 0
        species_name = species_results[0]["name"]
        assert "escherichia_coli" in species_name

        # Step 2: Search for dnaA gene
        gene_results = await server.bacteria_search_genes_simple(
            search_terms="dnaA",
            species=species_name,
        )
        assert "ensembl_id" in gene_results
        assert len(gene_results["ensembl_id"]) > 0
        gene_id = gene_results["ensembl_id"][0]
        assert gene_id  # Should be non-empty (e.g. 'b3702')

        # Step 3: Use existing gget_info to get gene details
        # (This proves that bacterial IDs work with existing tools)
        info_result = await server.get_gene_info_simple(ensembl_ids=gene_id)
        assert info_result is not None
