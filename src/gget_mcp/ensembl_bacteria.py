#!/usr/bin/env python3
"""Ensembl Bacteria REST API client for bacterial genome queries.

The gget library explicitly excludes bacteria from its search() and ref() functions.
This module fills that gap by querying the Ensembl REST API directly, which supports
all species including bacteria through its unified endpoints.

Existing gget tools (seq, info) already work with bacterial Ensembl IDs — this module
provides the missing discovery layer: finding species names, gene IDs, and reference FTP URLs.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

ENSEMBL_REST = "https://rest.ensembl.org"
BACTERIA_FTP_BASE = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria"
SPECIES_INDEX_URL = f"{BACTERIA_FTP_BASE}/current/species_EnsemblBacteria.txt"

# Collection pattern from core_db column: bacteria_85_collection_core_62_115_1
COLLECTION_RE = re.compile(r"bacteria_(\d+)_collection")


class EnsemblBacteriaClient:
    """Client for querying bacterial genomes via the Ensembl REST API.

    Provides three core capabilities:
    1. Species discovery — find the full Ensembl species name from partial input
    2. Gene search — find bacterial gene IDs by symbol using xrefs
    3. Reference FTP URLs — construct download links for bacterial genome files
    """

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        # Cached species index: species_name -> {core_db, taxonomy_id, ...}
        self._species_index: Optional[Dict[str, Dict[str, str]]] = None

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def search_species(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find bacterial species in Ensembl by name or taxonomy ID.

        Uses GET /info/genomes/taxonomy/{taxon} for taxonomy-based lookups,
        then falls back to prefix matching on the species index TSV.

        Args:
            query: Partial species name (e.g. "escherichia coli k-12") or NCBI taxonomy ID
            limit: Maximum results to return

        Returns:
            List of dicts with keys: name, species, display_name, taxonomy_id,
            assembly_name, assembly_accession, division
        """
        results = []

        # Strategy 1: Try REST taxonomy endpoint (works for genus, species names, tax IDs)
        try:
            results = self._taxonomy_search(query, limit)
        except Exception as e:
            logger.debug(f"Taxonomy search failed for '{query}': {e}")

        # Strategy 2: Fall back to prefix search on species index
        if not results:
            try:
                results = self._index_prefix_search(query, limit)
            except Exception as e:
                logger.debug(f"Index prefix search failed for '{query}': {e}")

        return results[:limit]

    def search_genes(
        self,
        search_terms: List[str],
        species: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search for bacterial genes by symbol or name.

        Uses GET /xrefs/symbol/{species}/{symbol} to resolve gene symbols,
        then POST /lookup/id to batch-fetch gene details.

        Args:
            search_terms: Gene symbols to search (e.g. ["dnaA", "gyrB"])
            species: Full Ensembl species name (use search_species to find it)
            limit: Max results per search term

        Returns:
            Dict matching gget.search() output schema with keys like
            ensembl_id, gene_name, ensembl_description, biotype, etc.
        """
        all_ids = []
        id_to_symbol = {}

        for symbol in search_terms:
            try:
                xrefs = self._xref_symbol_lookup(species, symbol)
                for xref in xrefs[:limit]:
                    ens_id = xref.get("id", "")
                    if ens_id and ens_id not in id_to_symbol:
                        all_ids.append(ens_id)
                        id_to_symbol[ens_id] = symbol
            except Exception as e:
                logger.warning(f"xref lookup failed for '{symbol}' in {species}: {e}")

        if not all_ids:
            return {}

        # Batch lookup gene details
        gene_details = self._lookup_ids(all_ids)

        # Format into gget-compatible output schema
        result: Dict[str, Any] = {
            "ensembl_id": {},
            "gene_name": {},
            "ensembl_description": {},
            "biotype": {},
            "chromosome": {},
            "start": {},
            "end": {},
            "strand": {},
            "species": {},
        }

        idx = 0
        for gene_id, details in gene_details.items():
            if details is None:
                continue
            result["ensembl_id"][idx] = gene_id
            result["gene_name"][idx] = details.get("display_name", id_to_symbol.get(gene_id, ""))
            result["ensembl_description"][idx] = details.get("description", "")
            result["biotype"][idx] = details.get("biotype", "")
            result["chromosome"][idx] = details.get("seq_region_name", "")
            result["start"][idx] = details.get("start", 0)
            result["end"][idx] = details.get("end", 0)
            result["strand"][idx] = details.get("strand", 0)
            result["species"][idx] = details.get("species", species)
            idx += 1

        return result

    def get_reference_urls(
        self,
        species: str,
        which: str = "all",
        release: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get FTP download URLs for a bacterial reference genome.

        Bacteria in Ensembl are organized into numbered collections. This method
        parses the species index to find the collection number, then constructs
        FTP URLs and scrapes directory listings for actual filenames.

        Args:
            species: Full Ensembl species name (e.g. "escherichia_coli_str_k_12_substr_mg1655_gca_000005845")
            which: Data type(s) — "all", "dna", "cdna", "cds", "pep", "gtf", "ncrna"
            release: Ensembl Genomes release (default: current)

        Returns:
            Dict with keys per data type, each containing url, files list, and metadata
        """
        # Find collection number from species index
        index = self._load_species_index()
        species_lower = species.lower().replace(" ", "_")

        species_info = index.get(species_lower)
        if not species_info:
            # Try partial match
            matches = [k for k in index if species_lower in k]
            if matches:
                species_lower = matches[0]
                species_info = index[species_lower]
            else:
                raise ValueError(
                    f"Species '{species}' not found in Ensembl Bacteria index. "
                    f"Use search_species() to find the correct name."
                )

        core_db = species_info.get("core_db", "")
        match = COLLECTION_RE.search(core_db)
        if not match:
            raise ValueError(
                f"Could not determine collection number from core_db '{core_db}' "
                f"for species '{species_lower}'"
            )
        collection_num = match.group(1)

        # Determine release path
        release_path = "current" if release is None else f"release-{release}"

        # Build base URL
        collection_dir = f"bacteria_{collection_num}_collection"
        base_url = f"{BACTERIA_FTP_BASE}/{release_path}"

        # Define data types to fetch
        data_types = {
            "dna": f"fasta/{collection_dir}/{species_lower}/dna/",
            "cdna": f"fasta/{collection_dir}/{species_lower}/cdna/",
            "cds": f"fasta/{collection_dir}/{species_lower}/cds/",
            "pep": f"fasta/{collection_dir}/{species_lower}/pep/",
            "ncrna": f"fasta/{collection_dir}/{species_lower}/ncrna/",
            "gtf": f"gtf/{collection_dir}/{species_lower}/",
        }

        if which != "all":
            requested = [w.strip().lower() for w in which.split(",")]
            data_types = {k: v for k, v in data_types.items() if k in requested}

        result: Dict[str, Any] = {
            "species": species_lower,
            "collection": collection_dir,
            "taxonomy_id": species_info.get("taxonomy_id", ""),
            "assembly_accession": species_info.get("assembly", ""),
        }

        for dtype, rel_path in data_types.items():
            dir_url = f"{base_url}/{rel_path}"
            files = self._scrape_ftp_directory(dir_url)
            result[dtype] = {
                "url": dir_url,
                "files": files,
            }

        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _taxonomy_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search species via GET /info/genomes/taxonomy/{taxon}."""
        taxon = quote(query.strip(), safe="")
        url = f"{ENSEMBL_REST}/info/genomes/taxonomy/{taxon}"
        resp = self._session.get(
            url,
            headers={"Accept": "application/json"},
            timeout=self._timeout,
        )

        if resp.status_code == 400:
            # "taxon not found" — not an error, just no results
            return []
        resp.raise_for_status()

        data = resp.json()
        if not isinstance(data, list):
            data = [data]

        # Filter to bacteria division only
        results = []
        for entry in data:
            if entry.get("division", "") == "EnsemblBacteria":
                results.append({
                    "name": entry.get("name", ""),
                    "species": entry.get("species", ""),
                    "display_name": entry.get("display_name", ""),
                    "taxonomy_id": entry.get("taxonomy_id", ""),
                    "assembly_name": entry.get("assembly_name", ""),
                    "assembly_accession": entry.get("assembly_accession", ""),
                    "division": entry.get("division", ""),
                })
        return results[:limit]

    def _index_prefix_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search the species index TSV by prefix matching."""
        index = self._load_species_index()
        query_lower = query.lower().replace(" ", "_")

        matches = []
        for species_name, info in index.items():
            if query_lower in species_name:
                matches.append({
                    "name": species_name,
                    "species": species_name,
                    "display_name": info.get("species", species_name).replace("_", " ").title(),
                    "taxonomy_id": info.get("taxonomy_id", ""),
                    "assembly_name": info.get("assembly", ""),
                    "assembly_accession": info.get("assembly", ""),
                    "division": "EnsemblBacteria",
                })
            if len(matches) >= limit:
                break

        return matches

    def _xref_symbol_lookup(self, species: str, symbol: str) -> List[Dict[str, Any]]:
        """Look up gene IDs by symbol via GET /xrefs/symbol/{species}/{symbol}."""
        url = f"{ENSEMBL_REST}/xrefs/symbol/{quote(species)}/{quote(symbol)}"
        resp = self._session.get(
            url,
            headers={"Accept": "application/json"},
            timeout=self._timeout,
        )
        if resp.status_code == 400:
            return []
        resp.raise_for_status()
        return resp.json()

    def _lookup_ids(self, ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Batch-fetch gene details via POST /lookup/id."""
        if not ids:
            return {}

        url = f"{ENSEMBL_REST}/lookup/id"
        # Ensembl API accepts up to 1000 IDs per POST
        batch_size = 1000
        result: Dict[str, Optional[Dict[str, Any]]] = {}

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            resp = self._session.post(
                url,
                json={"ids": batch},
                headers={"Accept": "application/json"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            for gene_id in batch:
                result[gene_id] = data.get(gene_id)

        return result

    def _load_species_index(self) -> Dict[str, Dict[str, str]]:
        """Download and parse species_EnsemblBacteria.txt (cached for session).

        The file is a ~6MB TSV with columns including:
        #name, species, division, taxonomy_id, assembly, assembly_accession,
        genebuild, variation, pan_compara, peptide_compara, genome_alignments,
        other_alignments, core_db, species_id

        Indexes by both the 'name' column (display name with spaces/parens)
        and the 'species' column (underscore format matching REST API names).
        """
        if self._species_index is not None:
            return self._species_index

        logger.info("Downloading Ensembl Bacteria species index (~6MB)...")
        resp = self._session.get(
            SPECIES_INDEX_URL,
            headers={"Accept": "text/plain"},
            timeout=60,
        )
        resp.raise_for_status()

        index: Dict[str, Dict[str, str]] = {}
        lines = resp.text.strip().split("\n")
        header = None

        for line in lines:
            if line.startswith("#"):
                # Parse header, stripping leading '#'
                header = [h.strip().lstrip("#") for h in line.split("\t")]
                continue
            if header is None:
                continue

            fields = line.split("\t")
            if len(fields) < len(header):
                continue

            row = dict(zip(header, fields))
            name = row.get("name", "").strip().lower()
            species_col = row.get("species", "").strip().lower()
            if name:
                index[name] = row
            # Also index by 'species' column (underscore format used by REST API)
            if species_col and species_col != name:
                index[species_col] = row

        self._species_index = index
        logger.info(f"Loaded {len(index)} bacterial species entries from Ensembl index")
        return index

    def _scrape_ftp_directory(self, url: str) -> List[str]:
        """Scrape an Ensembl FTP directory listing for filenames.

        The HTTPS FTP mirror returns HTML directory listings. We extract
        href values that look like data files (ending in .gz, .fa, .gtf, etc).
        """
        try:
            resp = self._session.get(url, timeout=self._timeout)
            if resp.status_code != 200:
                return []

            # Extract filenames from href attributes in the HTML listing
            hrefs = re.findall(r'href="([^"]+)"', resp.text)
            files = [
                h for h in hrefs
                if not h.startswith("?")
                and not h.startswith("/")
                and h != "../"
                and not h.endswith("/")
            ]
            return files
        except Exception as e:
            logger.debug(f"Failed to scrape FTP directory {url}: {e}")
            return []
