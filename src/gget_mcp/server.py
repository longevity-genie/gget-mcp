#!/usr/bin/env python3
"""gget MCP Server - Bioinformatics query interface using the gget library."""

import os
from typing import List, Optional, Union

from fastmcp import FastMCP
from eliot import start_action
import gget

# Configuration
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3002"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")

# Removed GgetResponse wrapper - tools now return data directly per MCP best practices

class GgetMCP(FastMCP):
    """gget MCP Server with bioinformatics tools."""
    
    def __init__(
        self, 
        name: str = "gget MCP Server",
        prefix: str = "gget_",
        **kwargs
    ):
        """Initialize the gget tools with FastMCP functionality."""
        super().__init__(name=name, **kwargs)
        
        self.prefix = prefix
        self._register_gget_tools()
    
    def _register_gget_tools(self):
        """Register gget-specific tools."""
        
        # Gene information and search tools
        self.tool(name=f"{self.prefix}search")(self.search_genes)
        self.tool(name=f"{self.prefix}info")(self.get_gene_info)
        self.tool(name=f"{self.prefix}seq")(self.get_sequences)
        
        # Reference genome tools
        self.tool(name=f"{self.prefix}ref")(self.get_reference)
        
        # Sequence analysis tools
        self.tool(name=f"{self.prefix}blast")(self.blast_sequence)
        self.tool(name=f"{self.prefix}blat")(self.blat_sequence)
        
        # Alignment tools
        self.tool(name=f"{self.prefix}muscle")(self.muscle_align)
        self.tool(name=f"{self.prefix}diamond")(self.diamond_align)
        
        # Expression and functional analysis
        self.tool(name=f"{self.prefix}archs4")(self.archs4_expression)
        self.tool(name=f"{self.prefix}enrichr")(self.enrichr_analysis)
        self.tool(name=f"{self.prefix}bgee")(self.bgee_orthologs)
        
        # Protein structure and function
        self.tool(name=f"{self.prefix}pdb")(self.get_pdb_structure)
        self.tool(name=f"{self.prefix}alphafold")(self.alphafold_predict)
        self.tool(name=f"{self.prefix}elm")(self.elm_analysis)
        
        # Cancer and mutation analysis
        self.tool(name=f"{self.prefix}cosmic")(self.cosmic_search)
        self.tool(name=f"{self.prefix}mutate")(self.mutate_sequences)
        
        # Drug and disease analysis
        self.tool(name=f"{self.prefix}opentargets")(self.opentargets_analysis)
        
        # Single-cell analysis
        self.tool(name=f"{self.prefix}cellxgene")(self.cellxgene_query)
        
        # Setup and utility functions
        self.tool(name=f"{self.prefix}setup")(self.setup_databases)

    async def search_genes(
        self, 
        search_terms: List[str], 
        species: str = "homo_sapiens",
        limit: int = 100
    ):
        """Search for genes using gene symbols, names, or synonyms.
        
        Use this tool FIRST when you have gene names/symbols and need to find their Ensembl IDs.
        Returns Ensembl IDs which are required for get_gene_info and get_sequences tools.
        
        Args:
            search_terms: List of gene symbols (e.g., ['TP53', 'BRCA1']) or names
            species: Target species (e.g., 'homo_sapiens', 'mus_musculus')
            limit: Maximum number of results per search term
        
        Example:
            Input: search_terms=['BRCA1'], species='homo_sapiens'
            Output: {'BRCA1': {'ensembl_id': 'ENSG00000012048', 'description': 'BRCA1 DNA repair...'}}
        
        Downstream tools that need the Ensembl IDs from this search:
            - get_gene_info: Get detailed gene information  
            - get_sequences: Get DNA/protein sequences
        """
        with start_action(action_type="gget_search", search_terms=search_terms, species=species):
            result = gget.search(search_terms, species=species, limit=limit)
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_gene_info(
        self, 
        ensembl_ids: List[str],
        verbose: bool = True
    ):
        """Get detailed information for genes using their Ensembl IDs.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: List of Ensembl gene IDs (e.g., ['ENSG00000141510'])
            verbose: Include additional annotation details
        
        Example workflow:
            1. search_genes(['TP53'], 'homo_sapiens') → get Ensembl ID 'ENSG00000141510'
            2. get_gene_info(['ENSG00000141510']) 
            
        Example output:
            {'ENSG00000141510': {'symbol': 'TP53', 'biotype': 'protein_coding', 
             'start': 7661779, 'end': 7687550, 'chromosome': '17'...}}
        """
        with start_action(action_type="gget_info", ensembl_ids=ensembl_ids):
            result = gget.info(ensembl_ids, verbose=verbose)
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_sequences(
        self, 
        ensembl_ids: List[str],
        translate: bool = False,
        isoforms: bool = False
    ):
        """Fetch nucleotide or amino acid sequences for genes.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: List of Ensembl gene IDs (e.g., ['ENSG00000141510'])
            translate: If True, returns protein sequences; if False, returns DNA sequences
            isoforms: Include alternative splice isoforms
        
        Example workflow for protein sequence:
            1. search_genes(['TP53'], 'homo_sapiens') → 'ENSG00000141510'
            2. get_sequences(['ENSG00000141510'], translate=True)
            
        Example output (protein):
            {'ENSG00000141510': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP...'}
            
        Example output (DNA):
            {'ENSG00000141510': 'ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTC...'}
        
        Downstream tools that use protein sequences:
            - alphafold_predict: Predict 3D structure from protein sequence
            - blast_sequence: Search for similar sequences
        """
        with start_action(action_type="gget_seq", ensembl_ids=ensembl_ids, translate=translate):
            result = gget.seq(ensembl_ids, translate=translate, isoforms=isoforms)
            return result

    async def get_reference(
        self, 
        species: str = "homo_sapiens",
        which: str = "all",
        release: Optional[int] = None
    ):
        """Get reference genome information from Ensembl."""
        with start_action(action_type="gget_ref", species=species, which=which):
            result = gget.ref(species=species, which=which, release=release)
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def blast_sequence(
        self, 
        sequence: str,
        program: str = "blastp",
        database: str = "nr",
        limit: int = 50,
        expect: float = 10.0
    ):
        """BLAST a nucleotide or amino acid sequence."""
        with start_action(action_type="gget_blast", sequence_length=len(sequence), program=program):
            result = gget.blast(
                sequence=sequence,
                program=program,
                database=database,
                limit=limit,
                expect=expect
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def blat_sequence(
        self, 
        sequence: str,
        seqtype: str = "DNA",
        assembly: str = "hg38"
    ):
        """Find genomic location of a sequence using BLAT."""
        with start_action(action_type="gget_blat", sequence_length=len(sequence), assembly=assembly):
            result = gget.blat(
                sequence=sequence,
                seqtype=seqtype,
                assembly=assembly
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def muscle_align(
        self, 
        sequences: List[str],
        super5: bool = False
    ):
        """Align multiple sequences using MUSCLE."""
        with start_action(action_type="gget_muscle", num_sequences=len(sequences)):
            result = gget.muscle(sequences=sequences, super5=super5)
            return result

    async def diamond_align(
        self, 
        sequences: Union[str, List[str]],
        reference: str,
        sensitivity: str = "very-sensitive",
        threads: int = 1
    ):
        """Align amino acid sequences to a reference using DIAMOND."""
        with start_action(action_type="gget_diamond", sensitivity=sensitivity):
            result = gget.diamond(
                sequences=sequences,
                reference=reference,
                sensitivity=sensitivity,
                threads=threads
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def archs4_expression(
        self, 
        gene: str,
        which: str = "tissue",
        species: str = "human"
    ):
        """Get tissue expression data from ARCHS4."""
        with start_action(action_type="gget_archs4", gene=gene, which=which):
            result = gget.archs4(gene=gene, which=which, species=species)
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def enrichr_analysis(
        self, 
        genes: List[str],
        database: str = "KEGG_2021_Human",
        species: str = "human"
    ):
        """Perform functional enrichment analysis using Enrichr."""
        with start_action(action_type="gget_enrichr", genes=genes, database=database):
            result = gget.enrichr(
                genes=genes,
                database=database,
                species=species
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def bgee_orthologs(
        self, 
        gene_id: str,
        type: str = "orthologs"
    ):
        """Find orthologs of a gene using Bgee database.
        
        PREREQUISITE: Use search_genes to get Ensembl ID first.
        
        Args:
            gene_id: Ensembl gene ID (e.g., 'ENSG00000012048' for BRCA1)
            type: Type of data ('orthologs' or 'expression')
        
        Example workflow:
            1. search_genes(['BRCA1']) → 'ENSG00000012048' 
            2. bgee_orthologs('ENSG00000012048') → ortholog data
        """
        with start_action(action_type="gget_bgee", gene_id=gene_id, type=type):
            result = gget.bgee(gene_id=gene_id, type=type)
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_pdb_structure(
        self, 
        pdb_id: str,
        resource: str = "pdb"
    ):
        """Fetch protein structure data from PDB using specific PDB IDs.
        
        IMPORTANT: This tool requires a specific PDB ID (e.g., '2GS6'), NOT gene names.
        
        For gene-to-structure workflows:
        1. Use search_genes to get Ensembl ID
        2. Use get_sequences with translate=True to get protein sequence  
        3. Use alphafold_predict for structure prediction, OR
        4. Search external databases (PDB website) for known PDB IDs, then use this tool
        
        Args:
            pdb_id: Specific PDB structure ID (e.g., '2GS6', '1EGF')
            resource: Database resource ('pdb' or 'alphafold')
        
        Example:
            Input: pdb_id='2GS6'
            Output: Structure data with coordinates, resolution, method, etc.
            
        Alternative workflow for gene structure prediction:
            1. search_genes(['EGFR']) → get Ensembl ID
            2. get_sequences([ensembl_id], translate=True) → get protein sequence
            3. alphafold_predict(protein_sequence) → predict structure
        """
        with start_action(action_type="gget_pdb", pdb_id=pdb_id):
            result = gget.pdb(pdb_id=pdb_id, resource=resource)
            return result

    async def alphafold_predict(
        self, 
        sequence: str,
        out: Optional[str] = None
    ):
        """Predict protein structure using AlphaFold from protein sequence.
        
        PREREQUISITE: Use get_sequences with translate=True to get protein sequence first.
        
        Workflow for gene structure prediction:
        1. search_genes → get Ensembl ID
        2. get_sequences with translate=True → get protein sequence
        3. alphafold_predict → predict structure
        
        Args:
            sequence: Amino acid sequence (protein, not DNA)
            out: Optional output directory for structure files
        
        Example full workflow:
            1. search_genes(['TP53']) → 'ENSG00000141510'
            2. get_sequences(['ENSG00000141510'], translate=True) → 'MEEPQSDPSVEPPLSQ...'
            3. alphafold_predict('MEEPQSDPSVEPPLSQ...')
            
        Example output:
            AlphaFold structure prediction data with confidence scores and coordinates
        """
        with start_action(action_type="gget_alphafold", sequence_length=len(sequence)):
            result = gget.alphafold(sequence=sequence, out=out)
            return result

    async def elm_analysis(
        self, 
        sequence: str,
        sensitivity: str = "very-sensitive",
        threads: int = 1,
        uniprot: bool = False,
        expand: bool = False
    ):
        """Find protein interaction domains and functions in amino acid sequences."""
        with start_action(action_type="gget_elm", sequence_length=len(sequence) if not uniprot else None):
            result = gget.elm(
                sequence=sequence,
                sensitivity=sensitivity,
                threads=threads,
                uniprot=uniprot,
                expand=expand
            )
            # ELM returns two dataframes: ortholog_df and regex_df
            if isinstance(result, tuple) and len(result) == 2:
                ortholog_df, regex_df = result
                data = {
                    "ortholog_df": ortholog_df.to_dict() if hasattr(ortholog_df, 'to_dict') else ortholog_df,
                    "regex_df": regex_df.to_dict() if hasattr(regex_df, 'to_dict') else regex_df
                }
            else:
                data = result
            
            return data

    async def cosmic_search(
        self, 
        searchterm: str,
        cosmic_tsv_path: Optional[str] = None,
        limit: int = 100
    ):
        """Search COSMIC database for cancer mutations and cancer-related data.
        
        Args:
            searchterm: Gene symbol or name to search for (e.g., 'PIK3CA', 'BRCA1')
            cosmic_tsv_path: Path to COSMIC TSV file (optional, uses default if None)
            limit: Maximum number of results to return
        
        Example:
            Input: searchterm='PIK3CA'
            Output: Mutation data including positions, amino acid changes, cancer types, etc.
            
        Note: This tool accepts gene symbols directly, no need for Ensembl ID conversion.
        """
        with start_action(action_type="gget_cosmic", searchterm=searchterm, limit=limit):
            result = gget.cosmic(
                searchterm=searchterm,
                cosmic_tsv_path=cosmic_tsv_path,
                limit=limit
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def mutate_sequences(
        self, 
        sequences: Union[str, List[str]],
        mutations: Union[str, List[str]],
        k: int = 30
    ):
        """Mutate nucleotide sequences based on specified mutations."""
        with start_action(action_type="gget_mutate", num_sequences=len(sequences) if isinstance(sequences, list) else 1):
            result = gget.mutate(
                sequences=sequences,
                mutations=mutations,
                k=k
            )
            return result

    async def opentargets_analysis(
        self, 
        ensembl_id: str,
        resource: str = "diseases",
        limit: Optional[int] = None
    ):
        """Explore diseases and drugs associated with a gene using Open Targets.
        
        PREREQUISITE: Use search_genes to get Ensembl ID first.
        
        Args:
            ensembl_id: Ensembl gene ID (e.g., 'ENSG00000141510' for APOE)
            resource: Type of information ('diseases', 'drugs', 'tractability', etc.)
            limit: Maximum number of results (optional)
        
        Example workflow:
            1. search_genes(['APOE']) → 'ENSG00000141510'
            2. opentargets_analysis('ENSG00000141510') → disease associations
        """
        with start_action(action_type="gget_opentargets", ensembl_id=ensembl_id, resource=resource):
            result = gget.opentargets(
                ensembl_id=ensembl_id,
                resource=resource,
                limit=limit
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def cellxgene_query(
        self, 
        gene: Optional[List[str]] = None,
        tissue: Optional[List[str]] = None,
        cell_type: Optional[List[str]] = None,
        species: str = "homo_sapiens"
    ):
        """Query single-cell RNA-seq data from CellxGene."""
        with start_action(action_type="gget_cellxgene", genes=gene, tissues=tissue):
            result = gget.cellxgene(
                gene=gene,
                tissue=tissue,
                cell_type=cell_type,
                species=species
            )
            return result

    async def setup_databases(
        self, 
        module: str
    ):
        """Setup databases for gget modules that require local data."""
        with start_action(action_type="gget_setup", module=module):
            # Valid modules that require setup
            valid_modules = ["elm", "cellxgene", "alphafold"]
            if module not in valid_modules:
                return {
                    "data": None,
                    "success": False,
                    "message": f"Invalid module '{module}'. Valid modules are: {', '.join(valid_modules)}"
                }
            
            result = gget.setup(module)
            return {
                "data": result,
                "success": True,
                "message": f"Setup completed for {module} module"
            }


def create_app():
    """Create and configure the FastMCP application."""
    return GgetMCP()

def cli_app():
    """CLI application entry point."""
    app = create_app()
    app.run(transport=DEFAULT_TRANSPORT, host=DEFAULT_HOST, port=DEFAULT_PORT)

def cli_app_stdio():
    """CLI application with stdio transport."""
    app = create_app()
    app.run(transport="stdio")

def cli_app_sse():
    """CLI application with SSE transport."""  
    app = create_app()
    app.run(transport="sse", host=DEFAULT_HOST, port=DEFAULT_PORT)

if __name__ == "__main__":
    cli_app_stdio() 