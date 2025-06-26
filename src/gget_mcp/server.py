#!/usr/bin/env python3
"""gget MCP Server - Bioinformatics query interface using the gget library."""

import os
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pathlib import Path

import typer
from typing_extensions import Annotated
from fastmcp import FastMCP

from .server_ext import GgetMCPExtended, SearchResult, SequenceResult, StructureResult, LocalFileResult

class TransportType(str, Enum):
    STDIO = "stdio"
    STDIO_LOCAL = "stdio-local"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"

# Configuration
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3002"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")

class GgetMCP(GgetMCPExtended):
    """Simplified gget MCP Server with essential bioinformatics tools."""
    
    def __init__(
        self, 
        name: str = "gget MCP Server",
        prefix: str = "gget_",
        transport_mode: str = "stdio",
        output_dir: Optional[str] = None,
        extended_mode: bool = False,
        **kwargs
    ):
        """Initialize the gget tools with FastMCP functionality."""
        self.extended_mode = extended_mode
        super().__init__(
            name=name, 
            prefix=prefix, 
            transport_mode=transport_mode, 
            output_dir=output_dir, 
            **kwargs
        )
    
    def _register_gget_tools(self):
        """Register gget tools - simplified by default, extended if requested."""
        
        if self.extended_mode:
            # Use the full extended versions from parent class
            super()._register_gget_tools()
        else:
            # Register simplified versions with essential parameters only
            
            # Gene information and search tools
            self.tool(name=f"{self.prefix}search")(self.search_simple)
            self.tool(name=f"{self.prefix}search_genes")(self.search_genes_simple)
            self.tool(name=f"{self.prefix}info")(self.get_gene_info_simple)
            
            # Sequence tools - use local wrapper if in local mode
            if self.transport_mode == "stdio-local":
                self.tool(name=f"{self.prefix}seq")(self.get_sequences_local_simple)
            else:
                self.tool(name=f"{self.prefix}seq")(self.get_sequences_simple)
            
            # Reference genome tools
            self.tool(name=f"{self.prefix}ref")(self.get_reference_simple)
            
            # Sequence analysis tools
            self.tool(name=f"{self.prefix}blast")(self.blast_sequence_simple)
            self.tool(name=f"{self.prefix}blat")(self.blat_sequence_simple)
            
            # Alignment tools - use local wrappers if in local mode
            if self.transport_mode == "stdio-local":
                self.tool(name=f"{self.prefix}muscle")(self.muscle_align_local_simple)
                self.tool(name=f"{self.prefix}diamond")(self.diamond_align_local_simple)
            else:
                self.tool(name=f"{self.prefix}muscle")(self.muscle_align_simple)
                self.tool(name=f"{self.prefix}diamond")(self.diamond_align_simple)
            
            # Expression and functional analysis
            self.tool(name=f"{self.prefix}archs4")(self.archs4_expression_simple)
            self.tool(name=f"{self.prefix}enrichr")(self.enrichr_analysis_simple)
            self.tool(name=f"{self.prefix}bgee")(self.bgee_orthologs_simple)
            
            # Protein structure and function - use local wrappers if in local mode
            if self.transport_mode == "stdio-local":
                self.tool(name=f"{self.prefix}pdb")(self.get_pdb_structure_local_simple)
                self.tool(name=f"{self.prefix}alphafold")(self.alphafold_predict_local_simple)
            else:
                self.tool(name=f"{self.prefix}pdb")(self.get_pdb_structure_simple)
                self.tool(name=f"{self.prefix}alphafold")(self.alphafold_predict_simple)
                
            self.tool(name=f"{self.prefix}elm")(self.elm_analysis_simple)
            
            # Cancer and mutation analysis
            self.tool(name=f"{self.prefix}cosmic")(self.cosmic_search_simple)
            self.tool(name=f"{self.prefix}mutate")(self.mutate_sequences_simple)
            
            # Drug and disease analysis
            self.tool(name=f"{self.prefix}opentargets")(self.opentargets_analysis_simple)
            
            # Single-cell analysis
            self.tool(name=f"{self.prefix}cellxgene")(self.cellxgene_query_simple)
            
            # Setup and utility functions
            self.tool(name=f"{self.prefix}setup")(self.setup_databases_simple)

    # Simplified method implementations with essential parameters only
    
    async def search_simple(
        self, 
        search_terms: Union[str, List[str]], 
        species: str = "homo_sapiens"
    ) -> SearchResult:
        """General search for any biological terms using gene symbols, names, or synonyms.
        
        This is a general search that looks broadly across gene names and descriptions.
        For specific gene symbol searches, use search_genes_simple instead.
        
        Args:
            search_terms: Search terms, names, or synonyms (e.g., 'cancer' or ['apoptosis', 'death'])
            species: Target species (e.g., 'homo_sapiens', 'mus_musculus')
        
        Returns:
            SearchResult: DataFrame with search results containing Ensembl IDs and descriptions
            
        Example:
            Input: search_terms='apoptosis', species='homo_sapiens'
            Output: DataFrame with genes related to apoptosis
        
        Note: Searches broadly in "gene name" and "description" sections of Ensembl database.
        Results are limited to prevent overwhelming LLM context.
        """
        # Calculate reasonable limit based on number of search terms
        # Keep total results small to avoid overwhelming LLM context (target: 2-4KB)
        if isinstance(search_terms, list):
            limit = min(15, len(search_terms) * 3)  # More generous for general search
        else:
            limit = 10  # Single term gets more results for general search
            
        return await super().search_genes(search_terms=search_terms, species=species, limit=limit)

    async def search_genes_simple(
        self, 
        search_terms: Union[str, List[str]], 
        species: str = "homo_sapiens",
        id_type: str = "gene"
    ) -> SearchResult:
        """Search for specific genes using gene symbols with enhanced search strategy.
        
        🚀 **BATCH PROCESSING SUPPORTED**: This function can process multiple genes in a single call!
        Use this tool FIRST when you have gene names/symbols and need to find their Ensembl IDs.
        Returns Ensembl IDs which are required for get_gene_info and get_sequences tools.
        
        IMPORTANT: Due to limitations in Ensembl search, short gene names often fail to find results.
        For best results, provide descriptive terms along with gene symbols:
        
        RECOMMENDED FORMAT: "GENE_SYMBOL descriptive_terms"
        Examples:
        - Instead of: "APP" 
        - Use: "APP amyloid precursor" or "APP amyloid beta precursor protein"
        - Instead of: ["BACE1", "MAPT"]
        - Use: ["BACE1 beta secretase", "MAPT microtubule tau"]
        
        This function uses AND search for multi-word terms and OR search for single words.
        
        Args:
            search_terms: SINGLE gene symbol OR LIST of gene symbols with descriptive terms
                         Single: 'APP amyloid precursor'
                         Batch: ['BACE1 beta secretase', 'MAPT tau', 'APOE apolipoprotein']
            species: Target species (e.g., 'homo_sapiens', 'mus_musculus')
            id_type: "gene" (default) or "transcript" - whether to return genes or transcripts
        
        Returns:
            SearchResult: DataFrame with gene search results containing Ensembl IDs and descriptions
                         Results from ALL search terms are combined in a single response
            
        Example (SINGLE GENE):
            Input: search_terms='APP amyloid precursor', species='homo_sapiens'
            Output: DataFrame with APP gene and related genes
            
        Example (BATCH PROCESSING, limit number of queries to 3-5 to avoid timeouts):
            Input: search_terms=['APOE apolipoprotein', 'APP amyloid', 'PSEN1 presenilin'], species='homo_sapiens'
            Output: DataFrame with ALL three genes and their Ensembl IDs in one response
        
        Downstream tools that need the Ensembl IDs from this search:
            - get_gene_info: Get detailed gene information  
            - get_sequences: Get DNA/protein sequences
        
        Note: For general biological term searches without gene focus, use search_simple.
        """
        import re
        
        # Convert to list if single string
        terms_list = search_terms if isinstance(search_terms, list) else [search_terms]
        
        all_results = {}
        
        # Process each search term individually with ENSG enhancement
        for search_term in terms_list:
            try:
                # Split search term into words for AND search
                search_words = search_term.strip().split()
                
                print(f"Searching for: {search_words}")
                
                # Use AND mode for multi-word terms, OR for single words
                search_mode = "and" if len(search_words) > 1 else "or"
                search_limit = 10 if search_mode == "and" else 20
                
                raw_result = await super().search_genes(
                    search_terms=search_words, 
                    species=species, 
                    id_type=id_type,
                    andor=search_mode,
                    limit=search_limit
                )
                
                if isinstance(raw_result, dict) and 'gene_name' in raw_result:
                    gene_names = raw_result['gene_name']
                    ensembl_ids = raw_result['ensembl_id']
                    descriptions = raw_result.get('ensembl_description', {})
                    
                    # Take top 5 results from AND search (they should be relevant due to ENSG filter)
                    selected_indices = list(gene_names.keys())[:5]
                    
                    # Add results to combined results
                    for idx in selected_indices:
                        result_idx = len(all_results.get('gene_name', {}))
                        for key in raw_result.keys():
                            if key not in all_results:
                                all_results[key] = {}
                            all_results[key][result_idx] = raw_result[key][idx]
                            
            except Exception as e:
                print(f"Warning: Failed to search for term '{search_term}': {e}")
                continue
        

        # If no results found, fall back to original method
        if not all_results:
            print(f"Smart search found no results, falling back to original search...")
            limit = min(10, len(terms_list) * 2)
            return await super().search_genes(
                search_terms=search_terms, 
                species=species, 
                id_type=id_type,
                limit=limit
            )
        
        return all_results

    async def get_gene_info_simple(
        self, 
        ensembl_ids: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """Get detailed gene and transcript metadata using Ensembl IDs.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: One or more Ensembl gene IDs (e.g., 'ENSG00000141510' or ['ENSG00000141510'])
                        Also supports WormBase and FlyBase IDs
            
        Returns:
            Dict[str, Any]: DataFrame with gene information containing metadata from multiple databases
        
        Example workflow:
            1. search_genes('TP53', 'homo_sapiens') → get Ensembl ID 'ENSG00000141510'
            2. get_gene_info('ENSG00000141510') 
            
        Example output:
            DataFrame with columns like 'ensembl_id', 'symbol', 'biotype', 'chromosome', 'start', 'end', 
            plus NCBI, UniProt, and optionally PDB information
        """
        return await super().get_gene_info(ensembl_ids=ensembl_ids)

    async def get_sequences_simple(
        self, 
        ensembl_ids: Union[str, List[str]],
        translate: bool = False
    ) -> SequenceResult:
        """Fetch nucleotide or amino acid sequence (FASTA) of genes or transcripts.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: One or more Ensembl gene IDs (e.g., 'ENSG00000141510' or ['ENSG00000141510'])
                        Also supports WormBase and FlyBase IDs
            translate: If True, returns amino acid sequences; if False, returns nucleotide sequences
            
        Returns:
            SequenceResult: List containing the requested sequences in FASTA format
        
        Example workflow for protein sequence:
            1. search_genes('TP53 protein', 'homo_sapiens') → 'ENSG00000141510'
            2. get_sequences('ENSG00000141510', translate=True)
            
        Example output:
            List of sequences in FASTA format: ['>ENSG00000141510', 'MEEPQSDPSVEPPLSQ...']
        
        Downstream tools that use protein sequences:
            - alphafold_predict: Predict 3D structure from protein sequence
            - blast_sequence: Search for similar sequences
        """
        return await super().get_sequences(ensembl_ids=ensembl_ids, translate=translate)

    async def get_reference_simple(
        self, 
        species: str = "homo_sapiens",
        which: Union[str, List[str]] = "all"
    ) -> Union[Dict[str, Any], List[str]]:
        """Fetch FTPs for reference genomes and annotations by species from Ensembl.
        
        Args:
            species: Species in format "genus_species" (e.g., "homo_sapiens"). 
                    Shortcuts supported: "human", "mouse", "human_grch37"
            which: Which results to return. Options: 'gtf', 'cdna', 'dna', 'cds', 'cdrna', 'pep', 'all'
        
        Returns:
            Union[Dict[str, Any], List[str]]: Dictionary with URLs, versions, and metadata
            
        Example:
            Input: species="homo_sapiens", which="gtf"
            Output: Dictionary containing GTF URLs with Ensembl version and release info
        """
        return await super().get_reference(species=species, which=which)

    async def blast_sequence_simple(
        self, 
        sequence: str,
        program: str = "default",
        database: str = "default"
    ) -> Dict[str, Any]:
        """BLAST a nucleotide or amino acid sequence against any BLAST database.
        
        Args:
            sequence: Nucleotide or amino acid sequence (string) or path to FASTA file
            program: BLAST program - 'blastn', 'blastp', 'blastx', 'tblastn', 'tblastx', or 'default' (auto-detect)
            database: BLAST database - 'nt', 'nr', 'refseq_rna', 'refseq_protein', 'swissprot', or 'default' (auto-detect)
        
        Returns:
            Dict[str, Any]: DataFrame with BLAST results including alignment details and scores
            
        Example:
            Input: sequence="ATGCGATCGTAGC", program="blastn", database="nt"
            Output: DataFrame with BLAST hits, E-values, scores, and alignments
        
        Note: NCBI server rule: Run scripts weekends or 9pm-5am ET weekdays for >50 searches
        Results are limited to 10 hits to prevent overwhelming LLM context - use extended blast_sequence for more results.
        """
        return await super().blast_sequence(sequence=sequence, program=program, database=database, limit=10)

    async def blat_sequence_simple(
        self, 
        sequence: str,
        assembly: str = "human"
    ) -> Dict[str, Any]:
        """BLAT a nucleotide or amino acid sequence against any BLAT UCSC assembly.
        
        Args:
            sequence: Nucleotide or amino acid sequence (string) or path to FASTA file containing one sequence
            assembly: Genome assembly - 'human' (hg38), 'mouse' (mm39), 'zebrafinch' (taeGut2)
        
        Returns:
            Dict[str, Any]: DataFrame with BLAT results including genomic coordinates and alignment details
            
        Example:
            Input: sequence="ATGCGATCGTAGC", assembly="human"
            Output: DataFrame with chromosome, start, end positions and alignment scores
        """
        return await super().blat_sequence(sequence=sequence, assembly=assembly)

    async def muscle_align_simple(
        self, 
        sequences: Union[List[str], str]
    ) -> Optional[str]:
        """Align multiple nucleotide or amino acid sequences using MUSCLE v5 algorithm.
        
        Args:
            sequences: List of sequences or path to FASTA file containing sequences to be aligned
        
        Returns:
            Optional[str]: Alignment results in aligned FASTA (.afa) format
            
        Example:
            Input: sequences=["ATGCGATC", "ATGCGTTC", "ATGCGATG"]
            Output: Aligned sequences in FASTA format
        """
        return await super().muscle_align(sequences=sequences)

    async def diamond_align_simple(
        self, 
        sequences: Union[str, List[str]],
        reference: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """Align multiple protein or translated DNA sequences using DIAMOND.
        
        Args:
            sequences: Query sequences (string, list) or path to FASTA file with sequences to align against reference
            reference: Reference sequences (string, list) or path to FASTA file with reference sequences
        
        Returns:
            Dict[str, Any]: DataFrame with DIAMOND alignment results including similarity scores and positions
            
        Example:
            Input: sequences=["MKVLWA"], reference=["MKVLWAICAV"]
            Output: DataFrame with alignment scores, positions, and match details
        """
        return await super().diamond_align(sequences=sequences, reference=reference)

    async def archs4_expression_simple(
        self, 
        gene: str,
        which: str = "correlation",
        species: str = "human"
    ) -> Dict[str, Any]:
        """Find correlated genes or tissue expression atlas using ARCHS4 RNA-seq database.
        
        Args:
            gene: Gene symbol (e.g., 'STAT4') or Ensembl ID if ensembl=True (e.g., 'ENSG00000138378')
            which: Analysis type - 'correlation' (most correlated genes) or 'tissue' (tissue expression atlas)
            species: Target species - 'human' or 'mouse' (only for tissue expression atlas)
        
        Returns:
            Dict[str, Any]: DataFrame with correlation table or tissue expression atlas
            
        Example (correlation):
            Input: gene="STAT4", which="correlation"
            Output: DataFrame with 20 most correlated genes and Pearson correlation coefficients
            
        Example (tissue):
            Input: gene="STAT4", which="tissue", species="human"  
            Output: DataFrame with tissue expression levels across human samples
            
        Results are limited to 20 correlated genes to prevent overwhelming LLM context - use extended archs4_expression for more results.
        """
        return await super().archs4_expression(gene=gene, which=which, species=species, gene_count=20)

    async def enrichr_analysis_simple(
        self, 
        genes: List[str],
        database: str = "pathway",
        species: str = "human"
    ) -> Dict[str, Any]:
        """Perform functional enrichment analysis on gene list using Enrichr.
        
        Args:
            genes: List of gene symbols (e.g., ['PHF14', 'RBM3']) or Ensembl IDs if ensembl=True
            database: Reference database shortcuts: 'pathway' (KEGG), 'transcription' (ChEA), 'ontology' (GO), 
                     'diseases_drugs' (GWAS), 'celltypes' (PanglaoDB), 'kinase_interactions' (KEA)
            species: Species database - 'human', 'mouse', 'fly', 'yeast', 'worm', 'fish'
        
        Returns:
            Dict[str, Any]: DataFrame with enrichment results including pathways, p-values, and statistical measures
            
        Example:
            Input: genes=['PHF14', 'RBM3', 'MSL1'], database='pathway'  
            Output: DataFrame with KEGG pathway enrichment results and statistics
        """
        # Map shortcuts to full database names
        database_map = {
            'pathway': 'KEGG_2021_Human',
            'transcription': 'ChEA_2016',
            'ontology': 'GO_Biological_Process_2021',
            'diseases_drugs': 'GWAS_Catalog_2019',
            'celltypes': 'PanglaoDB_Augmented_2021',
            'kinase_interactions': 'KEA_2015'
        }
        full_database = database_map.get(database, database)
        return await super().enrichr_analysis(genes=genes, database=full_database, species=species)

    async def bgee_orthologs_simple(
        self, 
        gene_id: str,
        type: str = "orthologs"
    ) -> Dict[str, Any]:
        """Get orthologs or expression data for a gene from Bgee database.
        
        PREREQUISITE: Use search_genes to get Ensembl ID first.
        
        Args:
            gene_id: Ensembl gene ID (e.g., 'ENSG00000012048' for BRCA1)
            type: Type of data to retrieve - 'orthologs' or 'expression'
            
        Returns:
            Dict[str, Any]: DataFrame with ortholog information across species or expression data from Bgee
        
        Example workflow:
            1. search_genes('BRCA1') → 'ENSG00000012048' 
            2. bgee_orthologs('ENSG00000012048') → ortholog data across species
        """
        return await super().bgee_orthologs(gene_id=gene_id, type=type)

    async def get_pdb_structure_simple(
        self, 
        pdb_id: str,
        resource: str = "pdb"
    ) -> StructureResult:
        """Query RCSB PDB for protein structure/metadata of a given PDB ID.
        
        IMPORTANT: This tool requires a specific PDB ID (e.g., '7S7U'), NOT gene names.
        
        Args:
            pdb_id: PDB ID to query (e.g., '7S7U', '2GS6')
            resource: Type of information - 'pdb' (structure), 'entry' (metadata), 'pubmed', 'assembly'
            
        Returns:
            StructureResult: JSON format (except resource='pdb' returns PDB format structure)
        
        Example:
            Input: pdb_id='7S7U', resource='pdb'
            Output: Protein structure in PDB format
            
        Alternative workflow for gene structure prediction:
            1. search_genes('EGFR') → get Ensembl ID
            2. get_sequences(ensembl_id, translate=True) → get protein sequence
            3. alphafold_predict(protein_sequence) → predict structure
        """
        return await super().get_pdb_structure(pdb_id=pdb_id, resource=resource)

    async def alphafold_predict_simple(
        self, 
        sequence: Union[str, List[str]]
    ) -> StructureResult:
        """Predict protein structure using simplified AlphaFold v2.3.0 algorithm.
        
        PREREQUISITE: Use get_sequences with translate=True to get protein sequence first.
        
        Args:
            sequence: Amino acid sequence (string), list of sequences, or path to FASTA file
            
        Returns:
            StructureResult: AlphaFold structure prediction - saves aligned error (JSON) and prediction (PDB) files
        
        Example full workflow:
            1. search_genes('TP53') → 'ENSG00000141510'
            2. get_sequences('ENSG00000141510', translate=True) → 'MEEPQSDPSVEPPLSQ...'
            3. alphafold_predict('MEEPQSDPSVEPPLSQ...')
            
        Note: This uses simplified AlphaFold without templates and limited MSA database.
        Please cite gget and AlphaFold papers when using this function.
        """
        return await super().alphafold_predict(sequence=sequence)

    async def elm_analysis_simple(
        self, 
        sequence: str,
        uniprot: bool = False
    ) -> Dict[str, Any]:
        """Locally predict Eukaryotic Linear Motifs from amino acid sequence or UniProt ID.
        
        Args:
            sequence: Amino acid sequence or UniProt accession (if uniprot=True)
            uniprot: If True, input is UniProt accession instead of amino acid sequence
        
        Returns:
            Dict[str, Any]: Two dataframes - ortholog motifs and regex motifs with domain predictions
                           
        Example:
            Input: sequence="MKVLWAICAVL", uniprot=False
            Output: {'ortholog_df': {...}, 'regex_df': {...}} with motif predictions
            
        Example (UniProt):
            Input: sequence="P04637", uniprot=True  
            Output: Motif analysis results for UniProt entry P04637
            
        Note: ELM data is for non-commercial use only (ELM Software License Agreement).
        """
        return await super().elm_analysis(sequence=sequence, uniprot=uniprot)

    async def cosmic_search_simple(
        self, 
        searchterm: str,
        cosmic_tsv_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search COSMIC database for cancer mutations or download COSMIC databases.
        
        Args:
            searchterm: Gene symbol or name to search for (e.g., 'PIK3CA', 'BRCA1')
            cosmic_tsv_path: Path to COSMIC TSV file (optional, uses default if None)
            
        Returns:
            Dict[str, Any]: Mutation data including positions, amino acid changes, cancer types
        
        Example:
            Input: searchterm='PIK3CA'
            Output: Mutation data including positions, amino acid changes, cancer types
            
        Note: This tool accepts gene symbols directly, no need for Ensembl ID conversion.
        Results are limited to 25 mutations to prevent overwhelming LLM context - use extended cosmic_search for more results.
        """
        return await super().cosmic_search(searchterm=searchterm, cosmic_tsv_path=cosmic_tsv_path, limit=25)

    async def mutate_sequences_simple(
        self, 
        sequences: Union[str, List[str]],
        mutations: Union[str, List[str]]
    ) -> Union[Dict[str, Any], List[str]]:
        """Mutate nucleotide sequences according to provided mutations in standard annotation.
        
        Args:
            sequences: Path to FASTA file or sequences as string/list (e.g., 'AGCTAGCT' or ['ACTG', 'ATCG'])
            mutations: Path to CSV/TSV file or mutations as string/list (e.g., 'c.2C>T' or ['c.2C>T', 'c.1A>C'])
                      Standard mutation annotation format required
        
        Returns:
            Union[Dict[str, Any], List[str]]: List of mutated sequences or updated DataFrame
        
        Example (simple):
            Input: sequences=['ATGCGATC'], mutations=['c.2T>G']
            Output: List of mutated sequences with flanking regions
            
        Note: Sequence IDs in FASTA must match seq_ID column in mutations file.
        Supports complex mutations: substitutions (c.2C>T), insertions, deletions, inversions.
        """
        return await super().mutate_sequences(sequences=sequences, mutations=mutations)

    async def opentargets_analysis_simple(
        self, 
        ensembl_id: str,
        resource: str = "diseases"
    ) -> Dict[str, Any]:
        """Query OpenTargets for diseases, drugs, and other data associated with a gene.
        
        PREREQUISITE: Use search_genes to get Ensembl ID first.
        
        Args:
            ensembl_id: Ensembl gene ID (e.g., 'ENSG00000169194')
            resource: Type of information - 'diseases', 'drugs', 'tractability', 'pharmacogenetics', 
                     'expression', 'depmap', 'interactions'
            
        Returns:
            Dict[str, Any]: DataFrame with disease/drug associations, clinical evidence, and experimental data
        
        Example workflow:
            1. search_genes('APOE') → 'ENSG00000141510'
            2. opentargets_analysis('ENSG00000141510') → disease associations
            
        Results are limited to 20 associations to prevent overwhelming LLM context - use extended opentargets_analysis for more results.
        """
        return await super().opentargets_analysis(ensembl_id=ensembl_id, resource=resource, limit=20)

    async def cellxgene_query_simple(
        self, 
        gene: Optional[Union[str, List[str]]] = None,
        tissue: Optional[Union[str, List[str]]] = None,
        cell_type: Optional[Union[str, List[str]]] = None,
        species: str = "homo_sapiens"
    ) -> Dict[str, Any]:
        """Query single-cell RNA-seq data from CZ CELLxGENE Discover using Census.
        
        NOTE: Querying large datasets requires >16 GB RAM and >5 Mbps internet connection.
        
        Args:
            gene: Gene name(s) or Ensembl ID(s) (e.g., ['ACE2', 'SLC5A1'])
            tissue: Tissue(s) to query (e.g., ['lung', 'blood'])
            cell_type: Cell type(s) to query (e.g., ['mucus secreting cell'])
            species: Target species - 'homo_sapiens' or 'mus_musculus'
        
        Returns:
            Dict[str, Any]: Metadata DataFrame only (to prevent overwhelming LLM context)
        
        Example:
            Input: gene=['ACE2'], tissue=['lung'], cell_type=['alveolar epithelial cell']
            Output: Metadata about single-cell datasets containing ACE2 in lung alveolar epithelial cells
            
        Note: Returns metadata only to keep response size manageable - use extended cellxgene_query for full expression data.
        """
        return await super().cellxgene_query(gene=gene, tissue=tissue, cell_type=cell_type, species=species, meta_only=True)

    async def setup_databases_simple(
        self, 
        module: str
    ) -> Dict[str, Any]:
        """Install third-party dependencies for specified gget modules.
        
        Args:
            module: gget module to install dependencies for - 'alphafold', 'cellxgene', 'elm', 'gpt', or 'cbio'
        
        Returns:
            Dict[str, Any]: Setup status with success indicator and messages
            
        Example:
            Input: module='elm'
            Output: Downloads and installs ELM dependencies for motif analysis
            
        Note: Available modules requiring setup: 'alphafold', 'cellxgene', 'elm', 'gpt', 'cbio'
        """
        return await super().setup_databases(module=module)

    # Local mode wrapper functions for large data
    async def get_sequences_local_simple(
        self, 
        ensembl_ids: Union[str, List[str]],
        translate: bool = False,
        output_path: Optional[str] = None,
        format: Literal["fasta"] = "fasta"
    ) -> LocalFileResult:
        """Fetch sequences and save to local file in stdio-local mode.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: One or more Ensembl gene IDs (e.g., 'ENSG00000141510' or ['ENSG00000141510'])
            translate: If True, returns amino acid sequences; if False, returns nucleotide sequences
            output_path: ABSOLUTE path to output file (e.g., '/home/user/sequences.fasta'). 
                        AVOID relative paths as they cause file location issues. Auto-generated if not provided.
            format: Output format (currently supports 'fasta')
        
        Returns:
            LocalFileResult: Contains ABSOLUTE path, format, and success information instead of sequence data
        """
        return await super().get_sequences_local(
            ensembl_ids=ensembl_ids, 
            translate=translate, 
            output_path=output_path, 
            format=format
        )

    async def get_pdb_structure_local_simple(
        self, 
        pdb_id: str,
        resource: str = "pdb",
        output_path: Optional[str] = None,
        format: Literal["pdb"] = "pdb"
    ) -> LocalFileResult:
        """Fetch PDB structure and save to local file in stdio-local mode.
        
        Args:
            pdb_id: PDB ID to query (e.g., '7S7U', '2GS6')
            resource: Type of information - 'pdb' (structure), 'entry', 'pubmed', 'assembly'
            output_path: ABSOLUTE path to output file (e.g., '/home/user/structure.pdb'). 
                        AVOID relative paths as they cause file location issues. Auto-generated if not provided.
            format: Output format (currently supports 'pdb')
        
        Returns:
            LocalFileResult: Contains ABSOLUTE path, format, and success information instead of structure data
        """
        return await super().get_pdb_structure_local(
            pdb_id=pdb_id, 
            resource=resource, 
            output_path=output_path, 
            format=format
        )

    async def alphafold_predict_local_simple(
        self, 
        sequence: Union[str, List[str]],
        output_path: Optional[str] = None,
        format: Literal["pdb"] = "pdb"
    ) -> LocalFileResult:
        """Predict protein structure using AlphaFold and save to local file.
        
        Args:
            sequence: Amino acid sequence (string), list of sequences, or ABSOLUTE path to FASTA file
            output_path: ABSOLUTE path to output file (e.g., '/home/user/prediction.pdb'). 
                        AVOID relative paths as they cause file location issues. Auto-generated if not provided.
            format: Output format (currently supports 'pdb')
        
        Returns:
            LocalFileResult: Contains ABSOLUTE path, format, and success information instead of structure data
        """
        return await super().alphafold_predict_local(
            sequence=sequence, 
            output_path=output_path, 
            format=format
        )

    async def muscle_align_local_simple(
        self, 
        sequences: Union[List[str], str],
        output_path: Optional[str] = None,
        format: Literal["fasta", "afa"] = "fasta"
    ) -> LocalFileResult:
        """Align sequences using MUSCLE and save to local file.
        
        Args:
            sequences: List of sequences or ABSOLUTE path to FASTA file containing sequences to be aligned
            output_path: ABSOLUTE path to output file (e.g., '/home/user/alignment.fasta'). 
                        AVOID relative paths as they cause file location issues. Auto-generated if not provided.
            format: Output format ('fasta' for FASTA format, 'afa' for aligned FASTA format)
        
        Returns:
            LocalFileResult: Contains ABSOLUTE path, format, and success information instead of alignment data
        """
        return await super().muscle_align_local(
            sequences=sequences, 
            output_path=output_path, 
            format=format
        )

    async def diamond_align_local_simple(
        self, 
        sequences: Union[str, List[str]],
        reference: Union[str, List[str]],
        output_path: Optional[str] = None,
        format: Literal["json", "tsv"] = "json"
    ) -> LocalFileResult:
        """Align sequences using DIAMOND and save to local file.
        
        Args:
            sequences: Query sequences (string, list) or ABSOLUTE path to FASTA file with sequences to align against reference
            reference: Reference sequences (string, list) or ABSOLUTE path to FASTA file with reference sequences
            output_path: ABSOLUTE path to output file (e.g., '/home/user/alignment.json'). 
                        AVOID relative paths as they cause file location issues. Auto-generated if not provided.
            format: Output format ('json' recommended, 'tsv' also supported)
        
        Returns:
            LocalFileResult: Contains ABSOLUTE path, format, and success information instead of alignment data
        """
        return await super().diamond_align_local(
            sequences=sequences, 
            reference=reference, 
            output_path=output_path, 
            format=format
        )


def create_app(transport_mode: str = "stdio", output_dir: Optional[str] = None, extended_mode: bool = False):
    """Create and configure the FastMCP application."""
    return GgetMCP(transport_mode=transport_mode, output_dir=output_dir, extended_mode=extended_mode)

# CLI application setup
cli_app = typer.Typer(help="gget MCP Server CLI")

@cli_app.command()
def server(
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = DEFAULT_PORT,
    transport: Annotated[str, typer.Option(help="Transport type: stdio, stdio-local, streamable-http, or sse")] = DEFAULT_TRANSPORT,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory for local files (stdio-local mode)")] = None,
    extended: Annotated[bool, typer.Option(help="Use extended mode with all parameters (fallback to full API)")] = False
):
    """Runs the gget MCP server."""
    # Validate transport value
    if transport not in ["stdio", "stdio-local", "streamable-http", "sse"]:
        typer.echo(f"Invalid transport: {transport}. Must be one of: stdio, stdio-local, streamable-http, sse")
        raise typer.Exit(1)
        
    app = create_app(transport_mode=transport, output_dir=output_dir, extended_mode=extended)

    # Different transports need different arguments
    if transport in ["stdio", "stdio-local"]:
        app.run(transport="stdio")  # Both stdio modes use stdio transport
    else:
        app.run(transport=transport, host=host, port=port)

@cli_app.command(name="server-stdio")
def server_stdio(
    extended: Annotated[bool, typer.Option(help="Use extended mode with all parameters (fallback to full API)")] = False
):
    """Runs the gget MCP server in stdio mode (standard input/output)."""
    app = create_app(transport_mode="stdio", extended_mode=extended)
    app.run(transport="stdio")

@cli_app.command(name="server-local")
def server_local(
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory for local files")] = None,
    extended: Annotated[bool, typer.Option(help="Use extended mode with all parameters (fallback to full API)")] = False
):
    """Runs the gget MCP server in stdio-local mode (saves large files locally)."""
    app = create_app(transport_mode="stdio-local", output_dir=output_dir, extended_mode=extended)
    app.run(transport="stdio")

@cli_app.command(name="server-http")
def server_http(
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = DEFAULT_PORT,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory for local files")] = None,
    extended: Annotated[bool, typer.Option(help="Use extended mode with all parameters (fallback to full API)")] = False
):
    """Runs the gget MCP server in streamable HTTP mode."""
    app = create_app(transport_mode="streamable-http", output_dir=output_dir, extended_mode=extended)
    app.run(transport="streamable-http", host=host, port=port)

@cli_app.command(name="server-sse")
def server_sse(
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = DEFAULT_PORT,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory for local files")] = None,
    extended: Annotated[bool, typer.Option(help="Use extended mode with all parameters (fallback to full API)")] = False
):
    """Runs the gget MCP server in Server-Sent Events (SSE) mode."""
    app = create_app(transport_mode="sse", output_dir=output_dir, extended_mode=extended)
    app.run(transport="sse", host=host, port=port)

if __name__ == "__main__":
    cli_app() 