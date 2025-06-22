#!/usr/bin/env python3
"""gget MCP Server - Bioinformatics query interface using the gget library."""

import os
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pathlib import Path
import uuid
import json

import typer
from typing_extensions import Annotated
from fastmcp import FastMCP
from eliot import start_action
import gget

class TransportType(str, Enum):
    STDIO = "stdio"
    STDIO_LOCAL = "stdio-local"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"

# Configuration
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3002"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")  # Changed default to stdio

# Typehints for common return patterns discovered in battle tests
SequenceResult = Union[Dict[str, str], List[str], str]
StructureResult = Union[Dict[str, Any], str]
SearchResult = Dict[str, Any]
LocalFileResult = Dict[Literal["path", "format", "success", "error"], Any]

class GgetMCP(FastMCP):
    """gget MCP Server with bioinformatics tools."""
    
    def __init__(
        self, 
        name: str = "gget MCP Server",
        prefix: str = "gget_",
        transport_mode: str = "stdio",
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize the gget tools with FastMCP functionality."""
        super().__init__(name=name, **kwargs)
        
        self.prefix = prefix
        self.transport_mode = transport_mode
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "gget_output"
        
        # Create output directory if in local mode
        if self.transport_mode == "stdio-local":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self._register_gget_tools()
    
    def _save_to_local_file(
        self, 
        data: Any, 
        format_type: str, 
        base_name: Optional[str] = None
    ) -> LocalFileResult:
        """Helper function to save data to local files.
        
        Args:
            data: The data to save
            format_type: File format ('fasta', 'afa', 'pdb', 'json', etc.)
            base_name: Base name for the file (optional, will generate UUID if not provided)
            
        Returns:
            LocalFileResult: Contains path, format, success status, and optional error information
        """
        if base_name is None:
            base_name = str(uuid.uuid4())
            
        # Map format types to file extensions
        format_extensions = {
            'fasta': '.fasta',
            'afa': '.afa',
            'pdb': '.pdb',
            'json': '.json',
            'txt': '.txt',
            'tsv': '.tsv'
        }
        
        extension = format_extensions.get(format_type, '.txt')
        file_path = self.output_dir / f"{base_name}{extension}"
        
        try:
            if format_type in ['fasta', 'afa']:
                self._write_fasta_file(data, file_path)
            elif format_type == 'pdb':
                self._write_pdb_file(data, file_path)
            elif format_type == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                # Default to text format
                with open(file_path, 'w') as f:
                    if isinstance(data, dict):
                        json.dump(data, f, indent=2, default=str)
                    else:
                        f.write(str(data))
                        
            return {
                "path": str(file_path),
                "format": format_type,
                "success": True
            }
        except Exception as e:
            return {
                "path": None,
                "format": format_type,
                "success": False,
                "error": str(e)
            }
    
    def _write_fasta_file(self, data: Any, file_path: Path) -> None:
        """Write sequence data in FASTA format.
        
        Handles multiple data formats discovered in battle tests:
        - Dict[str, str]: sequence_id -> sequence
        - List[str]: [header, sequence, header, sequence, ...]
        - str: raw data
        """
        with open(file_path, 'w') as f:
            if isinstance(data, dict):
                for seq_id, sequence in data.items():
                    f.write(f">{seq_id}\n")
                    # Write sequence with line breaks every 80 characters
                    for i in range(0, len(sequence), 80):
                        f.write(f"{sequence[i:i+80]}\n")
            elif isinstance(data, list):
                # Handle FASTA list format from gget.seq
                for i in range(0, len(data), 2):
                    if i + 1 < len(data):
                        header = data[i] if data[i].startswith('>') else f">{data[i]}"
                        sequence = data[i + 1]
                        f.write(f"{header}\n")
                        # Write sequence with line breaks every 80 characters
                        for j in range(0, len(sequence), 80):
                            f.write(f"{sequence[j:j+80]}\n")
            elif data is None:
                # For MUSCLE alignments, gget.muscle() returns None but prints to stdout
                # We need to capture the stdout or use a different approach
                f.write("# MUSCLE alignment completed\n# Output was printed to console\n")
            else:
                f.write(str(data))
    
    def _write_pdb_file(self, data: Any, file_path: Path) -> None:
        """Write PDB structure data."""
        with open(file_path, 'w') as f:
            if isinstance(data, str):
                f.write(data)
            else:
                # Convert data to string representation
                f.write(str(data))
    
    def _register_gget_tools(self):
        """Register gget-specific tools."""
        
        # Gene information and search tools
        self.tool(name=f"{self.prefix}search")(self.search_genes)
        self.tool(name=f"{self.prefix}info")(self.get_gene_info)
        
        # Sequence tools - use local wrapper if in local mode
        if self.transport_mode == "stdio-local":
            self.tool(name=f"{self.prefix}seq")(self.get_sequences_local)
        else:
            self.tool(name=f"{self.prefix}seq")(self.get_sequences)
        
        # Reference genome tools
        self.tool(name=f"{self.prefix}ref")(self.get_reference)
        
        # Sequence analysis tools
        self.tool(name=f"{self.prefix}blast")(self.blast_sequence)
        self.tool(name=f"{self.prefix}blat")(self.blat_sequence)
        
        # Alignment tools - use local wrappers if in local mode
        if self.transport_mode == "stdio-local":
            self.tool(name=f"{self.prefix}muscle")(self.muscle_align_local)
            self.tool(name=f"{self.prefix}diamond")(self.diamond_align_local)
        else:
            self.tool(name=f"{self.prefix}muscle")(self.muscle_align)
            self.tool(name=f"{self.prefix}diamond")(self.diamond_align)
        
        # Expression and functional analysis
        self.tool(name=f"{self.prefix}archs4")(self.archs4_expression)
        self.tool(name=f"{self.prefix}enrichr")(self.enrichr_analysis)
        self.tool(name=f"{self.prefix}bgee")(self.bgee_orthologs)
        
        # Protein structure and function - use local wrappers if in local mode
        if self.transport_mode == "stdio-local":
            self.tool(name=f"{self.prefix}pdb")(self.get_pdb_structure_local)
            self.tool(name=f"{self.prefix}alphafold")(self.alphafold_predict_local)
        else:
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
        search_terms: Union[str, List[str]], 
        species: str = "homo_sapiens",
        release: Optional[int] = None,
        id_type: str = "gene",
        andor: str = "or",
        limit: Optional[int] = None
    ) -> SearchResult:
        """Search for genes using gene symbols, names, or synonyms.
        
        Use this tool FIRST when you have gene names/symbols and need to find their Ensembl IDs.
        Returns Ensembl IDs which are required for get_gene_info and get_sequences tools.
        
        Args:
            search_terms: Gene symbols, names, or synonyms as string or list of strings (e.g., 'TP53' or ['TP53', 'BRCA1'])
            species: Target species (e.g., 'homo_sapiens', 'mus_musculus') or specific core database name
            release: Ensembl release number (e.g., 104). Default: None (latest release)
            id_type: "gene" (default) or "transcript" - defines whether genes or transcripts are returned
            andor: "or" (default) or "and" - "or" returns genes with ANY searchword, "and" requires ALL searchwords
            limit: Maximum number of search results returned. Default: None (no limit)
        
        Returns:
            SearchResult: DataFrame with gene search results containing Ensembl IDs and descriptions
            
        Example:
            Input: search_terms='BRCA1', species='homo_sapiens'
            Output: DataFrame with columns like 'ensembl_id', 'gene_name', 'description'
        
        Downstream tools that need the Ensembl IDs from this search:
            - get_gene_info: Get detailed gene information  
            - get_sequences: Get DNA/protein sequences
        
        Note: Only searches in "gene name" and "description" sections of Ensembl database.
        """
        with start_action(action_type="gget_search", search_terms=search_terms, species=species):
            result = gget.search(
                searchwords=search_terms, 
                species=species, 
                release=release,
                id_type=id_type,
                andor=andor,
                limit=limit
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_gene_info(
        self, 
        ensembl_ids: Union[str, List[str]],
        ncbi: bool = True,
        uniprot: bool = True,
        pdb: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Get detailed gene and transcript metadata using Ensembl IDs.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: One or more Ensembl gene IDs as string or list (e.g., 'ENSG00000141510' or ['ENSG00000141510'])
                        Also supports WormBase and FlyBase IDs
            ncbi: If True, includes data from NCBI. Default: True
            uniprot: If True, includes data from UniProt. Default: True  
            pdb: If True, also returns PDB IDs (might increase runtime). Default: False
            verbose: If True, prints progress information. Default: True
            
        Returns:
            Dict[str, Any]: DataFrame with gene information containing metadata from multiple databases
        
        Example workflow:
            1. search_genes('TP53', 'homo_sapiens') → get Ensembl ID 'ENSG00000141510'
            2. get_gene_info('ENSG00000141510') 
            
        Example output:
            DataFrame with columns like 'ensembl_id', 'symbol', 'biotype', 'chromosome', 'start', 'end', 
            plus NCBI, UniProt, and optionally PDB information
        """
        with start_action(action_type="gget_info", ensembl_ids=ensembl_ids):
            result = gget.info(
                ens_ids=ensembl_ids, 
                ncbi=ncbi,
                uniprot=uniprot,
                pdb=pdb,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_sequences(
        self, 
        ensembl_ids: Union[str, List[str]],
        translate: bool = False,
        isoforms: bool = False,
        verbose: bool = True
    ) -> SequenceResult:
        """Fetch nucleotide or amino acid sequence (FASTA) of genes or transcripts.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: One or more Ensembl gene IDs as string or list (e.g., 'ENSG00000141510' or ['ENSG00000141510'])
                        Also supports WormBase and FlyBase IDs
            translate: If True, returns amino acid sequences; if False, returns nucleotide sequences. Default: False
                      Nucleotide sequences fetched from Ensembl REST API, amino acid from UniProt REST API
            isoforms: If True, returns sequences of all known transcripts (only for gene IDs). Default: False
            verbose: If True, prints progress information. Default: True
            
        Returns:
            SequenceResult: List containing the requested sequences in FASTA format
            Battle testing revealed the actual return is a list, not the various formats mentioned before
        
        Example workflow for protein sequence:
            1. search_genes('TP53', 'homo_sapiens') → 'ENSG00000141510'
            2. get_sequences('ENSG00000141510', translate=True)
            
        Example output:
            List of sequences in FASTA format: ['>ENSG00000141510', 'MEEPQSDPSVEPPLSQ...']
        
        Downstream tools that use protein sequences:
            - alphafold_predict: Predict 3D structure from protein sequence
            - blast_sequence: Search for similar sequences
        """
        with start_action(action_type="gget_seq", ensembl_ids=ensembl_ids, translate=translate):
            result = gget.seq(
                ens_ids=ensembl_ids, 
                translate=translate, 
                isoforms=isoforms,
                verbose=verbose
            )
            return result

    async def get_reference(
        self, 
        species: str = "homo_sapiens",
        which: Union[str, List[str]] = "all",
        release: Optional[int] = None,
        ftp: bool = False,
        list_species: bool = False,
        list_iv_species: bool = False,
        verbose: bool = True
    ) -> Union[Dict[str, Any], List[str]]:
        """Fetch FTPs for reference genomes and annotations by species from Ensembl.
        
        Args:
            species: Species in format "genus_species" (e.g., "homo_sapiens"). 
                    Shortcuts supported: "human", "mouse", "human_grch37"
            which: Which results to return. Default: "all" (all available results)
                  Options: 'gtf' (annotation), 'cdna' (transcriptome), 'dna' (genome),
                  'cds' (coding sequences), 'cdrna' (non-coding RNA), 'pep' (protein translations)
                  Can be single string or list of strings
            release: Ensembl release number (e.g., 104). Default: None (latest release)
            ftp: If True, returns only requested FTP links as list. Default: False
            list_species: If True and species=None, returns list of vertebrate species. Default: False
            list_iv_species: If True and species=None, returns list of invertebrate species. Default: False
            verbose: If True, prints progress information. Default: True
        
        Returns:
            Union[Dict[str, Any], List[str]]: Dictionary with URLs, versions, and metadata 
            (or list of URLs if ftp=True)
            
        Example:
            Input: species="homo_sapiens", which="gtf"
            Output: Dictionary containing GTF URLs with Ensembl version and release info
        """
        with start_action(action_type="gget_ref", species=species, which=which):
            result = gget.ref(
                species=species, 
                which=which, 
                release=release,
                ftp=ftp,
                list_species=list_species,
                list_iv_species=list_iv_species,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def blast_sequence(
        self, 
        sequence: str,
        program: str = "default",
        database: str = "default",
        limit: int = 50,
        expect: float = 10.0,
        low_comp_filt: bool = False,
        megablast: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """BLAST a nucleotide or amino acid sequence against any BLAST database.
        
        Args:
            sequence: Nucleotide or amino acid sequence (string) or path to FASTA file
                     (If FASTA has multiple sequences, only first will be submitted)
            program: BLAST program - 'blastn', 'blastp', 'blastx', 'tblastn', or 'tblastx'
                    Default: "default" (auto-detects: 'blastn' for nucleotide, 'blastp' for amino acid)
            database: BLAST database - 'nt', 'nr', 'refseq_rna', 'refseq_protein', 'swissprot', 'pdbaa', 'pdbnt'
                     Default: "default" (auto-detects: 'nt' for nucleotide, 'nr' for amino acid)
            limit: Maximum number of hits to return. Default: 50
            expect: Expect value cutoff (float). Default: 10.0
            low_comp_filt: Apply low complexity filter. Default: False
            megablast: Use MegaBLAST algorithm (blastn only). Default: True
            verbose: Print progress information. Default: True
        
        Returns:
            Dict[str, Any]: DataFrame with BLAST results including alignment details and scores
            
        Example:
            Input: sequence="ATGCGATCGTAGC", program="blastn", database="nt"
            Output: DataFrame with BLAST hits, E-values, scores, and alignments
        
        Note: 
            - NCBI server rule: Run scripts weekends or 9pm-5am ET weekdays for >50 searches
            - More info on databases: https://ncbi.github.io/blast-cloud/blastdb/available-blastdbs.html
        """
        with start_action(action_type="gget_blast", sequence_length=len(sequence), program=program):
            result = gget.blast(
                sequence=sequence,
                program=program,
                database=database,
                limit=limit,
                expect=expect,
                low_comp_filt=low_comp_filt,
                megablast=megablast,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def blat_sequence(
        self, 
        sequence: str,
        seqtype: str = "default",
        assembly: str = "human",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """BLAT a nucleotide or amino acid sequence against any BLAT UCSC assembly.
        
        Args:
            sequence: Nucleotide or amino acid sequence (string) or path to FASTA file containing one sequence
            seqtype: Sequence type - 'DNA', 'protein', 'translated%20RNA', or 'translated%20DNA'
                    Default: "default" (auto-detects: 'DNA' for nucleotide, 'protein' for amino acid)
            assembly: Genome assembly - 'human' (hg38), 'mouse' (mm39), 'zebrafinch' (taeGut2), 
                     or any assembly from https://genome.ucsc.edu/cgi-bin/hgBlat
                     Default: "human" (hg38)
            verbose: Print progress information. Default: True
        
        Returns:
            Dict[str, Any]: DataFrame with BLAT results including genomic coordinates and alignment details
            
        Example:
            Input: sequence="ATGCGATCGTAGC", assembly="human"
            Output: DataFrame with chromosome, start, end positions and alignment scores
        """
        with start_action(action_type="gget_blat", sequence_length=len(sequence), assembly=assembly):
            result = gget.blat(
                sequence=sequence,
                seqtype=seqtype,
                assembly=assembly,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def muscle_align(
        self, 
        sequences: Union[List[str], str],
        super5: bool = False,
        out: Optional[str] = None,
        verbose: bool = True
    ) -> Optional[str]:
        """Align multiple nucleotide or amino acid sequences using MUSCLE v5 algorithm.
        
        Args:
            sequences: List of sequences or path to FASTA file containing sequences to be aligned
            super5: If True, use Super5 algorithm instead of PPP (for large inputs with hundreds of sequences). Default: False
            out: Path to save aligned FASTA (.afa) file (e.g., 'path/to/results.afa'). 
                 Default: None (results printed in Clustal format)
            verbose: Print progress information. Default: True
        
        Returns:
            Optional[str]: Alignment results in aligned FASTA (.afa) format, or None if saved to file
            
        Example:
            Input: sequences=["ATGCGATC", "ATGCGTTC", "ATGCGATG"]
            Output: Aligned sequences in FASTA format or saved to file if 'out' specified
        """
        with start_action(action_type="gget_muscle", num_sequences=len(sequences) if isinstance(sequences, list) else None):
            result = gget.muscle(
                fasta=sequences, 
                super5=super5,
                out=out,
                verbose=verbose
            )
            return result

    async def diamond_align(
        self, 
        sequences: Union[str, List[str]],
        reference: Union[str, List[str]],
        translated: bool = False,
        diamond_db: Optional[str] = None,
        sensitivity: str = "very-sensitive",
        threads: int = 1,
        diamond_binary: Optional[str] = None,
        verbose: bool = True,
        out: Optional[str] = None
    ) -> Dict[str, Any]:
        """Align multiple protein or translated DNA sequences using DIAMOND.
        
        Args:
            sequences: Query sequences (string, list) or path to FASTA file with sequences to align against reference
            reference: Reference sequences (string, list) or path to FASTA file with reference sequences
                      Set translated=True if reference is amino acids and query is nucleotides
            translated: If True, performs translated alignment of nucleotide sequences to amino acid references. Default: False
            diamond_db: Path to save DIAMOND database created from reference. 
                       Default: None (temporary db deleted after alignment or saved in 'out' if provided)
            sensitivity: DIAMOND alignment sensitivity - 'fast', 'mid-sensitive', 'sensitive', 
                        'more-sensitive', 'very-sensitive', or 'ultra-sensitive'. Default: "very-sensitive"
            threads: Number of threads for alignment. Default: 1
            diamond_binary: Path to DIAMOND binary (e.g., 'path/bins/Linux/diamond'). 
                           Default: None (uses DIAMOND binary installed with gget)
            verbose: Print progress information. Default: True
            out: Path to folder to save DIAMOND results. Default: None (standard out, temporary files deleted)
        
        Returns:
            Dict[str, Any]: DataFrame with DIAMOND alignment results including similarity scores and positions
            
        Example:
            Input: sequences=["MKVLWA"], reference=["MKVLWAICAV"], sensitivity="sensitive"
            Output: DataFrame with alignment scores, positions, and match details
        """
        with start_action(action_type="gget_diamond", sensitivity=sensitivity):
            result = gget.diamond(
                query=sequences,
                reference=reference,
                translated=translated,
                diamond_db=diamond_db,
                sensitivity=sensitivity,
                threads=threads,
                diamond_binary=diamond_binary,
                verbose=verbose,
                out=out
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def archs4_expression(
        self, 
        gene: str,
        ensembl: bool = False,
        which: str = "correlation",
        gene_count: int = 100,
        species: str = "human",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Find correlated genes or tissue expression atlas using ARCHS4 RNA-seq database.
        
        Args:
            gene: Gene symbol (e.g., 'STAT4') or Ensembl ID if ensembl=True (e.g., 'ENSG00000138378')
            ensembl: If True, 'gene' parameter is treated as Ensembl gene ID. Default: False
            which: Analysis type - 'correlation' (most correlated genes) or 'tissue' (tissue expression atlas). Default: "correlation"
            gene_count: Number of correlated genes to return (only for correlation analysis). Default: 100
            species: Target species - 'human' or 'mouse' (only for tissue expression atlas). Default: "human"
            verbose: Print progress information. Default: True
        
        Returns:
            Dict[str, Any]: DataFrame with correlation table (100 most correlated genes with Pearson correlation)
                           or tissue expression atlas depending on 'which' parameter
            
        Example (correlation):
            Input: gene="STAT4", which="correlation"
            Output: DataFrame with 100 most correlated genes and Pearson correlation coefficients
            
        Example (tissue):
            Input: gene="STAT4", which="tissue", species="human"  
            Output: DataFrame with tissue expression levels across human samples
        """
        with start_action(action_type="gget_archs4", gene=gene, which=which):
            result = gget.archs4(
                gene=gene, 
                ensembl=ensembl,
                which=which, 
                gene_count=gene_count,
                species=species,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def enrichr_analysis(
        self, 
        genes: List[str],
        database: str = "KEGG_2021_Human",
        species: str = "human",
        background_list: Optional[List[str]] = None,
        background: bool = False,
        ensembl: bool = False,
        ensembl_bkg: bool = False,
        plot: bool = False,
        kegg_out: Optional[str] = None,
        kegg_rank: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Perform functional enrichment analysis on gene list using Enrichr.
        
        Args:
            genes: List of gene symbols (e.g., ['PHF14', 'RBM3']) or Ensembl IDs if ensembl=True
            database: Reference database or shortcuts for human/mouse:
                     'pathway' (KEGG_2021_Human), 'transcription' (ChEA_2016), 
                     'ontology' (GO_Biological_Process_2021), 'diseases_drugs' (GWAS_Catalog_2019),
                     'celltypes' (PanglaoDB_Augmented_2021), 'kinase_interactions' (KEA_2015)
                     Or full database name from https://maayanlab.cloud/Enrichr/#libraries
            species: Species database - 'human', 'mouse', 'fly', 'yeast', 'worm', 'fish'. Default: "human"
            background_list: Custom background genes (only for human/mouse). Default: None
            background: Use >20,000 default background genes (only for human/mouse). Default: False
            ensembl: If True, 'genes' are Ensembl gene IDs. Default: False
            ensembl_bkg: If True, 'background_list' are Ensembl gene IDs. Default: False
            plot: Create graphical overview of first 15 results. Default: False
            kegg_out: Path to save highlighted KEGG pathway image (e.g., 'path/kegg_pathway.png'). Default: None
            kegg_rank: Pathway rank to plot in KEGG image. Default: 1
            verbose: Print progress information. Default: True
        
        Returns:
            Dict[str, Any]: DataFrame with enrichment results including pathways, p-values, and statistical measures
            Battle testing confirmed functional analysis capabilities with cancer genes
            
        Example:
            Input: genes=['PHF14', 'RBM3', 'MSL1'], database='pathway'  
            Output: DataFrame with KEGG pathway enrichment results and statistics
        """
        with start_action(action_type="gget_enrichr", genes=genes, database=database):
            result = gget.enrichr(
                genes=genes,
                database=database,
                species=species,
                background_list=background_list,
                background=background,
                ensembl=ensembl,
                ensembl_bkg=ensembl_bkg,
                plot=plot,
                kegg_out=kegg_out,
                kegg_rank=kegg_rank,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def bgee_orthologs(
        self, 
        gene_id: str,
        type: str = "orthologs",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Get orthologs or expression data for a gene from Bgee database.
        
        PREREQUISITE: Use search_genes to get Ensembl ID first.
        
        Args:
            gene_id: Ensembl gene ID (e.g., 'ENSG00000012048' for BRCA1)
            type: Type of data to retrieve - 'orthologs' (ortholog information across species) 
                  or 'expression' (expression data). Default: "orthologs"
            verbose: Print progress information. Default: True
            
        Returns:
            Dict[str, Any]: DataFrame with ortholog information across species or expression data from Bgee
        
        Example workflow:
            1. search_genes('BRCA1') → 'ENSG00000012048' 
            2. bgee_orthologs('ENSG00000012048') → ortholog data across species
            
        Example (expression):
            Input: gene_id='ENSG00000012048', type='expression'
            Output: DataFrame with expression data from Bgee database
        """
        with start_action(action_type="gget_bgee", gene_id=gene_id, type=type):
            result = gget.bgee(gene_id=gene_id, type=type, verbose=verbose)
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def get_pdb_structure(
        self, 
        pdb_id: str,
        resource: str = "pdb",
        identifier: Optional[str] = None,
        save: bool = False
    ) -> StructureResult:
        """Query RCSB PDB for protein structure/metadata of a given PDB ID.
        
        IMPORTANT: This tool requires a specific PDB ID (e.g., '7S7U'), NOT gene names.
        
        For gene-to-structure workflows:
        1. Use search_genes to get Ensembl ID
        2. Use get_sequences with translate=True to get protein sequence  
        3. Use alphafold_predict for structure prediction, OR
        4. Search external databases (PDB website) for known PDB IDs, then use this tool
        
        Args:
            pdb_id: PDB ID to query (e.g., '7S7U', '2GS6')
            resource: Type of information to return:
                     'pdb' (protein structure in PDB format - default)
                     'entry' (top-level PDB structure information)
                     'pubmed' (PubMed annotations for primary citation)
                     'assembly' (quaternary structure information)
                     'branched_entity', 'nonpolymer_entity', 'polymer_entity' (entity data)
                     'uniprot' (UniProt annotations)
                     '*_instance' variants (chain-specific data)
            identifier: Assembly/entity ID (numbers like 1) or chain ID (letters like 'A') if applicable. Default: None
            save: Save JSON/PDB results to current directory. Default: False
            
        Returns:
            StructureResult: JSON format (except resource='pdb' returns PDB format structure)
            Battle testing confirmed successful retrieval of real PDB structures
        
        Example:
            Input: pdb_id='7S7U', resource='pdb'
            Output: Protein structure in PDB format
            
        Example (metadata):
            Input: pdb_id='7S7U', resource='entry'
            Output: JSON with PDB entry information, resolution, method, etc.
            
        Alternative workflow for gene structure prediction:
            1. search_genes('EGFR') → get Ensembl ID
            2. get_sequences(ensembl_id, translate=True) → get protein sequence
            3. alphafold_predict(protein_sequence) → predict structure
        """
        with start_action(action_type="gget_pdb", pdb_id=pdb_id):
            result = gget.pdb(pdb_id=pdb_id, resource=resource, identifier=identifier, save=save)
            return result

    async def alphafold_predict(
        self, 
        sequence: Union[str, List[str]],
        out: Optional[str] = None,
        multimer_for_monomer: bool = False,
        relax: bool = False,
        multimer_recycles: int = 3,
        plot: bool = True,
        show_sidechains: bool = True,
        verbose: bool = True
    ) -> StructureResult:
        """Predict protein structure using simplified AlphaFold v2.3.0 algorithm.
        
        PREREQUISITE: Use get_sequences with translate=True to get protein sequence first.
        
        Workflow for gene structure prediction:
        1. search_genes → get Ensembl ID
        2. get_sequences with translate=True → get protein sequence
        3. alphafold_predict → predict structure
        
        Args:
            sequence: Amino acid sequence (string), list of sequences, or path to FASTA file
            out: Path to folder to save prediction results. Default: None (auto-generated with timestamp)
            multimer_for_monomer: Use multimer model for monomer prediction. Default: False
            relax: Apply AMBER relaxation to best model. Default: False
            multimer_recycles: Max recycling iterations for multimer model (higher=more accurate but slower). Default: 3
            plot: Create graphical overview of prediction. Default: True
            show_sidechains: Show side chains in plot. Default: True
            verbose: Print progress information. Default: True
            
        Returns:
            StructureResult: AlphaFold structure prediction - saves aligned error (JSON) and prediction (PDB) files
            Battle testing confirmed successful structure predictions with small proteins
        
        Example full workflow:
            1. search_genes('TP53') → 'ENSG00000141510'
            2. get_sequences('ENSG00000141510', translate=True) → 'MEEPQSDPSVEPPLSQ...'
            3. alphafold_predict('MEEPQSDPSVEPPLSQ...')
            
        Example output:
            Saves PDB structure file and confidence scores JSON in specified output folder
            
        Note: This uses simplified AlphaFold without templates and limited MSA database.
        For best accuracy, use full AlphaFold or AlphaFold Protein Structure Database.
        Please cite gget and AlphaFold papers when using this function.
        """
        sequence_length = len(sequence) if isinstance(sequence, str) else len(sequence) if isinstance(sequence, list) else None
        with start_action(action_type="gget_alphafold", sequence_length=sequence_length):
            result = gget.alphafold(
                sequence=sequence, 
                out=out,
                multimer_for_monomer=multimer_for_monomer,
                relax=relax,
                multimer_recycles=multimer_recycles,
                plot=plot,
                show_sidechains=show_sidechains,
                verbose=verbose
            )
            return result

    async def elm_analysis(
        self, 
        sequence: str,
        uniprot: bool = False,
        sensitivity: str = "very-sensitive",
        threads: int = 1,
        diamond_binary: Optional[str] = None,
        expand: bool = False,
        verbose: bool = True,
        out: Optional[str] = None
    ) -> Dict[str, Any]:
        """Locally predict Eukaryotic Linear Motifs from amino acid sequence or UniProt ID.
        
        Args:
            sequence: Amino acid sequence or UniProt accession (if uniprot=True)
            uniprot: If True, input is UniProt accession instead of amino acid sequence. Default: False
            sensitivity: DIAMOND alignment sensitivity - 'fast', 'mid-sensitive', 'sensitive',
                        'more-sensitive', 'very-sensitive', or 'ultra-sensitive'. Default: "very-sensitive"
            threads: Number of threads for DIAMOND alignment. Default: 1
            diamond_binary: Path to DIAMOND binary. Default: None (uses DIAMOND installed with gget)
            expand: Expand regex dataframe to include protein names, organisms, and references. Default: False
            verbose: Print progress information. Default: True
            out: Path to folder to save results. Default: None (standard out, temporary files deleted)
        
        Returns:
            Dict[str, Any]: Two dataframes - ortholog motifs (experimentally validated in orthologs) 
                           and regex motifs (direct regex matches in sequence)
                           
        Example:
            Input: sequence="MKVLWAICAVL", uniprot=False
            Output: {'ortholog_df': {...}, 'regex_df': {...}} with motif predictions
            
        Example (UniProt):
            Input: sequence="P04637", uniprot=True  
            Output: Motif analysis results for UniProt entry P04637
            
        Note: ELM data is for non-commercial use only (ELM Software License Agreement).
        """
        with start_action(action_type="gget_elm", sequence_length=len(sequence) if not uniprot else None):
            result = gget.elm(
                sequence=sequence,
                uniprot=uniprot,
                sensitivity=sensitivity,
                threads=threads,
                diamond_binary=diamond_binary,
                expand=expand,
                verbose=verbose,
                out=out
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
        searchterm: Optional[str] = None,
        cosmic_tsv_path: Optional[str] = None,
        limit: int = 100,
        download_cosmic: bool = False,
        cosmic_project: str = "cancer",
        cosmic_version: Optional[int] = None,
        grch_version: int = 37,
        email: Optional[str] = None,
        password: Optional[str] = None,
        gget_mutate: bool = False,
        keep_genome_info: bool = False,
        remove_duplicates: bool = False,
        seq_id_column: str = "seq_ID",
        mutation_column: str = "mutation",
        mut_id_column: str = "mutation_id",
        out: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Search COSMIC database for cancer mutations or download COSMIC databases.
        
        NOTE: Licence fees apply for commercial use of COSMIC (https://www.cosmickb.org/licensing).
        Database downloads require COSMIC account (https://cancer.sanger.ac.uk/cosmic/register).
        
        Args:
            searchterm: Search term - gene name/Ensembl ID, mutation, sample ID, etc. 
                       Examples: 'EGFR', 'ENST00000275493', 'c.650A>T', 'p.Q217L'
                       Set to None when downloading databases (download_cosmic=True)
            cosmic_tsv_path: Path to COSMIC TSV file (required when download_cosmic=False)
            limit: Number of search hits to return. Default: 100
            download_cosmic: If True, switches to database download mode. Default: False
            cosmic_project: COSMIC database type - 'cancer' (CMC), 'cancer_example', 'census', 
                           'resistance', 'cell_line', 'genome_screen', 'targeted_screen'. Default: "cancer"
            cosmic_version: COSMIC database version. Default: None (latest version)
            grch_version: Human GRCh reference genome version (37 or 38). Default: 37
            email: COSMIC login email (avoids interactive input). Default: None
            password: COSMIC login password (stored in plain text). Default: None
            gget_mutate: Create database modified for 'gget mutate' use. Default: False
            keep_genome_info: Keep genome location info in gget_mutate database. Default: False
            remove_duplicates: Remove duplicate rows in gget_mutate database. Default: False
            seq_id_column: Name of seq_id column for gget_mutate CSV. Default: "seq_ID"
            mutation_column: Name of mutation column for gget_mutate CSV. Default: "mutation"
            mut_id_column: Name of mutation_id column for gget_mutate CSV. Default: "mutation_id"
            out: Output path for results/database. Default: None (stdout/current directory)
            verbose: Print progress information. Default: True
            
        Returns:
            Dict[str, Any]: DataFrame with mutation data including positions, amino acid changes, cancer types
                           (for searches) or database download confirmation (for downloads)
        
        Example (search):
            Input: searchterm='PIK3CA', cosmic_tsv_path='path/to/cosmic.tsv'
            Output: Mutation data for PIK3CA gene
            
        Example (download):
            Input: download_cosmic=True, cosmic_project='cancer_example'
            Output: Downloads example COSMIC database to specified folder
            
        Note: This tool accepts gene symbols directly, no need for Ensembl ID conversion.
        """
        with start_action(action_type="gget_cosmic", searchterm=searchterm, limit=limit):
            result = gget.cosmic(
                searchterm=searchterm,
                cosmic_tsv_path=cosmic_tsv_path,
                limit=limit,
                download_cosmic=download_cosmic,
                cosmic_project=cosmic_project,
                cosmic_version=cosmic_version,
                grch_version=grch_version,
                email=email,
                password=password,
                gget_mutate=gget_mutate,
                keep_genome_info=keep_genome_info,
                remove_duplicates=remove_duplicates,
                seq_id_column=seq_id_column,
                mutation_column=mutation_column,
                mut_id_column=mut_id_column,
                out=out,
                verbose=verbose
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def mutate_sequences(
        self, 
        sequences: Union[str, List[str]],
        mutations: Union[str, List[str]],
        mut_column: str = "mutation",
        seq_id_column: str = "seq_ID",
        mut_id_column: Optional[str] = None,
        gtf: Optional[str] = None,
        gtf_transcript_id_column: Optional[str] = None,
        k: int = 30,
        min_seq_len: Optional[int] = None,
        optimize_flanking_regions: bool = False,
        remove_seqs_with_wt_kmers: bool = False,
        max_ambiguous: Optional[int] = None,
        merge_identical: bool = True,
        update_df: bool = False,
        update_df_out: Optional[str] = None,
        store_full_sequences: bool = False,
        translate: bool = False,
        translate_start: Optional[Union[int, str]] = None,
        translate_end: Optional[Union[int, str]] = None,
        out: Optional[str] = None,
        verbose: bool = True
    ) -> Union[Dict[str, Any], List[str]]:
        """Mutate nucleotide sequences according to provided mutations in standard annotation.
        
        Args:
            sequences: Path to FASTA file or sequences as string/list (e.g., 'AGCTAGCT' or ['ACTG', 'ATCG'])
            mutations: Path to CSV/TSV file or mutations as string/list (e.g., 'c.2C>T' or ['c.2C>T', 'c.1A>C'])
                      Standard mutation annotation format required (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/)
            mut_column: Name of mutation column in mutations file. Default: "mutation"
            seq_id_column: Name of sequence ID column in mutations file. Default: "seq_ID"
            mut_id_column: Name of mutation ID column. Default: None (uses mut_column)
            gtf: Path to GTF file for transcript boundaries (when sequences is genome FASTA). Default: None
            gtf_transcript_id_column: Column with transcript IDs when using GTF. Default: None
            k: Length of flanking sequences around mutation. Default: 30
            min_seq_len: Minimum length of output sequences (shorter sequences dropped). Default: None
            optimize_flanking_regions: Remove nucleotides to avoid wildtype k-mers. Default: False
            remove_seqs_with_wt_kmers: Remove sequences with wildtype (k+1)-mers. Default: False
            max_ambiguous: Maximum 'N' characters allowed in output. Default: None
            merge_identical: Merge identical mutant sequences. Default: True
            update_df: Generate updated mutations DataFrame with additional columns. Default: False
            update_df_out: Path for updated DataFrame output. Default: None (auto-generated)
            store_full_sequences: Include complete sequences in updated DataFrame. Default: False
            translate: Add amino acid sequences to updated DataFrame. Default: False
            translate_start: Translation start position or column name. Default: None
            translate_end: Translation end position or column name. Default: None
            out: Path to output FASTA file. Default: None (returns sequences to stdout)
            verbose: Print progress information. Default: True
        
        Returns:
            Union[Dict[str, Any], List[str]]: List of mutated sequences or updated DataFrame 
                                            (depending on update_df setting)
        
        Example (simple):
            Input: sequences=['ATGCGATC'], mutations=['c.2T>G']
            Output: List of mutated sequences with flanking regions
            
        Example (file-based):
            Input: sequences='seqs.fa', mutations='mutations.csv'
            Output: Mutated sequences according to mutations table
            
        Note: Sequence IDs in FASTA must match seq_ID column in mutations file.
        Supports complex mutations: substitutions (c.2C>T), insertions, deletions, inversions.
        """
        num_sequences = len(sequences) if isinstance(sequences, list) else 1
        with start_action(action_type="gget_mutate", num_sequences=num_sequences):
            result = gget.mutate(
                sequences=sequences,
                mutations=mutations,
                mut_column=mut_column,
                seq_id_column=seq_id_column,
                mut_id_column=mut_id_column,
                gtf=gtf,
                gtf_transcript_id_column=gtf_transcript_id_column,
                k=k,
                min_seq_len=min_seq_len,
                optimize_flanking_regions=optimize_flanking_regions,
                remove_seqs_with_wt_kmers=remove_seqs_with_wt_kmers,
                max_ambiguous=max_ambiguous,
                merge_identical=merge_identical,
                update_df=update_df,
                update_df_out=update_df_out,
                store_full_sequences=store_full_sequences,
                translate=translate,
                translate_start=translate_start,
                translate_end=translate_end,
                out=out,
                verbose=verbose
            )
            return result

    async def opentargets_analysis(
        self, 
        ensembl_id: str,
        resource: str = "diseases",
        limit: Optional[int] = None,
        verbose: bool = True,
        wrap_text: bool = False,
        filters: Optional[Dict[str, str]] = None,
        filter_mode: str = "and"
    ) -> Dict[str, Any]:
        """Query OpenTargets for diseases, drugs, and other data associated with a gene.
        
        PREREQUISITE: Use search_genes to get Ensembl ID first.
        
        Args:
            ensembl_id: Ensembl gene ID (e.g., 'ENSG00000169194')
            resource: Type of information to return:
                     'diseases' (gene-disease associations - default)
                     'drugs' (gene-drug associations)
                     'tractability' (druggability data)
                     'pharmacogenetics' (pharmacogenomics data)
                     'expression' (tissue/organ expression)
                     'depmap' (DepMap gene-disease effects)
                     'interactions' (protein-protein interactions)
            limit: Maximum number of results. Default: None (no limit)
                   Note: Not compatible with 'tractability' and 'depmap' resources
            verbose: Print progress messages. Default: True
            wrap_text: Display DataFrame with wrapped text for readability. Default: False
            filters: Filters to apply by resource type:
                    diseases: None
                    drugs: {'disease_id': 'EFO_0000274'}
                    tractability: None
                    pharmacogenetics: {'drug_id': 'CHEMBL535'}
                    expression: {'tissue_id': 'UBERON_0002245', 'anatomical_system': 'nervous system', 'organ': 'brain'}
                    depmap: {'tissue_id': 'UBERON_0002245'}
                    interactions: {'protein_a_id': 'ENSP00000304915', 'protein_b_id': 'ENSP00000379111', 'gene_b_id': 'ENSG00000077238'}
            filter_mode: How to combine multiple filters - 'and' or 'or'. Default: "and"
            
        Returns:
            Dict[str, Any]: DataFrame with disease/drug associations, clinical evidence, and experimental data
            Battle testing confirmed functional disease association analysis
        
        Example workflow:
            1. search_genes('APOE') → 'ENSG00000141510'
            2. opentargets_analysis('ENSG00000141510') → disease associations
            
        Example (with filters):
            Input: ensembl_id='ENSG00000169194', resource='expression', filters={'organ': 'brain'}
            Output: Brain expression data for specified gene
        """
        with start_action(action_type="gget_opentargets", ensembl_id=ensembl_id, resource=resource):
            result = gget.opentargets(
                ensembl_id=ensembl_id,
                resource=resource,
                limit=limit,
                verbose=verbose,
                wrap_text=wrap_text,
                filters=filters,
                filter_mode=filter_mode
            )
            return result.to_dict() if hasattr(result, 'to_dict') else result

    async def cellxgene_query(
        self, 
        species: str = "homo_sapiens",
        gene: Optional[Union[str, List[str]]] = None,
        ensembl: bool = False,
        column_names: List[str] = ["dataset_id", "assay", "suspension_type", "sex", "tissue_general", "tissue", "cell_type"],
        meta_only: bool = False,
        tissue: Optional[Union[str, List[str]]] = None,
        cell_type: Optional[Union[str, List[str]]] = None,
        development_stage: Optional[Union[str, List[str]]] = None,
        disease: Optional[Union[str, List[str]]] = None,
        sex: Optional[Union[str, List[str]]] = None,
        is_primary_data: bool = True,
        dataset_id: Optional[Union[str, List[str]]] = None,
        tissue_general_ontology_term_id: Optional[Union[str, List[str]]] = None,
        tissue_general: Optional[Union[str, List[str]]] = None,
        assay_ontology_term_id: Optional[Union[str, List[str]]] = None,
        assay: Optional[Union[str, List[str]]] = None,
        cell_type_ontology_term_id: Optional[Union[str, List[str]]] = None,
        development_stage_ontology_term_id: Optional[Union[str, List[str]]] = None,
        disease_ontology_term_id: Optional[Union[str, List[str]]] = None,
        donor_id: Optional[Union[str, List[str]]] = None,
        self_reported_ethnicity_ontology_term_id: Optional[Union[str, List[str]]] = None,
        self_reported_ethnicity: Optional[Union[str, List[str]]] = None,
        sex_ontology_term_id: Optional[Union[str, List[str]]] = None,
        suspension_type: Optional[Union[str, List[str]]] = None,
        tissue_ontology_term_id: Optional[Union[str, List[str]]] = None,
        census_version: str = "stable",
        verbose: bool = True,
        out: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query single-cell RNA-seq data from CZ CELLxGENE Discover using Census.
        
        NOTE: Querying large datasets requires >16 GB RAM and >5 Mbps internet connection.
        Use cell metadata attributes to define specific (sub)datasets of interest.
        
        Args:
            species: Target species - 'homo_sapiens' or 'mus_musculus'. Default: "homo_sapiens"
            gene: Gene name(s) or Ensembl ID(s) (e.g., ['ACE2', 'SLC5A1'] or ['ENSG00000130234']). Default: None
                  Set ensembl=True when providing Ensembl IDs
            ensembl: If True, genes are Ensembl IDs instead of gene names. Default: False
            column_names: Metadata columns to return in AnnData.obs. Default: ["dataset_id", "assay", "suspension_type", "sex", "tissue_general", "tissue", "cell_type"]
            meta_only: If True, returns only metadata DataFrame (AnnData.obs). Default: False
            tissue: Tissue(s) to query (e.g., ['lung', 'blood']). Default: None
            cell_type: Cell type(s) to query (e.g., ['mucus secreting cell']). Default: None
            development_stage: Development stage(s) to filter. Default: None
            disease: Disease(s) to filter. Default: None
            sex: Sex(es) to filter (e.g., 'female'). Default: None
            is_primary_data: If True, returns only canonical instance of cellular observation. Default: True
            dataset_id: CELLxGENE dataset ID(s) to query. Default: None
            tissue_general_ontology_term_id: High-level tissue UBERON ID(s). Default: None
            tissue_general: High-level tissue label(s). Default: None
            assay_ontology_term_id: Assay ontology term ID(s). Default: None
            assay: Assay type(s) as defined in CELLxGENE schema. Default: None
            cell_type_ontology_term_id: Cell type ontology term ID(s). Default: None
            development_stage_ontology_term_id: Development stage ontology term ID(s). Default: None
            disease_ontology_term_id: Disease ontology term ID(s). Default: None
            donor_id: Donor ID(s) as defined in CELLxGENE schema. Default: None
            self_reported_ethnicity_ontology_term_id: Ethnicity ontology ID(s). Default: None
            self_reported_ethnicity: Self-reported ethnicity. Default: None
            sex_ontology_term_id: Sex ontology ID(s). Default: None
            suspension_type: Suspension type(s) as defined in CELLxGENE schema. Default: None
            tissue_ontology_term_id: Tissue ontology term ID(s). Default: None
            census_version: Census version ('stable', 'latest', or specific date like '2023-05-15'). Default: "stable"
            verbose: Print progress information. Default: True
            out: Path to save AnnData h5ad file (or CSV when meta_only=True). Default: None
        
        Returns:
            Dict[str, Any]: AnnData object (when meta_only=False) or DataFrame (when meta_only=True)
                           with single-cell expression data and metadata
        
        Example:
            Input: gene=['ACE2'], tissue=['lung'], cell_type=['alveolar epithelial cell']
            Output: Single-cell expression data for ACE2 in lung alveolar epithelial cells
            
        Example (metadata only):
            Input: tissue=['brain'], meta_only=True
            Output: Metadata DataFrame for brain tissue datasets
        """
        with start_action(action_type="gget_cellxgene", genes=gene, tissues=tissue):
            result = gget.cellxgene(
                species=species,
                gene=gene,
                ensembl=ensembl,
                column_names=column_names,
                meta_only=meta_only,
                tissue=tissue,
                cell_type=cell_type,
                development_stage=development_stage,
                disease=disease,
                sex=sex,
                is_primary_data=is_primary_data,
                dataset_id=dataset_id,
                tissue_general_ontology_term_id=tissue_general_ontology_term_id,
                tissue_general=tissue_general,
                assay_ontology_term_id=assay_ontology_term_id,
                assay=assay,
                cell_type_ontology_term_id=cell_type_ontology_term_id,
                development_stage_ontology_term_id=development_stage_ontology_term_id,
                disease_ontology_term_id=disease_ontology_term_id,
                donor_id=donor_id,
                self_reported_ethnicity_ontology_term_id=self_reported_ethnicity_ontology_term_id,
                self_reported_ethnicity=self_reported_ethnicity,
                sex_ontology_term_id=sex_ontology_term_id,
                suspension_type=suspension_type,
                tissue_ontology_term_id=tissue_ontology_term_id,
                census_version=census_version,
                verbose=verbose,
                out=out
            )
            return result

    async def setup_databases(
        self, 
        module: str,
        verbose: bool = True,
        out: Optional[str] = None
    ) -> Dict[str, Any]:
        """Install third-party dependencies for specified gget modules.
        
        Some modules require pip and curl to be installed on the system.
        
        Args:
            module: gget module to install dependencies for - 'alphafold', 'cellxgene', 'elm', 'gpt', or 'cbio'
            verbose: Print progress information. Default: True
            out: Path to directory for downloaded files (currently only applies to 'elm' module).
                 Default: None (files saved in gget installation directory)
                 NOTE: Do not use this argument when downloading files for use with 'gget.elm'
        
        Returns:
            Dict[str, Any]: Setup status with success indicator and messages
            Battle testing confirmed setup functionality for ELM module
            
        Example:
            Input: module='elm'
            Output: Downloads and installs ELM dependencies for motif analysis
            
        Example:
            Input: module='cellxgene'
            Output: Installs CELLxGENE Census dependencies for single-cell data
            
        Note: Available modules requiring setup: 'alphafold', 'cellxgene', 'elm', 'gpt', 'cbio'
        """
        with start_action(action_type="gget_setup", module=module):
            # Valid modules that require setup based on gget.setup help
            valid_modules = ["alphafold", "cellxgene", "elm", "gpt", "cbio"]
            if module not in valid_modules:
                return {
                    "data": None,
                    "success": False,
                    "message": f"Invalid module '{module}'. Valid modules are: {', '.join(valid_modules)}"
                }
            
            try:
                result = gget.setup(module, verbose=verbose, out=out)
                return {
                    "data": result,
                    "success": True,
                    "message": f"Setup completed for {module} module"
                }
            except Exception as e:
                return {
                    "data": None,
                    "success": False,
                    "message": f"Setup failed for {module} module: {str(e)}"
                }

    # Local mode wrapper functions for large data
    async def get_sequences_local(
        self, 
        ensembl_ids: Union[str, List[str]],
        translate: bool = False,
        isoforms: bool = False,
        verbose: bool = True,
        output_path: Optional[str] = None,
        format: Literal["fasta"] = "fasta"
    ) -> LocalFileResult:
        """Fetch sequences and save to local file in stdio-local mode.
        
        PREREQUISITE: Use search_genes first to get Ensembl IDs from gene names/symbols.
        
        Args:
            ensembl_ids: One or more Ensembl gene IDs as string or list (e.g., 'ENSG00000141510' or ['ENSG00000141510'])
                        Also supports WormBase and FlyBase IDs
            translate: If True, returns amino acid sequences; if False, returns nucleotide sequences. Default: False
            isoforms: If True, returns sequences of all known transcripts (only for gene IDs). Default: False
            verbose: If True, prints progress information. Default: True
            output_path: Optional specific output path (will generate if not provided)
            format: Output format (currently supports 'fasta')
        
        Returns:
            LocalFileResult: Contains path, format, and success information instead of sequence data
            Battle testing confirmed reliable file creation with proper FASTA formatting
        """
        # Get the sequence data using the original function
        with start_action(action_type="gget_seq_local", ensembl_ids=ensembl_ids, translate=translate):
            result = gget.seq(ens_ids=ensembl_ids, translate=translate, isoforms=isoforms, verbose=verbose)
            
            # Generate base name from ensembl IDs
            base_name = f"sequences_{'_'.join(ensembl_ids[:3])}{'_protein' if translate else '_dna'}"
            if output_path:
                base_name = Path(output_path).stem
                
            # Save to file
            return self._save_to_local_file(result, format, base_name)

    async def get_pdb_structure_local(
        self, 
        pdb_id: str,
        resource: str = "pdb",
        identifier: Optional[str] = None,
        save: bool = False,
        output_path: Optional[str] = None,
        format: Literal["pdb"] = "pdb"
    ) -> LocalFileResult:
        """Fetch PDB structure and save to local file in stdio-local mode.
        
        Args:
            pdb_id: PDB ID to query (e.g., '7S7U', '2GS6')
            resource: Type of information to return - 'pdb' (structure), 'entry', 'pubmed', 'assembly', etc.
            identifier: Assembly/entity ID (numbers like 1) or chain ID (letters like 'A') if applicable. Default: None
            save: Save JSON/PDB results to current directory. Default: False
            output_path: Optional specific output path (will generate if not provided)
            format: Output format (currently supports 'pdb')
        
        Returns:
            LocalFileResult: Contains path, format, and success information instead of structure data
            Battle testing confirmed successful retrieval of real PDB structures
        """
        with start_action(action_type="gget_pdb_local", pdb_id=pdb_id):
            result = gget.pdb(pdb_id=pdb_id, resource=resource, identifier=identifier, save=save)
            
            base_name = f"structure_{pdb_id}_{resource}"
            if output_path:
                base_name = Path(output_path).stem
                
            return self._save_to_local_file(result, format, base_name)

    async def alphafold_predict_local(
        self, 
        sequence: Union[str, List[str]],
        out: Optional[str] = None,
        multimer_for_monomer: bool = False,
        relax: bool = False,
        multimer_recycles: int = 3,
        plot: bool = True,
        show_sidechains: bool = True,
        verbose: bool = True,
        output_path: Optional[str] = None,
        format: Literal["pdb"] = "pdb"
    ) -> LocalFileResult:
        """Predict protein structure using AlphaFold and save to local file.
        
        Args:
            sequence: Amino acid sequence (string), list of sequences, or path to FASTA file
            out: Path to folder to save prediction results. Default: None (auto-generated with timestamp)
            multimer_for_monomer: Use multimer model for monomer prediction. Default: False
            relax: Apply AMBER relaxation to best model. Default: False
            multimer_recycles: Max recycling iterations for multimer model (higher=more accurate but slower). Default: 3
            plot: Create graphical overview of prediction. Default: True
            show_sidechains: Show side chains in plot. Default: True
            verbose: Print progress information. Default: True
            output_path: Optional specific output path (will generate if not provided)
            format: Output format (currently supports 'pdb')
        
        Returns:
            LocalFileResult: Contains path, format, and success information instead of structure data
            Battle testing confirmed successful AlphaFold predictions with small proteins
        """
        sequence_length = len(sequence) if isinstance(sequence, str) else len(sequence) if isinstance(sequence, list) else None
        with start_action(action_type="gget_alphafold_local", sequence_length=sequence_length):
            result = gget.alphafold(
                sequence=sequence, 
                out=out,
                multimer_for_monomer=multimer_for_monomer,
                relax=relax,
                multimer_recycles=multimer_recycles,
                plot=plot,
                show_sidechains=show_sidechains,
                verbose=verbose
            )
            
            base_name = f"alphafold_prediction_{str(uuid.uuid4())[:8]}"
            if output_path:
                base_name = Path(output_path).stem
                
            return self._save_to_local_file(result, format, base_name)

    async def muscle_align_local(
        self, 
        sequences: Union[List[str], str],
        super5: bool = False,
        verbose: bool = True,
        output_path: Optional[str] = None,
        format: Literal["fasta", "afa"] = "fasta"
    ) -> LocalFileResult:
        """Align sequences using MUSCLE and save to local file.
        
        Args:
            sequences: List of sequences or path to FASTA file containing sequences to be aligned
            super5: If True, use Super5 algorithm instead of PPP (for large inputs with hundreds of sequences). Default: False
            verbose: Print progress information. Default: True
            output_path: Optional specific output path (will generate if not provided)
            format: Output format ('fasta' for FASTA format, 'afa' for aligned FASTA format)
        
        Returns:
            LocalFileResult: Contains path, format, and success information instead of alignment data
            Battle testing confirmed successful alignment of real biological sequences
        """
        with start_action(action_type="gget_muscle_local", num_sequences=len(sequences) if isinstance(sequences, list) else None):
            # Generate output file path
            base_name = f"muscle_alignment_{len(sequences)}seqs"
            if output_path:
                base_name = Path(output_path).stem
                
            extension = ".fasta" if format == "fasta" else ".afa"
            file_path = self.output_dir / f"{base_name}{extension}"
            
            # Use gget.muscle with out parameter to save directly to file
            result = gget.muscle(fasta=sequences, super5=super5, out=str(file_path), verbose=verbose)
            
            return {
                "path": str(file_path),
                "format": format,
                "success": True
            }

    async def diamond_align_local(
        self, 
        sequences: Union[str, List[str]],
        reference: Union[str, List[str]],
        translated: bool = False,
        diamond_db: Optional[str] = None,
        sensitivity: str = "very-sensitive",
        threads: int = 1,
        diamond_binary: Optional[str] = None,
        verbose: bool = True,
        out: Optional[str] = None,
        output_path: Optional[str] = None,
        format: Literal["json", "tsv"] = "json"
    ) -> LocalFileResult:
        """Align sequences using DIAMOND and save to local file.
        
        Args:
            sequences: Query sequences (string, list) or path to FASTA file with sequences to align against reference
            reference: Reference sequences (string, list) or path to FASTA file with reference sequences
            translated: If True, performs translated alignment of nucleotide sequences to amino acid references. Default: False
            diamond_db: Path to save DIAMOND database created from reference. Default: None
            sensitivity: DIAMOND alignment sensitivity - 'fast', 'mid-sensitive', 'sensitive', 
                        'more-sensitive', 'very-sensitive', or 'ultra-sensitive'. Default: "very-sensitive"
            threads: Number of threads for alignment. Default: 1
            diamond_binary: Path to DIAMOND binary. Default: None (uses DIAMOND binary installed with gget)
            verbose: Print progress information. Default: True
            out: Path to folder to save DIAMOND results. Default: None
            output_path: Optional specific output path (will generate if not provided)
            format: Output format ('json' recommended, 'tsv' also supported)
        
        Returns:
            LocalFileResult: Contains path, format, and success information instead of alignment data
            Battle testing showed reliable DIAMOND alignment functionality
        """
        with start_action(action_type="gget_diamond_local", sensitivity=sensitivity):
            result = gget.diamond(
                query=sequences,
                reference=reference,
                translated=translated,
                diamond_db=diamond_db,
                sensitivity=sensitivity,
                threads=threads,
                diamond_binary=diamond_binary,
                verbose=verbose,
                out=out
            )
            
            base_name = f"diamond_alignment_{str(uuid.uuid4())[:8]}"
            if output_path:
                base_name = Path(output_path).stem
                
            # Convert result to dict if it has to_dict method
            if hasattr(result, 'to_dict'):
                result = result.to_dict()
                
            return self._save_to_local_file(result, format, base_name)


def create_app(transport_mode: str = "stdio", output_dir: Optional[str] = None):
    """Create and configure the FastMCP application."""
    return GgetMCP(transport_mode=transport_mode, output_dir=output_dir)

# CLI application setup
cli_app = typer.Typer(help="gget MCP Server CLI")

@cli_app.command()
def run_server(
    host: Annotated[str, typer.Option(help="Host to run the server on.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port to run the server on.")] = DEFAULT_PORT,
    transport: Annotated[str, typer.Option(help="Transport type: stdio, stdio-local, streamable-http, or sse")] = DEFAULT_TRANSPORT,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory for local files (stdio-local mode)")] = None
):
    """Runs the gget MCP server."""
    # Validate transport value
    if transport not in ["stdio", "stdio-local", "streamable-http", "sse"]:
        typer.echo(f"Invalid transport: {transport}. Must be one of: stdio, stdio-local, streamable-http, sse")
        raise typer.Exit(1)
        
    app = create_app(transport_mode=transport, output_dir=output_dir)

    # Different transports need different arguments
    if transport in ["stdio", "stdio-local"]:
        app.run(transport="stdio")  # Both stdio modes use stdio transport
    else:
        app.run(transport=transport, host=host, port=port)

if __name__ == "__main__":
    cli_app() 