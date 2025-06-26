"""Tests for gget-mcp server functionality."""

import pytest
from unittest.mock import Mock, patch
from gget_mcp.server import GgetMCP
import tempfile
import os
from pathlib import Path


class TestGgetMCP:
    """Test cases for GgetMCP server."""
    
    @pytest.fixture
    def server(self):
        """Create a test server instance (simplified mode by default)."""
        return GgetMCP()
    
    @pytest.fixture
    def extended_server(self):
        """Create a test server instance for extended mode."""
        return GgetMCP(extended_mode=True)
    
    @pytest.fixture
    def local_server(self):
        """Create a test server instance for local mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield GgetMCP(transport_mode="stdio-local", output_dir=temp_dir)

    def test_server_initialization(self, server):
        """Test that server initializes correctly."""
        assert server.prefix == "gget_"
        assert hasattr(server, '_register_gget_tools')
        assert server.extended_mode == False
    
    def test_extended_server_initialization(self, extended_server):
        """Test that extended server initializes correctly."""
        assert extended_server.prefix == "gget_"
        assert hasattr(extended_server, '_register_gget_tools')
        assert extended_server.extended_mode == True
    
    # Tests for simplified API (using default server fixture)
    @pytest.mark.asyncio
    async def test_search_genes_simple_success(self, server):
        """Test successful gene search with simplified API."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"gene1": "data1"}
        mock_result.__len__ = Mock(return_value=1)
        
        with patch('gget.search', return_value=mock_result):
            response = await server.search_genes_simple(
                search_terms=["BRCA1"], 
                species="homo_sapiens"
            )
            
        assert response == {"gene1": "data1"}
    
    @pytest.mark.asyncio
    async def test_get_gene_info_simple_success(self, server):
        """Test successful gene info retrieval with simplified API."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"ENSG123": {"name": "Test Gene"}}
        
        with patch('gget.info', return_value=mock_result):
            response = await server.get_gene_info_simple(
                ensembl_ids=["ENSG123"]
            )
            
        assert response == {"ENSG123": {"name": "Test Gene"}}
    
    @pytest.mark.asyncio
    async def test_get_sequences_simple_success(self, server):
        """Test successful sequence retrieval with simplified API."""
        mock_result = {"ENSG123": "ATGCGATCG"}
        
        with patch('gget.seq', return_value=mock_result):
            response = await server.get_sequences_simple(
                ensembl_ids=["ENSG123"],
                translate=False
            )
            
        assert response == mock_result
    
    @pytest.mark.asyncio
    async def test_blast_sequence_simple_success(self, server):
        """Test successful BLAST search with simplified API."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"hit1": "data1"}
        mock_result.__len__ = Mock(return_value=1)
        
        with patch('gget.blast', return_value=mock_result):
            response = await server.blast_sequence_simple(
                sequence="ATGCGATCG",
                program="blastn"
            )
            
        assert response == {"hit1": "data1"}
    
    @pytest.mark.asyncio
    async def test_enrichr_analysis_simple_success(self, server):
        """Test successful enrichment analysis with simplified API."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"pathway1": "significant"}
        
        with patch('gget.enrichr', return_value=mock_result):
            response = await server.enrichr_analysis_simple(
                genes=["BRCA1", "TP53"],
                database="pathway"  # Using shortcut
            )
            
        assert response == {"pathway1": "significant"}

    @pytest.mark.asyncio
    async def test_alzheimer_genes_batch_processing(self, server):
        """Test batch processing with important Alzheimer genes for search_genes_simple and get_gene_info_simple."""
        # List of important Alzheimer's disease genes
        alzheimer_genes = [
            "APOE", "APP", "PSEN1", "PSEN2", "TREM2", "CLU", "CR1", "BIN1", 
            "ABCA7", "CD33", "MS4A6A", "MS4A4E", "CD2AP", "EPHA1", "INPP5D",
            "MEF2C", "HLA-DRB1", "PTK2B", "SORL1", "SLC24A4", "CASS4", "FERMT2"
        ]
        
        # Mock search result with Ensembl IDs for each gene
        mock_search_result = Mock()
        mock_search_dict = {}
        mock_ensembl_ids = []
        
        for i, gene in enumerate(alzheimer_genes):
            ensembl_id = f"ENSG{i:011d}"  # Generate mock Ensembl IDs
            mock_ensembl_ids.append(ensembl_id)
            mock_search_dict[gene] = {
                "ensembl_id": ensembl_id,
                "gene_name": gene,
                "description": f"Test description for {gene}"
            }
        
        mock_search_result.to_dict.return_value = mock_search_dict
        mock_search_result.__len__ = Mock(return_value=len(alzheimer_genes))
        
        # Test batch search for all Alzheimer genes
        with patch('gget.search', return_value=mock_search_result):
            search_response = await server.search_genes_simple(
                search_terms=alzheimer_genes,  # Batch request
                species="homo_sapiens"
            )
        
        assert len(search_response) == len(alzheimer_genes)
        assert all(gene in search_response for gene in alzheimer_genes)
        
        # Mock gene info result
        mock_info_result = Mock()
        mock_info_dict = {}
        
        for ensembl_id, gene in zip(mock_ensembl_ids, alzheimer_genes):
            mock_info_dict[ensembl_id] = {
                "ensembl_id": ensembl_id,
                "symbol": gene,
                "biotype": "protein_coding",
                "chromosome": "1",  # Simplified for test
                "start": 100000,
                "end": 200000,
                "description": f"Detailed info for {gene}"
            }
        
        mock_info_result.to_dict.return_value = mock_info_dict
        
        # Test batch gene info retrieval
        with patch('gget.info', return_value=mock_info_result):
            info_response = await server.get_gene_info_simple(
                ensembl_ids=mock_ensembl_ids  # Batch request with results from search
            )
        
        assert len(info_response) == len(alzheimer_genes)
        assert all(ensembl_id in info_response for ensembl_id in mock_ensembl_ids)
        
        # Verify that each gene info contains expected fields
        for ensembl_id in mock_ensembl_ids:
            gene_info = info_response[ensembl_id]
            assert "symbol" in gene_info
            assert "biotype" in gene_info
            assert "chromosome" in gene_info
            assert gene_info["biotype"] == "protein_coding"

    @pytest.mark.asyncio
    async def test_single_vs_batch_consistency(self, server):
        """Test that single gene requests work the same as batch requests with one gene."""
        test_gene = "APOE"
        mock_ensembl_id = "ENSG00000130203"
        
        # Mock results for single gene
        mock_search_single = Mock()
        mock_search_single.to_dict.return_value = {
            test_gene: {
                "ensembl_id": mock_ensembl_id,
                "gene_name": test_gene,
                "description": "Apolipoprotein E"
            }
        }
        
        mock_info_single = Mock()
        mock_info_single.to_dict.return_value = {
            mock_ensembl_id: {
                "ensembl_id": mock_ensembl_id,
                "symbol": test_gene,
                "biotype": "protein_coding"
            }
        }
        
        # Test single gene (string parameter)
        with patch('gget.search', return_value=mock_search_single):
            single_search = await server.search_genes_simple(
                search_terms=test_gene,  # Single string
                species="homo_sapiens"
            )
        
        with patch('gget.info', return_value=mock_info_single):
            single_info = await server.get_gene_info_simple(
                ensembl_ids=mock_ensembl_id  # Single string
            )
        
        # Test batch with one gene (list parameter)
        with patch('gget.search', return_value=mock_search_single):
            batch_search = await server.search_genes_simple(
                search_terms=[test_gene],  # List with one element
                species="homo_sapiens"
            )
        
        with patch('gget.info', return_value=mock_info_single):
            batch_info = await server.get_gene_info_simple(
                ensembl_ids=[mock_ensembl_id]  # List with one element
            )
        
        # Results should be identical
        assert single_search == batch_search
        assert single_info == batch_info

    # Tests for extended API (keeping existing tests and using extended_server fixture)
    @pytest.mark.asyncio
    async def test_search_genes_success(self, extended_server):
        """Test successful gene search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"gene1": "data1"}
        mock_result.__len__ = Mock(return_value=1)
        
        with patch('gget.search', return_value=mock_result):
            response = await extended_server.search_genes(
                search_terms=["BRCA1"], 
                species="homo_sapiens"
            )
            
        assert response == {"gene1": "data1"}
    
    @pytest.mark.asyncio
    async def test_search_genes_failure(self, extended_server):
        """Test gene search failure handling."""
        with patch('gget.search', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await extended_server.search_genes(
                    search_terms=["INVALID"], 
                    species="homo_sapiens"
                )
    
    @pytest.mark.asyncio
    async def test_get_gene_info_success(self, extended_server):
        """Test successful gene info retrieval."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"ENSG123": {"name": "Test Gene"}}
        
        with patch('gget.info', return_value=mock_result):
            response = await extended_server.get_gene_info(
                ensembl_ids=["ENSG123"]
            )
            
        assert response == {"ENSG123": {"name": "Test Gene"}}
    
    @pytest.mark.asyncio
    async def test_get_sequences_success(self, extended_server):
        """Test successful sequence retrieval."""
        mock_result = {"ENSG123": "ATGCGATCG"}
        
        with patch('gget.seq', return_value=mock_result):
            response = await extended_server.get_sequences(
                ensembl_ids=["ENSG123"],
                translate=False
            )
            
        assert response == mock_result
    
    @pytest.mark.asyncio
    async def test_blast_sequence_success(self, extended_server):
        """Test successful BLAST search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"hit1": "data1"}
        mock_result.__len__ = Mock(return_value=1)
        
        with patch('gget.blast', return_value=mock_result):
            response = await extended_server.blast_sequence(
                sequence="ATGCGATCG",
                program="blastn"
            )
            
        assert response == {"hit1": "data1"}
    
    @pytest.mark.asyncio 
    async def test_enrichr_analysis_success(self, extended_server):
        """Test successful enrichment analysis."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"pathway1": "significant"}
        
        with patch('gget.enrichr', return_value=mock_result):
            response = await extended_server.enrichr_analysis(
                genes=["BRCA1", "TP53"],
                database="KEGG_2021_Human"
            )
            
        assert response == {"pathway1": "significant"}

    @pytest.mark.asyncio
    async def test_diamond_align_success(self, extended_server):
        """Test successful DIAMOND alignment."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"alignment": "data"}
        
        with patch('gget.diamond', return_value=mock_result):
            response = await extended_server.diamond_align(
                sequences="MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLAS",
                reference="MSSSSWLLLSLVEVTAAQSTIEQQAKTFLDKFHEAEDLFYQSLLAS"
            )
            
        assert response == {"alignment": "data"}

    @pytest.mark.asyncio
    async def test_diamond_align_failure(self, extended_server):
        """Test DIAMOND alignment failure handling."""
        with patch('gget.diamond', side_effect=Exception("DIAMOND Error")):
            with pytest.raises(Exception, match="DIAMOND Error"):
                await extended_server.diamond_align(
                    sequences="INVALID",
                    reference="INVALID"
                )

    @pytest.mark.asyncio
    async def test_bgee_orthologs_success(self, extended_server):
        """Test successful Bgee ortholog search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"orthologs": ["gene1", "gene2"]}
        
        with patch('gget.bgee', return_value=mock_result):
            response = await extended_server.bgee_orthologs(
                gene_id="ENSG00000012048",  # Fixed parameter name
                type="orthologs"
            )
            
        assert response == {"orthologs": ["gene1", "gene2"]}

    @pytest.mark.asyncio
    async def test_bgee_orthologs_failure(self, extended_server):
        """Test Bgee ortholog search failure handling."""
        with patch('gget.bgee', side_effect=Exception("Bgee Error")):
            with pytest.raises(Exception, match="Bgee Error"):
                await extended_server.bgee_orthologs(
                    gene_id="INVALID",  # Fixed parameter name
                    type="orthologs"
                )

    @pytest.mark.asyncio
    async def test_elm_analysis_success(self, extended_server):
        """Test successful ELM analysis."""
        mock_ortholog_df = Mock()
        mock_ortholog_df.to_dict.return_value = {"ortholog": "data"}
        mock_regex_df = Mock()
        mock_regex_df.to_dict.return_value = {"regex": "data"}
        
        with patch('gget.elm', return_value=(mock_ortholog_df, mock_regex_df)):
            response = await extended_server.elm_analysis(
                sequence="LIAQSIGQASFV",
                sensitivity="very-sensitive"
            )
            
        expected_data = {
            "ortholog_df": {"ortholog": "data"},
            "regex_df": {"regex": "data"}
        }
        assert response == expected_data

    @pytest.mark.asyncio
    async def test_elm_analysis_uniprot_success(self, extended_server):
        """Test successful ELM analysis with UniProt ID."""
        mock_ortholog_df = Mock()
        mock_ortholog_df.to_dict.return_value = {"ortholog": "data"}
        mock_regex_df = Mock()
        mock_regex_df.to_dict.return_value = {"regex": "data"}
        
        with patch('gget.elm', return_value=(mock_ortholog_df, mock_regex_df)):
            response = await extended_server.elm_analysis(
                sequence="Q02410",
                uniprot=True,
                expand=True
            )
            
        expected_data = {
            "ortholog_df": {"ortholog": "data"},
            "regex_df": {"regex": "data"}
        }
        assert response == expected_data

    @pytest.mark.asyncio
    async def test_elm_analysis_failure(self, extended_server):
        """Test ELM analysis failure handling."""
        with patch('gget.elm', side_effect=Exception("ELM Error")):
            with pytest.raises(Exception, match="ELM Error"):
                await extended_server.elm_analysis(
                    sequence="INVALID"
                )

    @pytest.mark.asyncio
    async def test_mutate_sequences_success(self, extended_server):
        """Test successful sequence mutation."""
        mock_result = ["ATGCGTCG", "ATGCAACG"]
        
        with patch('gget.mutate', return_value=mock_result):
            response = await extended_server.mutate_sequences(
                sequences=["ATGCGATC"],
                mutations=["c.2C>T"]
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_mutate_sequences_multiple(self, extended_server):
        """Test successful mutation of multiple sequences."""
        mock_result = ["ATGCGTCG", "ATGCATCG", "ATGCGTAG"] 
        
        with patch('gget.mutate', return_value=mock_result):
            response = await extended_server.mutate_sequences(
                sequences=["ATGCGATC", "ATGCGCTG"],
                mutations=["c.2C>T", "c.3G>A"],
                k=25
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_mutate_sequences_failure(self, extended_server):
        """Test mutation failure handling."""
        with patch('gget.mutate', side_effect=Exception("Mutation Error")):
            with pytest.raises(Exception, match="Mutation Error"):
                await extended_server.mutate_sequences(
                    sequences=["INVALID"],
                    mutations=["invalid"]
                )

    @pytest.mark.asyncio
    async def test_opentargets_analysis_success(self, extended_server):
        """Test successful OpenTargets analysis."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"disease1": "associated"}
        
        with patch('gget.opentargets', return_value=mock_result):
            response = await extended_server.opentargets_analysis(
                ensembl_id="ENSG00000169194",
                resource="diseases"
            )
            
        assert response == {"disease1": "associated"}

    @pytest.mark.asyncio
    async def test_opentargets_analysis_drugs(self, extended_server):
        """Test OpenTargets drug analysis."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"drug1": "targets"}
        
        with patch('gget.opentargets', return_value=mock_result):
            response = await extended_server.opentargets_analysis(
                ensembl_id="ENSG00000169194",
                resource="drugs",
                limit=10
            )
            
        assert response == {"drug1": "targets"}

    @pytest.mark.asyncio
    async def test_opentargets_analysis_failure(self, extended_server):
        """Test OpenTargets analysis failure handling."""
        with patch('gget.opentargets', side_effect=Exception("OpenTargets Error")):
            with pytest.raises(Exception, match="OpenTargets Error"):
                await extended_server.opentargets_analysis(
                    ensembl_id="INVALID"
                )

    @pytest.mark.asyncio
    async def test_setup_databases_success(self, extended_server):
        """Test successful database setup."""
        mock_result = {
            "success": True,
            "message": "ELM databases installed successfully"
        }
        
        with patch('gget.setup', return_value=mock_result):
            response = await extended_server.setup_databases(
                module="elm"
            )
        
        expected_response = {
            "data": mock_result,
            "success": True,
            "message": "Setup completed for elm module"
        }
        assert response == expected_response

    @pytest.mark.asyncio
    async def test_setup_databases_invalid_module(self, extended_server):
        """Test setup with invalid module."""
        response = await extended_server.setup_databases(
            module="invalid_module"
        )
        
        assert response["success"] is False
        assert "Invalid module 'invalid_module'" in response["message"]
        assert "alphafold, cellxgene, elm, gpt, cbio" in response["message"]

    @pytest.mark.asyncio
    async def test_setup_databases_failure(self, extended_server):
        """Test setup database failure handling."""
        with patch('gget.setup', side_effect=Exception("Setup failed")):
            response = await extended_server.setup_databases(
                module="elm"
            )
        
        expected_response = {
            "data": None,
            "success": False,
            "message": "Setup failed for elm module: Setup failed"
        }
        assert response == expected_response

    @pytest.mark.asyncio
    async def test_get_reference_success(self, extended_server):
        """Test successful reference genome retrieval."""
        mock_result = {"gtf": "ftp://example.com/file.gtf"}
        
        with patch('gget.ref', return_value=mock_result):
            response = await extended_server.get_reference(
                species="homo_sapiens",
                which="gtf"
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_blat_sequence_success(self, extended_server):
        """Test successful BLAT search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"chr1": "12345"}
        
        with patch('gget.blat', return_value=mock_result):
            response = await extended_server.blat_sequence(
                sequence="ATGCGATCG",
                assembly="human",
                seqtype="DNA"
            )
            
        assert response == {"chr1": "12345"}

    @pytest.mark.asyncio
    async def test_muscle_align_success(self, extended_server):
        """Test successful MUSCLE alignment."""
        mock_result = ">seq1\nATGCGATC\n>seq2\nATGCGTTC"
        
        with patch('gget.muscle', return_value=mock_result):
            response = await extended_server.muscle_align(
                sequences=["ATGCGATC", "ATGCGTTC"],
                super5=False
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_archs4_expression_success(self, extended_server):
        """Test successful ARCHS4 expression analysis."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"gene1": 0.95}
        
        with patch('gget.archs4', return_value=mock_result):
            response = await extended_server.archs4_expression(
                gene="STAT4",
                which="correlation"
            )
            
        assert response == {"gene1": 0.95}

    @pytest.mark.asyncio
    async def test_get_pdb_structure_success(self, extended_server):
        """Test successful PDB structure retrieval."""
        mock_result = "HEADER    TRANSFERASE                             01-JAN-00   1ABC"
        
        with patch('gget.pdb', return_value=mock_result):
            response = await extended_server.get_pdb_structure(
                pdb_id="1ABC",
                resource="pdb"
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_alphafold_predict_success(self, extended_server):
        """Test successful AlphaFold prediction."""
        mock_result = "HEADER    ALPHAFOLD PREDICTION"
        
        with patch('gget.alphafold', return_value=mock_result):
            response = await extended_server.alphafold_predict(
                sequence="MKVLWAICAV"
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_cosmic_search_success(self, extended_server):
        """Test successful COSMIC search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"mutation1": "pathogenic"}
        
        with patch('gget.cosmic', return_value=mock_result):
            response = await extended_server.cosmic_search(
                searchterm="PIK3CA",
                limit=50
            )
            
        assert response == {"mutation1": "pathogenic"}

    @pytest.mark.asyncio
    async def test_cellxgene_query_success(self, extended_server):
        """Test successful CELLxGENE query."""
        mock_result = {"cell1": "expression_data"}
        
        with patch('gget.cellxgene', return_value=mock_result):
            response = await extended_server.cellxgene_query(
                gene=["ACE2"],
                tissue=["lung"]
            )
            
        assert response == mock_result

    # New tests for local mode functions
    @pytest.mark.asyncio
    async def test_get_sequences_local_dna_success(self, local_server):
        """Test successful DNA sequence retrieval in local mode."""
        mock_result = {
            "ENSG00000141510": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG",
            "ENSG00000012048": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATC"
        }
        
        with patch('gget.seq', return_value=mock_result):
            response = await local_server.get_sequences_local(
                ensembl_ids=["ENSG00000141510", "ENSG00000012048"],
                translate=False,
                format="fasta"
            )
            
        # Should return file path information
        assert response["success"] is True
        assert response["format"] == "fasta"
        assert response["path"] is not None
        assert Path(response["path"]).exists()
        assert "sequences_ENSG00000141510_ENSG00000012048_dna" in response["path"]
        assert response["path"].endswith(".fasta")
        
        # Verify file content
        with open(response["path"], 'r') as f:
            content = f.read()
            assert ">ENSG00000141510" in content
            assert ">ENSG00000012048" in content
            assert "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG" in content

    @pytest.mark.asyncio
    async def test_get_sequences_local_protein_success(self, local_server):
        """Test successful protein sequence retrieval in local mode."""
        mock_result = {
            "ENSG00000141510": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP",
            "ENSG00000012048": "MDLSALRVEEVQNVINAASQKGNQAMWSLVPFLAQQKGLSQRQQSQNQK"
        }
        
        with patch('gget.seq', return_value=mock_result):
            response = await local_server.get_sequences_local(
                ensembl_ids=["ENSG00000141510", "ENSG00000012048"],
                translate=True,
                format="fasta"
            )
            
        # Should return file path information
        assert response["success"] is True
        assert response["format"] == "fasta"
        assert response["path"] is not None
        assert Path(response["path"]).exists()
        assert "sequences_ENSG00000141510_ENSG00000012048_protein" in response["path"]
        assert response["path"].endswith(".fasta")
        
        # Verify file content
        with open(response["path"], 'r') as f:
            content = f.read()
            assert ">ENSG00000141510" in content
            assert ">ENSG00000012048" in content
            assert "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP" in content

    @pytest.mark.asyncio
    async def test_get_sequences_local_custom_path(self, local_server):
        """Test sequence retrieval with custom output path."""
        mock_result = {"ENSG00000141510": "ATGGAGGAGCCGCAGTCAG"}
        
        with patch('gget.seq', return_value=mock_result):
            response = await local_server.get_sequences_local(
                ensembl_ids=["ENSG00000141510"],
                translate=False,
                output_path="tp53_sequence",
                format="fasta"
            )
            
        assert response["success"] is True
        assert "tp53_sequence.fasta" in response["path"]

    @pytest.mark.asyncio
    async def test_get_sequences_local_single_gene(self, local_server):
        """Test sequence retrieval for single gene."""
        mock_result = {"ENSG00000012048": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTC"}
        
        with patch('gget.seq', return_value=mock_result):
            response = await local_server.get_sequences_local(
                ensembl_ids=["ENSG00000012048"],
                translate=False,
                format="fasta"
            )
            
        assert response["success"] is True
        assert "sequences_ENSG00000012048_dna" in response["path"]
        assert response["path"].endswith(".fasta")

    @pytest.mark.asyncio
    async def test_get_pdb_structure_local_success(self, local_server):
        """Test successful PDB structure retrieval in local mode."""
        mock_result = "HEADER    TUMOR SUPPRESSOR/DNA                     20-JUL-00   1TUP\nATOM      1  N   SER A   1      25.369  23.134  23.337  1.00 25.00           N"
        
        with patch('gget.pdb', return_value=mock_result):
            response = await local_server.get_pdb_structure_local(
                pdb_id="1TUP",
                resource="pdb",
                format="pdb"
            )
            
        assert response["success"] is True
        assert response["format"] == "pdb"
        assert response["path"] is not None
        assert Path(response["path"]).exists()
        assert "structure_1TUP_pdb" in response["path"]
        assert response["path"].endswith(".pdb")
        
        # Verify file content
        with open(response["path"], 'r') as f:
            content = f.read()
            assert "HEADER    TUMOR SUPPRESSOR/DNA" in content
            assert "ATOM      1  N   SER A   1" in content

    @pytest.mark.asyncio
    async def test_get_pdb_structure_local_alphafold(self, local_server):
        """Test PDB structure retrieval from AlphaFold in local mode."""
        mock_result = "HEADER    ALPHAFOLD PREDICTION                     01-JAN-22   AF12\nATOM      1  N   MET A   1      12.345  67.890  12.345  1.00 90.00           N"
        
        with patch('gget.pdb', return_value=mock_result):
            response = await local_server.get_pdb_structure_local(
                pdb_id="AF-P04637-F1",
                resource="alphafold",
                format="pdb"
            )
            
        assert response["success"] is True
        assert "structure_AF-P04637-F1_alphafold" in response["path"]
        assert response["path"].endswith(".pdb")

    @pytest.mark.asyncio
    async def test_get_pdb_structure_local_custom_path(self, local_server):
        """Test PDB structure retrieval with custom output path."""
        mock_result = "HEADER    TEST STRUCTURE\nATOM      1  N   MET A   1"
        
        with patch('gget.pdb', return_value=mock_result):
            response = await local_server.get_pdb_structure_local(
                pdb_id="1ABC",
                output_path="my_structure",
                format="pdb"
            )
            
        assert response["success"] is True
        assert "my_structure.pdb" in response["path"]

    @pytest.mark.asyncio
    async def test_alphafold_predict_local_success(self, local_server):
        """Test successful AlphaFold prediction in local mode."""
        mock_result = "HEADER    ALPHAFOLD PREDICTION\nATOM      1  N   MET A   1      0.000   0.000   0.000  1.00 95.00           N"
        
        with patch('gget.alphafold', return_value=mock_result):
            response = await local_server.alphafold_predict_local(
                sequence="MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP",
                format="pdb"
            )
            
        assert response["success"] is True
        assert response["format"] == "pdb"
        assert response["path"] is not None
        assert Path(response["path"]).exists()
        assert "alphafold_prediction_" in response["path"]
        assert response["path"].endswith(".pdb")

    @pytest.mark.asyncio
    async def test_alphafold_predict_local_custom_path(self, local_server):
        """Test AlphaFold prediction with custom output path."""
        mock_result = "HEADER    ALPHAFOLD PREDICTION\nATOM      1  N   MET A   1"
        
        with patch('gget.alphafold', return_value=mock_result):
            response = await local_server.alphafold_predict_local(
                sequence="MVLSPADKTNVKAAW",
                output_path="my_prediction",
                format="pdb"
            )
            
        assert response["success"] is True
        assert "my_prediction.pdb" in response["path"]

    @pytest.mark.asyncio
    async def test_muscle_align_local_success(self, local_server):
        """Test successful MUSCLE alignment in local mode."""
        mock_result = ">seq1\nATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG\n>seq2\nATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAatGTCATTAATGCTATGCAGAAAATC\n>seq3\nATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG"
        
        with patch('gget.muscle', return_value=mock_result):
            response = await local_server.muscle_align_local(
                sequences=[
                    "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG",
                    "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATC",
                    "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG"
                ],
                format="fasta"
            )
            
        assert response["success"] is True
        assert response["format"] == "fasta"
        assert response["path"] is not None
        assert "muscle_alignment_3seqs" in response["path"]
        assert response["path"].endswith(".fasta")

    @pytest.mark.asyncio
    async def test_muscle_align_local_super5(self, local_server):
        """Test MUSCLE alignment with super5 algorithm."""
        mock_result = ">aligned_seq1\nATG-GAG\n>aligned_seq2\nATGGAG-"
        
        with patch('gget.muscle', return_value=mock_result):
            response = await local_server.muscle_align_local(
                sequences=["ATGGAG", "ATGGAG"],
                super5=True,
                format="fasta"
            )
            
        assert response["success"] is True
        assert "muscle_alignment_2seqs" in response["path"]
        assert response["path"].endswith(".fasta")

    @pytest.mark.asyncio
    async def test_muscle_align_local_custom_path(self, local_server):
        """Test MUSCLE alignment with custom output path."""
        mock_result = {"seq1": "ATGGAG", "seq2": "ATGGAG"}
        
        with patch('gget.muscle', return_value=mock_result):
            response = await local_server.muscle_align_local(
                sequences=["ATGGAG", "ATGGAG"],
                output_path="my_alignment",
                format="fasta"
            )
            
        assert response["success"] is True
        assert "my_alignment.fasta" in response["path"]

    @pytest.mark.asyncio
    async def test_diamond_align_local_success(self, local_server):
        """Test successful DIAMOND alignment in local mode."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {
            "query_id": ["seq1", "seq2"],
            "subject_id": ["hit1", "hit2"],
            "identity": [85.5, 92.3],
            "alignment_length": [100, 120],
            "e_value": [1e-20, 1e-25]
        }
        
        with patch('gget.diamond', return_value=mock_result):
            response = await local_server.diamond_align_local(
                sequences=["MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLAS"],
                reference="uniprot",
                format="json"
            )
            
        assert response["success"] is True
        assert response["format"] == "json"
        assert response["path"] is not None
        assert Path(response["path"]).exists()
        assert "diamond_alignment_" in response["path"]
        assert response["path"].endswith(".json")

    @pytest.mark.asyncio
    async def test_diamond_align_local_multiple_sequences(self, local_server):
        """Test DIAMOND alignment with multiple sequences."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"alignment": "data"}
        
        with patch('gget.diamond', return_value=mock_result):
            response = await local_server.diamond_align_local(
                sequences=[
                    "MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLAS",
                    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
                ],
                reference="uniprot",
                sensitivity="sensitive",
                threads=2,
                format="json"
            )
            
        assert response["success"] is True
        assert response["format"] == "json"

    @pytest.mark.asyncio
    async def test_diamond_align_local_custom_path(self, local_server):
        """Test DIAMOND alignment with custom output path."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"alignment": "data"}
        
        with patch('gget.diamond', return_value=mock_result):
            response = await local_server.diamond_align_local(
                sequences="MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLAS",
                reference="uniprot",
                output_path="my_diamond_results",
                format="json"
            )
            
        assert response["success"] is True
        assert "my_diamond_results.json" in response["path"]

    @pytest.mark.asyncio
    async def test_local_mode_file_operations(self, local_server):
        """Test file operations and cleanup in local mode."""
        # Test that output directory exists
        assert local_server.output_dir.exists()
        assert local_server.transport_mode == "stdio-local"
        
        # Test _save_to_local_file helper
        test_data = {"gene1": "ATGGAGGAG", "gene2": "ATGGATTTAT"}
        result = local_server._save_to_local_file(test_data, "fasta", "test_sequences")
        
        assert result["success"] is True
        assert result["format"] == "fasta"
        assert Path(result["path"]).exists()
        assert "test_sequences.fasta" in result["path"]

    @pytest.mark.asyncio
    async def test_local_mode_error_handling(self, local_server):
        """Test error handling in local mode functions."""
        with patch('gget.seq', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await local_server.get_sequences_local(
                    ensembl_ids=["INVALID"],
                    translate=False
                )

    @pytest.mark.asyncio
    async def test_local_mode_different_formats(self, local_server):
        """Test different output formats in local mode."""
        # Test JSON format
        test_data = {"key": "value"}
        result = local_server._save_to_local_file(test_data, "json", "test_json")
        assert result["success"] is True
        assert result["format"] == "json"
        
        # Test TSV format
        result = local_server._save_to_local_file(test_data, "tsv", "test_tsv")
        assert result["success"] is True
        assert result["format"] == "tsv"
        
        # Test default format
        result = local_server._save_to_local_file(test_data, "unknown", "test_default")
        assert result["success"] is True
        assert result["format"] == "unknown"

    def test_server_initialization_local_mode(self):
        """Test server initialization in local mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            server = GgetMCP(transport_mode="stdio-local", output_dir=temp_dir)
            assert server.transport_mode == "stdio-local"
            assert server.output_dir == Path(temp_dir)
            assert server.output_dir.exists()

    def test_server_initialization_default_output_dir(self):
        """Test server initialization with default output directory."""
        server = GgetMCP(transport_mode="stdio-local")
        assert server.transport_mode == "stdio-local"
        assert server.output_dir == Path.cwd() / "gget_output" 