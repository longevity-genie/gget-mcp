"""Tests for gget-mcp server functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from gget_mcp.server import GgetMCP


class TestGgetMCP:
    """Test cases for GgetMCP server."""
    
    @pytest.fixture
    def server(self):
        """Create a test server instance."""
        return GgetMCP()
    
    def test_server_initialization(self, server):
        """Test that server initializes correctly."""
        assert server.prefix == "gget_"
        assert hasattr(server, '_register_gget_tools')
    
    @pytest.mark.asyncio
    async def test_search_genes_success(self, server):
        """Test successful gene search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"gene1": "data1"}
        mock_result.__len__ = Mock(return_value=1)
        
        with patch('gget.search', return_value=mock_result):
            response = await server.search_genes(
                search_terms=["BRCA1"], 
                species="homo_sapiens"
            )
            
        assert response == {"gene1": "data1"}
    
    @pytest.mark.asyncio
    async def test_search_genes_failure(self, server):
        """Test gene search failure handling."""
        with patch('gget.search', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await server.search_genes(
                    search_terms=["INVALID"], 
                    species="homo_sapiens"
                )
    
    @pytest.mark.asyncio
    async def test_get_gene_info_success(self, server):
        """Test successful gene info retrieval."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"ENSG123": {"name": "Test Gene"}}
        
        with patch('gget.info', return_value=mock_result):
            response = await server.get_gene_info(
                ensembl_ids=["ENSG123"]
            )
            
        assert response == {"ENSG123": {"name": "Test Gene"}}
    
    @pytest.mark.asyncio
    async def test_get_sequences_success(self, server):
        """Test successful sequence retrieval."""
        mock_result = {"ENSG123": "ATGCGATCG"}
        
        with patch('gget.seq', return_value=mock_result):
            response = await server.get_sequences(
                ensembl_ids=["ENSG123"],
                translate=False
            )
            
        assert response == mock_result
    
    @pytest.mark.asyncio
    async def test_blast_sequence_success(self, server):
        """Test successful BLAST search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"hit1": "data1"}
        mock_result.__len__ = Mock(return_value=1)
        
        with patch('gget.blast', return_value=mock_result):
            response = await server.blast_sequence(
                sequence="ATGCGATCG",
                program="blastn"
            )
            
        assert response == {"hit1": "data1"}
    
    @pytest.mark.asyncio 
    async def test_enrichr_analysis_success(self, server):
        """Test successful enrichment analysis."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"pathway1": "significant"}
        
        with patch('gget.enrichr', return_value=mock_result):
            response = await server.enrichr_analysis(
                genes=["BRCA1", "TP53"],
                database="KEGG_2021_Human"
            )
            
        assert response == {"pathway1": "significant"}

    @pytest.mark.asyncio
    async def test_diamond_align_success(self, server):
        """Test successful DIAMOND alignment."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"alignment": "data"}
        
        with patch('gget.diamond', return_value=mock_result):
            response = await server.diamond_align(
                sequences="MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLAS",
                reference="MSSSSWLLLSLVEVTAAQSTIEQQAKTFLDKFHEAEDLFYQSLLAS"
            )
            
        assert response == {"alignment": "data"}

    @pytest.mark.asyncio
    async def test_diamond_align_failure(self, server):
        """Test DIAMOND alignment failure handling."""
        with patch('gget.diamond', side_effect=Exception("DIAMOND Error")):
            with pytest.raises(Exception, match="DIAMOND Error"):
                await server.diamond_align(
                    sequences="INVALID",
                    reference="INVALID"
                )

    @pytest.mark.asyncio
    async def test_bgee_orthologs_success(self, server):
        """Test successful Bgee ortholog search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"orthologs": ["gene1", "gene2"]}
        
        with patch('gget.bgee', return_value=mock_result):
            response = await server.bgee_orthologs(
                gene_id="ENSG00000012048",  # Fixed parameter name
                type="orthologs"
            )
            
        assert response == {"orthologs": ["gene1", "gene2"]}

    @pytest.mark.asyncio
    async def test_bgee_orthologs_failure(self, server):
        """Test Bgee ortholog search failure handling."""
        with patch('gget.bgee', side_effect=Exception("Bgee Error")):
            with pytest.raises(Exception, match="Bgee Error"):
                await server.bgee_orthologs(
                    gene_id="INVALID",  # Fixed parameter name
                    type="orthologs"
                )

    @pytest.mark.asyncio
    async def test_elm_analysis_success(self, server):
        """Test successful ELM analysis."""
        mock_ortholog_df = Mock()
        mock_ortholog_df.to_dict.return_value = {"ortholog": "data"}
        mock_regex_df = Mock()
        mock_regex_df.to_dict.return_value = {"regex": "data"}
        
        with patch('gget.elm', return_value=(mock_ortholog_df, mock_regex_df)):
            response = await server.elm_analysis(
                sequence="LIAQSIGQASFV",
                sensitivity="very-sensitive"
            )
            
        expected_data = {
            "ortholog_df": {"ortholog": "data"},
            "regex_df": {"regex": "data"}
        }
        assert response == expected_data

    @pytest.mark.asyncio
    async def test_elm_analysis_uniprot_success(self, server):
        """Test successful ELM analysis with UniProt ID."""
        mock_ortholog_df = Mock()
        mock_ortholog_df.to_dict.return_value = {"ortholog": "data"}
        mock_regex_df = Mock()
        mock_regex_df.to_dict.return_value = {"regex": "data"}
        
        with patch('gget.elm', return_value=(mock_ortholog_df, mock_regex_df)):
            response = await server.elm_analysis(
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
    async def test_elm_analysis_failure(self, server):
        """Test ELM analysis failure handling."""
        with patch('gget.elm', side_effect=Exception("ELM Error")):
            with pytest.raises(Exception, match="ELM Error"):
                await server.elm_analysis(
                    sequence="INVALID"
                )

    @pytest.mark.asyncio
    async def test_mutate_sequences_success(self, server):
        """Test successful sequence mutation."""
        mock_result = ["ATCTCTAAGCT"]
        
        with patch('gget.mutate', return_value=mock_result):
            response = await server.mutate_sequences(
                sequences="ATCGCTAAGCT",
                mutations="c.4G>T"
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_mutate_sequences_multiple(self, server):
        """Test successful mutation of multiple sequences."""
        mock_result = ["ATCTCTAAGCT", "GATCTA"]
        
        with patch('gget.mutate', return_value=mock_result):
            response = await server.mutate_sequences(
                sequences=["ATCGCTAAGCT", "TAGCTA"],
                mutations=["c.4G>T", "c.1_3inv"],
                k=30
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_mutate_sequences_failure(self, server):
        """Test sequence mutation failure handling."""
        with patch('gget.mutate', side_effect=Exception("Mutation Error")):
            with pytest.raises(Exception, match="Mutation Error"):
                await server.mutate_sequences(
                    sequences="INVALID",
                    mutations="invalid_mutation"
                )

    @pytest.mark.asyncio
    async def test_opentargets_analysis_success(self, server):
        """Test successful Open Targets analysis."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"diseases": ["disease1", "disease2"]}
        
        with patch('gget.opentargets', return_value=mock_result):
            response = await server.opentargets_analysis(
                ensembl_id="ENSG00000012048",  # Fixed parameter name
                resource="diseases"
            )
            
        assert response == {"diseases": ["disease1", "disease2"]}

    @pytest.mark.asyncio
    async def test_opentargets_analysis_drugs(self, server):
        """Test successful Open Targets drug analysis."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"drugs": ["drug1", "drug2"]}
        
        with patch('gget.opentargets', return_value=mock_result):
            response = await server.opentargets_analysis(
                ensembl_id="ENSG00000012048",  # Fixed parameter name
                resource="drugs",
                limit=50
            )
            
        assert response == {"drugs": ["drug1", "drug2"]}

    @pytest.mark.asyncio
    async def test_opentargets_analysis_failure(self, server):
        """Test Open Targets analysis failure handling."""
        with patch('gget.opentargets', side_effect=Exception("OpenTargets Error")):
            with pytest.raises(Exception, match="OpenTargets Error"):
                await server.opentargets_analysis(
                    ensembl_id="INVALID"  # Fixed parameter name
                )

    @pytest.mark.asyncio
    async def test_setup_databases_success(self, server):
        """Test successful database setup."""
        mock_result = "Setup completed"
        
        with patch('gget.setup', return_value=mock_result):
            response = await server.setup_databases(module="elm")
            
        expected_response = {
            "data": mock_result,
            "success": True,
            "message": "Setup completed for elm module"
        }
        assert response == expected_response

    @pytest.mark.asyncio
    async def test_setup_databases_invalid_module(self, server):
        """Test database setup with invalid module."""
        response = await server.setup_databases(module="invalid_module")
            
        assert response["success"] is False
        assert "Invalid module 'invalid_module'" in response["message"]
        assert "elm, cellxgene, alphafold" in response["message"]

    @pytest.mark.asyncio
    async def test_setup_databases_failure(self, server):
        """Test database setup failure handling."""
        with patch('gget.setup', side_effect=Exception("Setup Error")):
            with pytest.raises(Exception, match="Setup Error"):
                await server.setup_databases(module="elm")

    @pytest.mark.asyncio
    async def test_get_reference_success(self, server):
        """Test successful reference genome retrieval."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"genome": "data"}
        
        with patch('gget.ref', return_value=mock_result):
            response = await server.get_reference(
                species="homo_sapiens",
                which="dna"
            )
            
        assert response == {"genome": "data"}

    @pytest.mark.asyncio
    async def test_blat_sequence_success(self, server):
        """Test successful BLAT search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"location": "chr1:1000-2000"}
        
        with patch('gget.blat', return_value=mock_result):
            response = await server.blat_sequence(
                sequence="ATGCGATCG",
                seqtype="DNA",
                assembly="hg38"
            )
            
        assert response == {"location": "chr1:1000-2000"}

    @pytest.mark.asyncio
    async def test_muscle_align_success(self, server):
        """Test successful MUSCLE alignment."""
        mock_result = "aligned_sequences"
        
        with patch('gget.muscle', return_value=mock_result):
            response = await server.muscle_align(
                sequences=["ATGCGATCG", "ATGCGATCC"],
                super5=False
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_archs4_expression_success(self, server):
        """Test successful ARCHS4 expression query."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"expression": "data"}
        
        with patch('gget.archs4', return_value=mock_result):
            response = await server.archs4_expression(
                gene="BRCA1",
                which="tissue",
                species="human"
            )
            
        assert response == {"expression": "data"}

    @pytest.mark.asyncio
    async def test_get_pdb_structure_success(self, server):
        """Test successful PDB structure retrieval."""
        mock_result = "pdb_structure_data"
        
        with patch('gget.pdb', return_value=mock_result):
            response = await server.get_pdb_structure(
                pdb_id="1R42",
                resource="pdb"
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_alphafold_predict_success(self, server):
        """Test successful AlphaFold prediction."""
        mock_result = "alphafold_structure"
        
        with patch('gget.alphafold', return_value=mock_result):
            response = await server.alphafold_predict(
                sequence="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
            )
            
        assert response == mock_result

    @pytest.mark.asyncio
    async def test_cosmic_search_success(self, server):
        """Test successful COSMIC search."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"mutations": ["mut1", "mut2"]}
        
        with patch('gget.cosmic', return_value=mock_result):
            response = await server.cosmic_search(
                searchterm="BRCA1",  # This parameter is correct
                limit=100
            )
            
        assert response == {"mutations": ["mut1", "mut2"]}

    @pytest.mark.asyncio
    async def test_cellxgene_query_success(self, server):
        """Test successful CellxGene query."""
        mock_result = "cellxgene_data"
        
        with patch('gget.cellxgene', return_value=mock_result):
            response = await server.cellxgene_query(
                gene=["ACE2", "SLC5A1"],
                tissue=["lung"],
                cell_type=["mucus secreting cell"],
                species="homo_sapiens"
            )
            
        assert response == mock_result 