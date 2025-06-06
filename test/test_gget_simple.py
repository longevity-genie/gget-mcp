import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the src directory to Python path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from gget_mcp.server import GgetMCP


class TestGgetMCPBasic:
    """Basic tests to verify functionality after removing GgetResponse wrapper."""
    
    @pytest.fixture
    def server(self):
        """Create a GgetMCP server instance for testing."""
        return GgetMCP()
    
    def test_server_initialization(self, server):
        """Test that the server initializes correctly."""
        assert server is not None
        assert hasattr(server, 'search_genes')
        assert hasattr(server, 'get_gene_info')
        assert hasattr(server, 'get_sequences')
    
    @pytest.mark.asyncio
    async def test_search_genes_returns_data_directly(self, server):
        """Test that search_genes returns data directly, not wrapped."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"gene1": "data1"}
        
        with patch('gget.search', return_value=mock_result):
            response = await server.search_genes(
                search_terms=["BRCA1"], 
                species="homo_sapiens"
            )
            
        # Should return data directly, not wrapped in GgetResponse
        assert response == {"gene1": "data1"}
        assert not hasattr(response, 'success')  # No wrapper attributes
        assert not hasattr(response, 'message')
    
    @pytest.mark.asyncio
    async def test_get_gene_info_returns_data_directly(self, server):
        """Test that get_gene_info returns data directly."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {"ENSG123": {"name": "Test Gene"}}
        
        with patch('gget.info', return_value=mock_result):
            response = await server.get_gene_info(
                ensembl_ids=["ENSG123"]
            )
            
        # Should return data directly
        assert response == {"ENSG123": {"name": "Test Gene"}}
    
    @pytest.mark.asyncio
    async def test_get_sequences_returns_data_directly(self, server):
        """Test that get_sequences returns data directly."""
        mock_result = {"ENSG123": "ATGCGATCG"}
        
        with patch('gget.seq', return_value=mock_result):
            response = await server.get_sequences(
                ensembl_ids=["ENSG123"],
                translate=False
            )
            
        # Should return data directly
        assert response == mock_result
    
    @pytest.mark.asyncio
    async def test_exceptions_propagate_naturally(self, server):
        """Test that exceptions propagate naturally without wrapper."""
        with patch('gget.search', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await server.search_genes(
                    search_terms=["INVALID"], 
                    species="homo_sapiens"
                ) 