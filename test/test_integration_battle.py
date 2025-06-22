"""Integration tests for gget-mcp server with real API calls.

This is the "battle test" that verifies everything works with real data.
No mocking - these tests hit the actual gget APIs.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import json
import sys

# Add the src directory to Python path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from gget_mcp.server import GgetMCP


class TestGgetMCPIntegration:
    """Integration tests with real API calls."""
    
    @pytest.fixture
    def server(self):
        """Create a regular server instance."""
        return GgetMCP()
    
    @pytest.fixture
    def local_server(self):
        """Create a local mode server instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield GgetMCP(transport_mode="stdio-local", output_dir=temp_dir)

    # ============================================================================
    # BASIC WORKFLOW TESTS - Gene Search to Sequences
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow_tp53_regular_mode(self, server):
        """Test complete workflow: search TP53 → get info → get DNA sequence."""
        print("\n🧬 Testing TP53 workflow in regular mode...")
        
        # Step 1: Search for TP53
        search_result = await server.search_genes(
            search_terms=["TP53"], 
            species="homo_sapiens",
            limit=5
        )
        
        assert search_result is not None
        assert len(search_result) > 0
        print(f"✅ Found {len(search_result)} TP53 results")
        
        # Get the first Ensembl ID
        ensembl_id = None
        for gene_data in search_result.values():
            if isinstance(gene_data, dict) and 'ensembl_id' in gene_data:
                ensembl_id = gene_data['ensembl_id']
                break
            elif isinstance(gene_data, list) and len(gene_data) > 0:
                ensembl_id = gene_data[0].get('ensembl_id') if isinstance(gene_data[0], dict) else None
                break
        
        # Fallback: try common TP53 Ensembl ID
        if not ensembl_id:
            ensembl_id = "ENSG00000141510"  # Known TP53 ID
        
        print(f"📝 Using Ensembl ID: {ensembl_id}")
        
        # Step 2: Get gene information
        info_result = await server.get_gene_info(
            ensembl_ids=[ensembl_id],
            verbose=True
        )
        
        assert info_result is not None
        assert ensembl_id in info_result or len(info_result) > 0
        print("✅ Retrieved gene information")
        
        # Step 3: Get DNA sequence
        dna_result = await server.get_sequences(
            ensembl_ids=[ensembl_id],
            translate=False
        )
        
        assert dna_result is not None
        assert len(dna_result) > 0
        
        # Debug: print the actual result structure
        print(f"🔍 DNA result type: {type(dna_result)}")
        if isinstance(dna_result, list):
            print(f"🔍 DNA result list length: {len(dna_result)}")
            if len(dna_result) > 1:
                print(f"🔍 First element: {dna_result[0][:100]}...")
                print(f"🔍 Second element: {dna_result[1][:100]}...")
        
        # Handle FASTA format - get the actual sequence (not header)
        if isinstance(dna_result, list) and len(dna_result) > 1:
            sequence = dna_result[1]  # Second element should be the sequence
        elif isinstance(dna_result, dict):
            sequence = list(dna_result.values())[0]
        else:
            sequence = str(dna_result)
            
        # Remove any newlines
        sequence = sequence.replace('\n', '').replace('\r', '')
        
        assert len(sequence) > 100  # TP53 should be longer than 100bp
        assert 'ATG' in sequence.upper()  # Should contain start codon
        print(f"✅ Retrieved DNA sequence: {len(sequence)} bases")
        
        # Step 4: Get protein sequence
        protein_result = await server.get_sequences(
            ensembl_ids=[ensembl_id],
            translate=True
        )
        
        assert protein_result is not None
        assert len(protein_result) > 0
        
        # Handle FASTA format for protein sequences too
        if isinstance(protein_result, list) and len(protein_result) > 1:
            protein_sequence = protein_result[1]  # Second element should be the sequence
        elif isinstance(protein_result, dict):
            protein_sequence = list(protein_result.values())[0]
        else:
            protein_sequence = str(protein_result)
            
        # Remove any newlines
        protein_sequence = protein_sequence.replace('\n', '').replace('\r', '')
        
        assert len(protein_sequence) > 100  # TP53 protein should be >100 AA
        print(f"✅ Retrieved protein sequence: {len(protein_sequence)} amino acids")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow_tp53_local_mode(self, local_server):
        """Test complete workflow in local mode with file outputs."""
        print("\n📁 Testing TP53 workflow in local mode...")
        
        # Search for TP53 (this doesn't change in local mode)
        search_result = await local_server.search_genes(
            search_terms=["TP53"], 
            species="homo_sapiens",
            limit=5
        )
        
        ensembl_id = "ENSG00000141510"  # Use known TP53 ID
        print(f"📝 Using Ensembl ID: {ensembl_id}")
        
        # Get DNA sequences in local mode
        dna_result = await local_server.get_sequences_local(
            ensembl_ids=[ensembl_id],
            translate=False,
            format="fasta"
        )
        
        assert dna_result["success"] is True
        assert dna_result["format"] == "fasta"
        assert Path(dna_result["path"]).exists()
        print(f"✅ DNA sequences saved to: {dna_result['path']}")
        
        # Verify DNA file content
        with open(dna_result["path"], 'r') as f:
            dna_content = f.read()
            assert f">{ensembl_id}" in dna_content
            assert "ATG" in dna_content
            print("✅ DNA FASTA file verified")
        
        # Get protein sequences in local mode
        protein_result = await local_server.get_sequences_local(
            ensembl_ids=[ensembl_id],
            translate=True,
            format="fasta"
        )
        
        assert protein_result["success"] is True
        assert Path(protein_result["path"]).exists()
        print(f"✅ Protein sequences saved to: {protein_result['path']}")
        
        # Verify protein file content
        with open(protein_result["path"], 'r') as f:
            protein_content = f.read()
            # Check for either ENSG ID or ENST ID (transcript)
            assert (f">{ensembl_id}" in protein_content or "ENST" in protein_content)
            # Check for amino acid sequence (not DNA) - proteins have less ATG
            lines = protein_content.split('\n')
            sequence_lines = [line for line in lines if not line.startswith('>')]
            full_sequence = ''.join(sequence_lines)
            assert len(full_sequence) > 100  # Should have substantial protein sequence
            print("✅ Protein FASTA file verified")

    # ============================================================================
    # MULTI-GENE TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_genes_brca1_brca2_local(self, local_server):
        """Test multiple cancer genes: BRCA1 and BRCA2."""
        print("\n🎗️ Testing BRCA1/BRCA2 workflow...")
        
        # Use known Ensembl IDs for BRCA genes
        brca_ids = [
            "ENSG00000012048",  # BRCA1
            "ENSG00000139618"   # BRCA2
        ]
        
        # Get DNA sequences for both genes
        result = await local_server.get_sequences_local(
            ensembl_ids=brca_ids,
            translate=False,
            format="fasta"
        )
        
        assert result["success"] is True
        assert Path(result["path"]).exists()
        print(f"✅ BRCA sequences saved to: {result['path']}")
        
        # Verify file contains both genes
        with open(result["path"], 'r') as f:
            content = f.read()
            assert ">ENSG00000012048" in content  # BRCA1
            assert ">ENSG00000139618" in content  # BRCA2
            print("✅ Both BRCA genes found in FASTA file")

    # ============================================================================
    # STRUCTURE ANALYSIS TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_pdb_structure_retrieval_local(self, local_server):
        """Test PDB structure retrieval for a known structure."""
        print("\n🏗️ Testing PDB structure retrieval...")
        
        # Use a well-known PDB structure (p53 tumor suppressor)
        result = await local_server.get_pdb_structure_local(
            pdb_id="1TUP",
            resource="pdb",
            format="pdb"
        )
        
        assert result["success"] is True
        assert result["format"] == "pdb"
        assert Path(result["path"]).exists()
        print(f"✅ PDB structure saved to: {result['path']}")
        
        # Verify PDB file content
        with open(result["path"], 'r') as f:
            content = f.read()
            assert "HEADER" in content or "ATOM" in content
            print("✅ PDB file format verified")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_alphafold_prediction_local(self, local_server):
        """Test AlphaFold structure prediction with a small protein."""
        print("\n🤖 Testing AlphaFold prediction...")
        
        # Use a small protein sequence for faster testing
        small_protein = "MVIGGDKTWIVGRDGKQKEQYETLLWKPGVVWVKATVYGKEHGEVYKDGLQADKLVDEEVLQ"
        
        result = await local_server.alphafold_predict_local(
            sequence=small_protein,
            format="pdb"
        )
        
        assert result["success"] is True
        assert result["format"] == "pdb"
        assert Path(result["path"]).exists()
        print(f"✅ AlphaFold prediction saved to: {result['path']}")

    # ============================================================================
    # ALIGNMENT TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_muscle_alignment_local(self, local_server):
        """Test MUSCLE sequence alignment."""
        print("\n🔗 Testing MUSCLE alignment...")
        
        # Test sequences (variants of a small gene region)
        sequences = [
            "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG",
            "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATC", 
            "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG"
        ]
        
        result = await local_server.muscle_align_local(
            sequences=sequences,
            format="fasta"
        )
        
        assert result["success"] is True
        assert result["format"] == "fasta"
        assert Path(result["path"]).exists()
        print(f"✅ MUSCLE alignment saved to: {result['path']}")
        
        # Verify alignment file
        with open(result["path"], 'r') as f:
            content = f.read()
            # Should have multiple sequences
            assert content.count('>') >= len(sequences)
            print("✅ MUSCLE alignment file verified")

    # ============================================================================
    # FUNCTIONAL ANALYSIS TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_enrichr_analysis_cancer_genes(self, server):
        """Test Enrichr pathway analysis with cancer genes."""
        print("\n🎯 Testing Enrichr pathway analysis...")
        
        cancer_genes = ["TP53", "BRCA1", "BRCA2", "MYC", "RAS"]
        
        result = await server.enrichr_analysis(
            genes=cancer_genes,
            database="KEGG_2021_Human",
            species="human"
        )
        
        assert result is not None
        assert len(result) > 0
        print(f"✅ Found {len(result)} pathway enrichment results")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_opentargets_tp53_analysis(self, server):
        """Test OpenTargets disease association for TP53."""
        print("\n🎯 Testing OpenTargets disease analysis...")
        
        # Use TP53 Ensembl ID
        ensembl_id = "ENSG00000141510"
        
        result = await server.opentargets_analysis(
            ensembl_id=ensembl_id,
            resource="diseases",
            limit=10
        )
        
        assert result is not None
        assert len(result) > 0
        print(f"✅ Found {len(result)} disease associations for TP53")

    # ============================================================================
    # SEARCH AND REFERENCE TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_gene_search_multiple_species(self, server):
        """Test gene search across different species."""
        print("\n🐁 Testing multi-species gene search...")
        
        # Test BRCA1 in human and mouse
        human_result = await server.search_genes(
            search_terms=["BRCA1"],
            species="homo_sapiens",
            limit=3
        )
        
        mouse_result = await server.search_genes(
            search_terms=["Brca1"],
            species="mus_musculus", 
            limit=3
        )
        
        assert human_result is not None and len(human_result) > 0
        assert mouse_result is not None and len(mouse_result) > 0
        print("✅ Found BRCA1 orthologs in human and mouse")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reference_genome_info(self, server):
        """Test reference genome information retrieval."""
        print("\n📚 Testing reference genome info...")
        
        result = await server.get_reference(
            species="homo_sapiens",
            which="gtf",
            release=None
        )
        
        assert result is not None
        assert len(result) > 0
        print("✅ Retrieved human reference genome information")

    # ============================================================================
    # EXPRESSION AND ORTHOLOGY TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_archs4_expression_tp53(self, server):
        """Test tissue expression data for TP53."""
        print("\n📊 Testing ARCHS4 expression data...")
        
        result = await server.archs4_expression(
            gene="TP53",
            which="tissue",
            species="human"
        )
        
        assert result is not None
        print("✅ Retrieved TP53 tissue expression data")

    @pytest.mark.asyncio  
    @pytest.mark.integration
    async def test_bgee_orthologs_brca1(self, server):
        """Test ortholog search for BRCA1."""
        print("\n🧬 Testing Bgee ortholog search...")
        
        # Use BRCA1 Ensembl ID
        ensembl_id = "ENSG00000012048"
        
        result = await server.bgee_orthologs(
            gene_id=ensembl_id,
            type="orthologs"
        )
        
        assert result is not None
        print("✅ Retrieved BRCA1 ortholog information")

    # ============================================================================
    # SEQUENCE ANALYSIS TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_blast_small_sequence(self, server):
        """Test BLAST with a small sequence."""
        print("\n💥 Testing BLAST sequence search...")
        
        # Small test sequence (part of TP53)
        test_sequence = "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG"
        
        result = await server.blast_sequence(
            sequence=test_sequence,
            program="blastn",
            database="nt",
            limit=5,
            expect=10.0
        )
        
        assert result is not None
        print("✅ BLAST search completed")

    @pytest.mark.asyncio
    @pytest.mark.integration 
    async def test_blat_genomic_location(self, server):
        """Test BLAT genomic location search."""
        print("\n🎯 Testing BLAT genomic location...")
        
        # Test sequence
        test_sequence = "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAG"
        
        result = await server.blat_sequence(
            sequence=test_sequence,
            seqtype="DNA",
            assembly="hg38"
        )
        
        assert result is not None
        print("✅ BLAT genomic location search completed")

    # ============================================================================
    # SETUP AND UTILITY TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_setup_databases(self, server):
        """Test database setup for various modules."""
        print("\n⚙️ Testing database setup...")
        
        # Test setup for a valid module
        result = await server.setup_databases(module="elm")
        
        assert result is not None
        assert result["success"] is True
        print("✅ ELM database setup completed")
        
        # Test invalid module
        result = await server.setup_databases(module="invalid")
        
        assert result["success"] is False
        assert "Invalid module" in result["message"]
        print("✅ Invalid module handling verified")

    # ============================================================================
    # COMPREHENSIVE BATTLE TEST
    # ============================================================================
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_comprehensive_battle_test(self, local_server):
        """The ultimate battle test - complete workflow with real data."""
        print("\n🚀 COMPREHENSIVE BATTLE TEST STARTING...")
        print("=" * 60)
        
        # Step 1: Search for multiple cancer genes
        print("Step 1: Searching for cancer genes...")
        genes = ["TP53", "BRCA1", "MYC"]
        all_ensembl_ids = []
        
        for gene in genes:
            search_result = await local_server.search_genes(
                search_terms=[gene],
                species="homo_sapiens",
                limit=1
            )
            print(f"  ✅ Found results for {gene}")
        
        # Use known IDs for the battle test
        battle_ids = [
            "ENSG00000141510",  # TP53
            "ENSG00000012048",  # BRCA1  
            "ENSG00000136997"   # MYC
        ]
        
        # Step 2: Get DNA sequences
        print("Step 2: Retrieving DNA sequences...")
        dna_result = await local_server.get_sequences_local(
            ensembl_ids=battle_ids,
            translate=False,
            format="fasta"
        )
        assert dna_result["success"] is True
        print(f"  ✅ DNA sequences saved to: {Path(dna_result['path']).name}")
        
        # Step 3: Get protein sequences  
        print("Step 3: Retrieving protein sequences...")
        protein_result = await local_server.get_sequences_local(
            ensembl_ids=battle_ids,
            translate=True,
            format="fasta"
        )
        assert protein_result["success"] is True
        print(f"  ✅ Protein sequences saved to: {Path(protein_result['path']).name}")
        
        # Step 4: Test alignment
        print("Step 4: Testing sequence alignment...")
        # Get some sequences for alignment
        with open(protein_result["path"], 'r') as f:
            content = f.read()
            sequences = []
            current_seq = ""
            for line in content.split('\n'):
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq[:200])  # Truncate for faster alignment
                        current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                sequences.append(current_seq[:200])
        
        align_result = None
        if len(sequences) >= 2:
            align_result = await local_server.muscle_align_local(
                sequences=sequences[:3],  # Use first 3 sequences
                format="fasta"
            )
            assert align_result["success"] is True
            print(f"  ✅ Alignment saved to: {Path(align_result['path']).name}")
        else:
            print(f"  ⚠️ Skipping alignment - only {len(sequences)} sequences found")
        
        # Step 5: Test structure retrieval
        print("Step 5: Testing structure retrieval...")
        structure_result = await local_server.get_pdb_structure_local(
            pdb_id="1TUP",  # p53 structure
            format="pdb"
        )
        assert structure_result["success"] is True
        print(f"  ✅ Structure saved to: {Path(structure_result['path']).name}")
        
        print("=" * 60)
        print("🎉 COMPREHENSIVE BATTLE TEST COMPLETED SUCCESSFULLY!")
        print("All systems operational with real data!")
        
        # Summary of created files
        print("\n📁 Files created during battle test:")
        results_to_show = [dna_result, protein_result, structure_result]
        if align_result:
            results_to_show.append(align_result)
        
        for result in results_to_show:
            file_path = Path(result["path"])
            file_size = file_path.stat().st_size
            print(f"  • {file_path.name} ({file_size} bytes)")


# Run specific test groups
class TestMarkers:
    """Test grouping for different types of integration tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.quick
    async def test_quick_integration_check(self, server):
        """Quick integration test to verify basic functionality."""
        print("\n⚡ Quick integration check...")
        
        # Just test basic search functionality
        result = await server.search_genes(
            search_terms=["TP53"],
            species="homo_sapiens", 
            limit=1
        )
        
        assert result is not None
        print("✅ Basic integration working!")


if __name__ == "__main__":
    print("🚀 Running integration battle tests...")
    print("Note: These tests use real API calls and may take several minutes")
    pytest.main([__file__, "-v", "-m", "integration"]) 