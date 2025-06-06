#!/usr/bin/env python3
"""
Basic usage example for gget-mcp server.

This script demonstrates how to create and interact with the gget MCP server
for testing purposes.
"""

import asyncio
from gget_mcp.server import GgetMCP

async def test_gget_functions():
    """Test basic gget functionality through the MCP server."""
    
    # Create server instance
    server = GgetMCP()
    print("✅ gget-mcp server created successfully")
    
    # Test 1: Search for genes
    print("\n🔍 Testing gene search...")
    try:
        search_result = await server.search_genes(
            search_terms=["BRCA1"],
            species="homo_sapiens",
            limit=5
        )
        print(f"Search result: {search_result.success}")
        if search_result.success:
            print(f"Message: {search_result.message}")
        else:
            print(f"Error: {search_result.message}")
    except Exception as e:
        print(f"Search failed: {e}")
    
    # Test 2: Get gene information
    print("\n📊 Testing gene info retrieval...")
    try:
        info_result = await server.get_gene_info(
            ensembl_ids=["ENSG00000012048"],  # BRCA1
            verbose=True
        )
        print(f"Info result: {info_result.success}")
        if info_result.success:
            print(f"Message: {info_result.message}")
        else:
            print(f"Error: {info_result.message}")
    except Exception as e:
        print(f"Info retrieval failed: {e}")
    
    # Test 3: Get sequences
    print("\n🧬 Testing sequence retrieval...")
    try:
        seq_result = await server.get_sequences(
            ensembl_ids=["ENSG00000012048"],  # BRCA1
            translate=False,
            isoforms=False
        )
        print(f"Sequence result: {seq_result.success}")
        if seq_result.success:
            print(f"Message: {seq_result.message}")
        else:
            print(f"Error: {seq_result.message}")
    except Exception as e:
        print(f"Sequence retrieval failed: {e}")
    
    # Test 4: BLAST a short sequence
    print("\n💥 Testing BLAST...")
    try:
        blast_result = await server.blast_sequence(
            sequence="ATGAAAGAAACCGCCGTCCTGTCAGCCCTGGCC",
            program="blastn",
            database="nt",
            limit=5
        )
        print(f"BLAST result: {blast_result.success}")
        if blast_result.success:
            print(f"Message: {blast_result.message}")
        else:
            print(f"Error: {blast_result.message}")
    except Exception as e:
        print(f"BLAST failed: {e}")
    
    # Test 5: Get reference genome info
    print("\n📚 Testing reference genome info...")
    try:
        ref_result = await server.get_reference(
            species="homo_sapiens",
            which="dna"
        )
        print(f"Reference result: {ref_result.success}")
        if ref_result.success:
            print(f"Message: {ref_result.message}")
        else:
            print(f"Error: {ref_result.message}")
    except Exception as e:
        print(f"Reference retrieval failed: {e}")
    
    print("\n✨ Basic usage test completed!")

def main():
    """Main function to run the test."""
    print("🚀 Starting gget-mcp basic usage test...")
    asyncio.run(test_gget_functions())

if __name__ == "__main__":
    main() 