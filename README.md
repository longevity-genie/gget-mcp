# gget-mcp

[![Tests](https://github.com/longevity-genie/gget-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/longevity-genie/gget-mcp/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/gget-mcp.svg)](https://badge.fury.io/py/gget-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

MCP (Model Context Protocol) server for the [gget](https://github.com/pachterlab/gget) bioinformatics library.

![gget-mcp Example](images/screenshot_example.png)

This server implements the Model Context Protocol (MCP) for gget, providing a standardized interface for accessing powerful bioinformatics tools and databases. MCP enables AI assistants and agents to perform complex genomics queries through structured interfaces.

The gget bioinformatics toolkit provides efficient querying of genomic databases and includes functions for:
- Gene and sequence information retrieval
- BLAST and sequence alignment
- Protein structure prediction
- Gene expression analysis
- Functional enrichment analysis
- And much more!

## 🚀 Quick Example: What Can You Do?

<details>
<summary>See how the gget MCP server works in practice with AI assistants</summary>

**👤 User:** "Analyze the TP53 gene - find its details and protein sequence, then check for cancer mutations"

**🤖 AI Assistant:** I'll help you analyze TP53 step by step:

*[Uses gget_search]* → Found TP53 (ENSG00000141510) on chromosome 17  
*[Uses gget_info]* → TP53 is a protein-coding tumor suppressor gene  
*[Uses gget_seq with translate=True]* → Retrieved 393 amino acid protein sequence  
*[Uses gget_cosmic]* → Found hotspot mutations R175H, R248Q, R273H in multiple cancers

**👤 User:** "Now predict the 3D structure of this protein sequence: MSKGEELFTGVVPILVEL..."

**🤖 AI Assistant:** Let me analyze this sequence:

*[Uses gget_blast]* → Identified as Green Fluorescent Protein (GFP)  
*[Uses gget_alphafold]* → Predicted classic beta-barrel structure with 90%+ confidence

**👤 User:** "Perform pathway enrichment for DNA repair genes: TP53, BRCA1, BRCA2, ATM"

**🤖 AI Assistant:** *[Uses gget_enrichr]* → Top pathways: DNA Damage Response, Homologous Recombination, Cell Cycle Checkpoints, p53 Signaling

**See how powerful this is?** You can now ask natural language questions about genes, sequences, mutations, and pathways - the AI does the complex bioinformatics work for you!

</details>

## About MCP (Model Context Protocol)

<details>
<summary>Learn more about the Model Context Protocol</summary>

MCP is a protocol that bridges the gap between AI systems and specialized domain tools. It enables:

- **Structured Access**: Direct connection to powerful bioinformatics tools and databases
- **Natural Language Queries**: Simplified interaction with complex genomics workflows
- **Type Safety**: Strong typing and validation through FastMCP
- **AI Integration**: Seamless integration with AI assistants and agents

If you want to understand more about what the Model Context Protocol is and how to use it more efficiently, you can take the [DeepLearning AI Course](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/) or search for MCP videos on YouTube.

</details>

## Available Tools

<details>
<summary>Comprehensive bioinformatics functionality through gget</summary>

### Gene Information & Search
- **`gget_search`**: Find Ensembl IDs associated with search terms
- **`gget_info`**: Fetch detailed information for Ensembl IDs
- **`gget_seq`**: Retrieve nucleotide or amino acid sequences
- **`gget_ref`**: Get reference genome information from Ensembl

### Sequence Analysis
- **`gget_blast`**: BLAST nucleotide or amino acid sequences
- **`gget_blat`**: Find genomic locations of sequences
- **`gget_muscle`**: Align multiple sequences

### Expression & Functional Analysis
- **`gget_archs4`**: Get gene expression data from ARCHS4
- **`gget_enrichr`**: Perform gene set enrichment analysis

### Protein Structure & Function
- **`gget_pdb`**: Fetch protein structure data from PDB
- **`gget_alphafold`**: Predict protein structure using AlphaFold

### Cancer & Mutation Analysis
- **`gget_cosmic`**: Search COSMIC database for cancer mutations

### Single-cell Analysis
- **`gget_cellxgene`**: Query single-cell RNA-seq data from CellxGene

</details>

## Quick Start

<details>
<summary>Installing uv (optional - uvx can auto-install)</summary>

```bash
# Download and install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
uvx --version
```

uvx is a very nice tool that can run a python package installing it if needed.

</details>

### Running with uvx

You can run the gget-mcp server directly using uvx without cloning the repository:

```bash
# Run the server in streamed http mode (default)
uvx gget-mcp
```

<details>
<summary>Other uvx modes (STDIO, HTTP, SSE)</summary>

#### STDIO Mode (for MCP clients that require stdio, can be useful when you want to save files)

```bash
# Or explicitly specify stdio mode
uvx gget-mcp stdio
```

#### HTTP Mode (Web Server)
```bash
# Run the server in streamable HTTP mode on default (3002) port
uvx gget-mcp server

# Run on a specific port
uvx gget-mcp server --port 8000
```

#### SSE Mode (Server-Sent Events)
```bash
# Run the server in SSE mode
uvx gget-mcp sse
```

</details>

In cases when there are problems with uvx often they can be caused by cleaning uv cache:
```
uv cache clean
```

The HTTP mode will start a web server that you can access at `http://localhost:3002/mcp` (with documentation at `http://localhost:3002/docs`). The STDIO mode is designed for MCP clients that communicate via standard input/output, while SSE mode uses Server-Sent Events for real-time communication.

**Note:** Currently, we do not have a Swagger/OpenAPI interface, so accessing the server directly in your browser will not show much useful information. To explore the available tools and capabilities, you should either use the MCP Inspector (see below) or connect through an MCP client to see the available tools.

## Configuring your AI Client (Anthropic Claude Desktop, Cursor, Windsurf, etc.)

We provide preconfigured JSON files for different use cases. Here are the actual configuration examples:

### STDIO Mode Configuration (Recommended)

Use this configuration for most AI clients. Create or update your MCP configuration file:

```json
{
  "mcpServers": {
    "gget-mcp": {
      "command": "uvx",
      "args": ["gget-mcp", "stdio"]
    }
  }
}
```

### HTTP Mode Configuration

For HTTP mode, ensure the server is running first with `uvx gget-mcp server`:

```json
{
  "mcpServers": {
    "gget-mcp": {
      "command": "uvx",
      "args": ["gget-mcp", "server"]
    }
  }
}
```

### STDIO-Local Mode Configuration (File Saving)

Use this mode when you want to save large output files (sequences, structures, alignments) to disk instead of returning them as text:

```json
{
  "mcpServers": {
    "gget-mcp": {
      "command": "uvx",
      "args": ["gget-mcp", "server", "--transport", "stdio-local", "--output-dir", "./gget_output"]
    }
  }
}
```

### Local Development Configuration

For local development when you've cloned the repository:

```json
{
  "mcpServers": {
    "gget-mcp": {
      "command": "uv",
      "args": ["run", "stdio"],
      "env": {
        "MCP_PORT": "3002",
        "MCP_HOST": "0.0.0.0",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

**Files Available:**
- **STDIO mode:** `mcp-config-stdio.json`
- **HTTP mode:** `mcp-config.json` 
- **STDIO-Local mode:** `mcp-config-stdio-local.json`
- **Local development:** `mcp-config-stdio-debug.json`

### Configuration Video Tutorial

For a visual guide on how to configure MCP servers with AI clients, check out our [configuration tutorial video](https://www.youtube.com/watch?v=Xo0sHWGJvE0) for our sister MCP server (biothings-mcp). The configuration principles are exactly the same for the gget MCP server - just use the appropriate JSON configuration files provided above.

### Inspecting gget MCP server

<details>
<summary>Using MCP Inspector to explore server capabilities</summary>

If you want to inspect the methods provided by the MCP server, use npx (you may need to install nodejs and npm):

For STDIO mode with uvx:
```bash
npx @modelcontextprotocol/inspector --config mcp-config-stdio.json --server gget-mcp
```

For HTTP mode (ensure server is running first):
```bash
npx @modelcontextprotocol/inspector --config mcp-config.json --server gget-mcp
```

For local development:
```bash
npx @modelcontextprotocol/inspector --config mcp-config-stdio-debug.json --server gget-mcp
```

You can also run the inspector manually and configure it through the interface:
```bash
npx @modelcontextprotocol/inspector
```

After that you can explore the tools and resources with MCP Inspector at http://127.0.0.1:6274 (note, if you run inspector several times it can change port)

</details>

### Integration with AI Systems

Simply point your AI client (like Cursor, Windsurf, ClaudeDesktop, VS Code with Copilot, or [others](https://github.com/punkpeye/awesome-mcp-clients)) to use the appropriate configuration file from the repository.

## Repository setup

<details>
<summary>For developers: cloning and running locally</summary>

```bash
# Clone the repository
git clone https://github.com/longevity-genie/gget-mcp.git
cd gget-mcp
uv sync
```

### Running the MCP Server

If you already cloned the repo you can run the server with uv:

```bash
# Start the MCP server locally (HTTP mode)
uv run server

# Or start in STDIO mode  
uv run stdio

# Or start in SSE mode
uv run sse
```

</details>

## Safety Features

- **Input validation**: Comprehensive parameter validation for all gget functions
- **Error handling**: Robust error handling with informative messages
- **Rate limiting**: Respectful usage of external APIs and databases
- **Data validation**: Type checking and data format validation

## Testing & Verification

<details>
<summary>Developer information: Testing and CI/CD workflows</summary>

The MCP server is provided with comprehensive tests including both unit tests and integration tests for network-dependent operations.

### Running Tests

Run tests for the MCP server:
```bash
# Run all tests (excluding expensive judge tests)
uv run pytest -vvv -m "not judge"

# Run only fast tests (skip slow, integration, and judge tests)
uv run pytest -vvv -m "not slow and not integration and not judge"

# Run only integration tests
uv run pytest -vvv -m integration

# Run judge tests (expensive LLM tests - requires API keys and may cost money)
uv run pytest test/test_judge.py -vvv

# Run tests with coverage (excluding judge tests)
uv run pytest -vvv --cov=src/gget_mcp --cov-report=term-missing -m "not judge"
```

**Note on Judge Tests**: The judge tests (`test_judge.py`) use large language models to evaluate the AI agent's performance with gget tools. These tests:
- Are automatically excluded from CI/CD to avoid costs
- Require API keys (GEMINI_API_KEY or similar) 
- May incur charges from LLM API providers
- Are designed for local development and manual evaluation
- Provide valuable insights into how well the tools work with AI agents

### GitHub Actions Workflows

This project includes several GitHub Actions workflows:

- **CI** (`.github/workflows/ci.yml`): Runs on every push/PR with basic linting and fast tests
- **Tests** (`.github/workflows/test.yml`): Comprehensive testing across multiple OS and Python versions
  - Fast tests on all platforms
  - Integration tests (may require network access)
  - Slow tests (like BLAST operations) - runs only on Ubuntu with Python 3.11
  - Code coverage reporting
  - Security checks with bandit and safety
- **Publish** (`.github/workflows/publish.yml`): Publishes to PyPI when tags are pushed

You can use MCP inspector with locally built MCP server same way as with uvx.

</details>

*Note: Using the MCP Inspector is optional. Most MCP clients (like Cursor, Windsurf, etc.) will automatically display the available tools from this server once configured. However, the Inspector can be useful for detailed testing and exploration.*

*If you choose to use the Inspector via `npx`, ensure you have Node.js and npm installed. Using [nvm](https://github.com/nvm-sh/nvm) (Node Version Manager) is recommended for managing Node.js versions.*

## Example Questions from Test Suite

Here are validated example questions that you can ask the AI assistant when using this MCP server:

* "Find information about the human TP53 gene and get its protein sequence."
* "What are the orthologs of BRCA1 gene across different species?"
* "Perform enrichment analysis for a set of cancer-related genes: TP53, BRCA1, BRCA2, ATM, CHEK2."
* "Get the 3D structure information for the protein encoded by the EGFR gene."
* "Find mutations in the COSMIC database for the PIK3CA gene."
* "Analyze gene expression patterns for insulin (INS) gene across different tissues."
* "Perform BLAST search with a DNA sequence to identify its origin: ATGGCGCCCGAACAGGGAC."
* "Find diseases associated with the APOE gene using OpenTargets."
* "Get reference genome information for mouse (Mus musculus)."
* "Align multiple protein sequences and identify conserved regions."

These questions are taken directly from our automated test suite and are validated to work correctly with the available gget tools.

## Contributing

We welcome contributions from the community! 🎉 Whether you're a researcher, developer, or enthusiast interested in bioinformatics and genomics research, there are many ways to get involved:

**We especially encourage you to try our MCP server and share your feedback with us!** Your experience using the server, any issues you encounter, and suggestions for improvement are incredibly valuable for making this tool better for the entire research community.

### Ways to Contribute

- **🐛 Bug Reports**: Found an issue? Please open a GitHub issue with detailed information
- **💡 Feature Requests**: Have ideas for new functionality? We'd love to hear them!
- **📝 Documentation**: Help improve our documentation, examples, or tutorials
- **🧪 Testing**: Add test cases, especially for edge cases or new query patterns
- **🔍 Data Quality**: Help identify and report data inconsistencies or suggest improvements
- **🚀 Performance**: Optimize queries, improve caching, or enhance server performance
- **🌐 Integration**: Create examples for new MCP clients or AI systems
- **🎥 Tutorials & Videos**: Create tutorials, video guides, or educational content showing how to use MCP servers
- **📖 User Stories**: Share your research workflows and success stories using our MCP servers
- **🤝 Community Outreach**: Help us evangelize MCP adoption in the bioinformatics community

**Tutorials, videos, and user stories are especially valuable to us!** We're working to push the bioinformatics community toward AI adoption, and real-world examples of how researchers use our MCP servers help demonstrate the practical benefits and encourage wider adoption.

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`uv run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow the existing code style (we use `ruff` for formatting and linting)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and write clear commit messages

### Questions or Ideas?

Don't hesitate to open an issue for discussion! We're friendly and always happy to help newcomers get started. Your contributions help advance open science and bioinformatics research for everyone. 🧬✨

## Known Issues

<details>
<summary>Technical limitations and considerations</summary>

### External Dependencies
Some gget functions depend on external web services and databases that may occasionally be unavailable or rate-limited. The server implements proper error handling and retries where appropriate.

### Test Coverage
While we provide comprehensive tests including integration tests for network-dependent operations, some test cases may be sensitive to external service availability. Some automated test results may need manual validation.

</details>

## License

This project is licensed under the MIT License.

## Acknowledgments

- [gget](https://github.com/pachterlab/gget) - The amazing bioinformatics toolkit this server wraps
- [FastMCP](https://github.com/jlowin/fastmcp) - The MCP server framework used
- [Model Context Protocol](https://modelcontextprotocol.io/) - The protocol specification

### Other MCP Servers by Longevity Genie

We also develop other specialized MCP servers for biomedical research:

- **[biothings-mcp](https://github.com/longevity-genie/biothings-mcp)** - MCP server for BioThings.io APIs, providing access to gene annotation (mygene.info), variant annotation (myvariant.info), and chemical compound data (mychem.info)
- **[opengenes-mcp](https://github.com/longevity-genie/opengenes-mcp)** - MCP server for OpenGenes database, providing access to aging and longevity research data
- **[synergy-age-mcp](https://github.com/longevity-genie/synergy-age-mcp)** - MCP server for SynergyAge database, providing access to synergistic and antagonistic longevity gene interactions

We are supported by:

[![HEALES](https://github.com/longevity-genie/biothings-mcp/raw/main/images/heales.jpg)](https://heales.org/)

*HEALES - Healthy Life Extension Society*

and

[![IBIMA](https://github.com/longevity-genie/biothings-mcp/raw/main/images/IBIMA.jpg)](https://ibima.med.uni-rostock.de/)

[IBIMA - Institute for Biostatistics and Informatics in Medicine and Ageing Research](https://ibima.med.uni-rostock.de/)
