# QCA-AID Project Guide for AI Agents

This document provides essential information for AI agents working with the QCA-AID codebase, a Python tool for AI-supported qualitative content analysis implementing Mayring's method.

## Project Overview

QCA-AID (Qualitative Content Analysis with AI-supported Discovery) combines traditional qualitative research methods with AI capabilities to assist researchers in analyzing document and interview data. The tool supports deductive, inductive, and abductive analysis modes with multi-coder reliability features.

## Codebase Structure

```
QCA-AID/
├── QCA-AID.py                 # Main launcher script
├── QCA_AID_assets/           # Core application code
│   ├── main.py              # Main execution coordinator
│   ├── core/                # Configuration and data models
│   ├── preprocessing/       # Document loading and text processing
│   ├── analysis/            # Coding logic (deductive, inductive, manual)
│   ├── quality/             # Reliability and review management
│   ├── management/          # Category and development history management
│   ├── export/              # Results exporting functionality
│   └── output/              # Runtime output directory
├── input/                   # Input documents directory
├── output/                  # Analysis results directory
├── QCA_Prompts.py           # LLM prompt templates
├── QCA_Utils.py             # Utility functions and helpers
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── QCA-AID-Codebook.xlsx   # Primary configuration file
```

## Key Components

### Main Execution Flow
1. `QCA-AID.py` → Launches `QCA_AID_assets/main.py`
2. Configuration loaded from `QCA-AID-Codebook.xlsx`
3. Documents loaded from `input/` directory
4. Text chunking and preprocessing
5. Relevance checking to filter segments
6. Coding (deductive/inductive/abductive modes)
7. Manual coding (optional)
8. Intercoder reliability analysis
9. Review and consensus building
10. Results exported to `output/` directory

### Core Modules
- **analysis_manager.py**: Coordinates the entire analysis workflow
- **deductive_coding.py**: Handles deductive category application
- **inductive_coding.py**: Manages inductive category development
- **manual_coding.py**: Provides GUI for human coding
- **relevance_checker.py**: Filters text segments before coding
- **results_exporter.py**: Exports analysis results to Excel

## Configuration

The primary configuration is in `QCA-AID-Codebook.xlsx` which contains:
- Research question
- Deductive categories with definitions and examples
- Coding rules and exclusions
- Technical settings (model, directories, chunk sizes)

Secondary configuration is in `QCA_AID_assets/core/config.py` which provides defaults.

## Essential Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install spaCy German language model
python -m spacy download de_core_news_sm
```

### Running Analysis
```bash
# Run the main analysis
python QCA-AID.py
```

### Development
```bash
# Check for syntax errors
python -m py_compile QCA-AID.py

# Run with specific Python version (3.11 or older)
python3.11 QCA-AID.py
```

## Important Conventions

### Coding Patterns
1. Uses asyncio for concurrent processing
2. Configuration loaded via Excel file (`QCA-AID-Codebook.xlsx`)
3. Text documents processed in configurable chunks with overlap
4. Multiple analysis modes (deductive, inductive, abductive, grounded)
5. Multi-coder support with reliability calculations
6. Manual coding interface via Tkinter GUI

### Key Variables
- `CONFIG`: Global configuration dictionary
- `FORSCHUNGSFRAGE`: Research question
- `DEDUKTIVE_KATEGORIEN`: Deductive category definitions
- `CHUNK_SIZE`: Size of text segments for analysis
- `BATCH_SIZE`: Number of segments processed together
- `ANALYSIS_MODE`: Can be 'deductive', 'inductive', 'abductive', or 'grounded'

### Naming Conventions
- Python files use snake_case
- Classes use PascalCase
- Configuration constants in UPPER_CASE
- Private methods prefixed with underscore

## Testing Approach

The project doesn't have formal unit tests. Testing is done through:
1. Manual execution with sample data
2. Verification of output files in `output/` directory
3. Cross-checking reliability statistics
4. Review of exported Excel results

## Common Gotchas

1. **Python Version**: Requires Python 3.11 or older (spaCy compatibility)
2. **API Keys**: Must be stored in `~/.environ.env` file
3. **File Formats**: Only supports .txt, .pdf, .docx input files
4. **Memory Usage**: Large documents may require significant RAM
5. **Threading Issues**: Tkinter GUI requires special thread handling
6. **Context Coding**: When enabled, documents must be processed sequentially

## Refactoring Status

**QCA_Utils.py Refactoring** (In Planning Phase):
- See `REFACTORING_PLAN.md` and `REFACTORING_SUMMARY.txt` for detailed plan
- Current: 3954+ line monolith with 15 classes
- Target: 6 focused modules (utils/llm/, utils/config/, utils/tracking/, utils/dialog/, utils/export/, utils/io/)
- Timeline: 8 phases, ~8 hours
- Status: Plan created, ready for Phase 1 (Structure Setup)

## Key Workflows

### Adding New Analysis Mode
1. Update `ANALYSIS_MODE` handling in `analysis_manager.py`
2. Add mode-specific logic in `_process_{mode}_mode` methods
3. Update configuration validation
4. Test with sample data

### Modifying Category Structure
1. Update `DEDUKTIVE_KATEGORIEN` in `config.py` or Excel
2. Ensure subcategory structure matches expected format
3. Update validation rules in `CategoryValidator`
4. Test category application

### Extending Export Functionality
1. Modify `results_exporter.py`
2. Update Excel template in `export_results` method
3. Add new sheets or columns as needed
4. Maintain compatibility with existing output format

## Dependencies

Key external libraries:
- openai: For OpenAI API integration
- pandas/openpyxl: For Excel processing
- spacy: For text preprocessing
- PyPDF2/python-docx: For document parsing
- tkinter: For manual coding interface
- mistralai: For Mistral API support (optional)