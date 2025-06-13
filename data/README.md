# BioReasoning Data Curation

Jupyter notebooks for processing genetic variant data and creating ML datasets for biological reasoning tasks.

## Notebooks

**Core Analysis**
- `BioReasoning_DataCuration_KEGG.ipynb` - KEGG pathway analysis with Claude API
- `Clinvar_Coding.ipynb` - ClinVar variant processing and gene mapping
- `Clinvar_SNV_Non_SNV.ipynb` - SNV/structural variant datasets with VEP annotations

**KEGG Pipeline**  
- `KEGG_Data_1.ipynb` - KEGG network data processing and variant identification
- `KEGG_Data_2.ipynb` - Variant parsing and sequence generation
- `KEGG_Data_3.ipynb` - Final ML dataset creation with Q&A pairs

**Variant Prediction**
- `VEP.ipynb` - Variant effect prediction datasets (ClinVar, OMIM, eQTL)

## Setup

```bash
brew install brewsci/bio/edirect  # For ClinVar (macOS)
export ANTHROPIC_API_KEY="your-key"  # For KEGG analysis
```

## Usage

Each notebook has a configuration section - update paths/keys as needed, then run sequentially.

**Key Outputs:**
- KEGG biological reasoning datasets
- ClinVar variant-disease associations  
- VEP prediction task datasets
- Genomic sequences with variant context
