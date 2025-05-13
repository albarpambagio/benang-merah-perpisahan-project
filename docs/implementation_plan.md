## Benang Merah Perpisahan: An Analysis of Indonesian Public Divorce Documents – Refined Implementation Plan

### Project Structure

- Goal: Extract trends, emotions, and social signals from Indonesian public divorce documents.
- Pipeline: Scraping → OCR → Preprocessing → NLP Analyses → Reporting

---

### 1. Data Collection & Metadata Extraction

#### Source
- Website: putusan3.mahkamahagung.go.id (Perceraian)

#### Method
- Tool: Scrapling
- Enhancements:
  - Parallel scraping
  - Extract accompanying case metadata (Nomor, Tanggal, Hakim, Amar, page views, downloads)

```python
for page in pagination:
    links = get_all_case_links(page)
    for link in links:
        if has_pdf(link):
            download_pdf(link)
            extract_metadata(link)  # New addition
```

#### Risk Mitigation
- Add request delay and retry logic

---

### 2. OCR (for scanned PDFs)

#### Tooling
- Primary: surya
- Fallback: EasyOCR (Indonesian support)
- Additional: Tesseract with layout-aware fallback

#### Quality Check
- Apply confidence scoring
- Manually review low-confidence files
- Fix fragmented text (e.g., "Mah\nka\nmah" → "Mahkamah")

---

### 3. Text Preprocessing

#### Techniques
- Remove legal boilerplate and headers
- Normalize:
  - Spelling
  - Legal/Islamic terms (e.g., ba'da dukhul, verstek)
  - Family roles (e.g., Ibu Kandung, Sepupu)
- Standardize demographic terms (e.g., education, occupation)

#### Language Support
- Custom dictionary: legal, Arabic, Latin terms
- Tokenizer: IndoBERT or Indo-LegalBERT

---

### 4. NLP Analyses

#### 1. Topic Modeling
- Goal: Surface recurring case themes
- Tool: BERTopic
- Enhancements:
  - Sentence embeddings (IndoBERT/SBERT)
  - Manual topic labeling

#### 2. Emotion Analysis
- Goal: Capture emotional tone by section
- Approach:
  - Split text by section (e.g., DUDUK PERKARA, PERTIMBANGAN)
  - Use fine-tuned Indonesian emotion classifier
- Output:
  - Section-level emotion heatmaps

#### 3. Gender-Based Filing and Role Analysis
- Goal: Track who files more—husbands or wives
- Additional: Demographic analysis (education level, occupation)
- Approach: Rule-based extraction of Pemohon vs Termohon roles

#### 4. Outcome Classification
- Goal: Predict or extract ruling outcomes
- Target labels: Dikabulkan, Ditolak, Verstek, Talak raj'i
- Method: BERT-based text classifier or regex-based rule system
- Challenge: Amar section formatting

#### 5. Regional Mapping
- Goal: Visualize geographic divorce patterns
- Approach:
  - Extract court and residence locations
  - Use gazetteer/geocoding (e.g., GeoNames)
  - Normalize district names (e.g., Tangerang vs Kota Tangerang)

---

## Exploratory Data Analysis (EDA) Ideas

### 1. Case Filing Trends Over Time
- Plot: Number of divorce filings per month/year
- Data: Tanggal Register and Tanggal Putusan from metadata
- Goal: Understand seasonal or annual trends in divorce filings

### 2. Gender Roles in Divorce Filings
- Metric: Percentage of filings initiated by husband (cerai talak) vs wife (cerai gugat)
- Data: Role labels from PDF text (Pemohon, Termohon plus gender names)
- Visualization: Pie chart or bar chart by year or region

### 3. Geographic Distribution
- Metric: Case counts per court or city
- Data: Lembaga Peradilan, addresses in PDF text
- Visualization: Choropleth or dot map of Indonesia

### 4. Amar Putusan Outcome Frequency
- Metric: Frequency of Dikabulkan, Ditolak, Verstek, Talak Raj'i
- Data: Amar section from metadata or PDF
- Visualization: Bar chart by outcome

### 5. Duration Analysis
- Metric: Days between register date and ruling date
- Data: Tanggal Register to Tanggal Dibacakan
- Goal: Distribution of case durations
- Visualization: Histogram and boxplot

### 6. Occupational Demographics
- Metric: Top occupations of Pemohon and Termohon
- Data: Free-text fields in PDF (e.g., Ojek Online, Karyawan Swasta)
- Visualization: Word cloud or ranked bar chart
- Additional: Cross by gender or role

### 7. Educational Background Distribution
- Metric: Educational attainment of plaintiffs/defendants (e.g., SLTA, S1)
- Data: Extract from PDF profiles
- Visualization: Bar chart by education level

### 8. Marriage Duration Before Divorce
- Metric: Time from Tanggal Nikah to Tanggal Register
- Data: Tanggal Nikah from PDF plus register date
- Goal: Histogram of marriage length before filing

### 9. Children Involved in Divorce
- Metric: Number of children per case
- Data: Text mentions such as "dikaruniai 2 orang anak"
- Visualization: Histogram of child count per case
- Additional: Slice by gender role or outcome

### 10. Reason/Conflict Type Frequency
- Metric: Types of conflicts (e.g., kurang perhatian, perselingkuhan, KDRT)
- Data: Extract reasons from Duduk Perkara
- Visualization: Word cloud or multi-label bar chart

### 11. Case Popularity Analysis
- Metric: Page views and downloads per case
- Data: Page views and downloads from metadata
- Visualization: Distribution plots, correlation with case features (region, outcome, gender, etc.)
- Goal: Identify which cases attract more public attention and analyze potential biases

### 8. Data Handling Strategies

#### Data Validation and Cleaning
- Implement automated scripts to check for missing, inconsistent, or implausible values (e.g., negative durations, future dates, malformed IDs).
- Enforce data structure and types using JSON Schema or Pydantic.
- Periodically conduct manual spot checks for quality assurance.

#### Data Versioning
- Store raw, intermediate, and final datasets separately.
- Use DVC (Data Version Control), Git LFS, or clear folder naming conventions to track changes in data over time.

#### Data Backup and Redundancy
- Schedule automated backups to cloud storage (e.g., Google Drive, S3, Zenodo).
- Store critical data in at least two geographically separate locations.

#### Data Access and Security
- Restrict access to sensitive/raw data to authorized personnel only.
- Keep logs of data access and modifications for traceability.

#### Data Anonymization and Redaction
- Use automated scripts to remove or mask personal identifiers before data leaves the secure environment.
- Maintain logs of what was redacted and why, for transparency.

#### Data Provenance and Lineage
- Record the source, extraction date, and processing steps for each data item.
- Document how each dataset was derived, including scripts and parameters used.

#### Efficient Data Storage and Retrieval
- For large datasets, store data in chunks or use lazy loading to avoid memory issues.
- Index key fields (e.g., case ID, date) for fast retrieval.

#### Data Interoperability
- Use standard formats (JSON, CSV, Parquet) for compatibility with other tools.
- Consider building simple APIs or using data catalogs for team access.

#### Data Documentation
- Maintain a living data dictionary describing each field, its type, and possible values.
- Keep logs of all data processing steps for reproducibility.

#### Data Deletion and Takedown
- Implement a process for removing data upon request, and document how requests are handled.

---

### 9. Data Format Discourse: JSON for Metadata and OCR Outputs

JSON (JavaScript Object Notation) is the recommended format for storing both metadata and OCR outputs in this project.

#### Advantages
- Human- and machine-readable, easy to parse and generate.
- Supports complex, nested, and hierarchical data structures (e.g., case metadata, OCR text by page/section, confidence scores).
- Widely supported by programming languages and data tools (Python, JavaScript, Pandas, etc.).
- Flexible and extensible: new fields can be added without breaking existing code.
- Suitable for storing intermediate results in data pipelines and for versioning in Git.

#### Example Structures

Metadata Example:
```json
{
  "case_id": "619/Pdt.G/2025/PA.Tng",
  "register_date": "2025-03-12",
  "court": "PA TANGERANG",
  "judge": "Dra. Hj. Nikma",
  "outcome": "Dikabulkan",
  "page_views": 123,
  "downloads": 45
}
```

OCR Output Example:
```json
{
  "case_id": "619/Pdt.G/2025/PA.Tng",
  "ocr": [
    { "page": 1, "text": "Isi halaman 1...", "confidence": 0.98 },
    { "page": 2, "text": "Isi halaman 2...", "confidence": 0.95 }
  ]
}
```

#### Limitations
- For extremely large datasets, JSON can become unwieldy. Consider splitting files (one per case) or using more efficient formats (e.g., Parquet, NDJSON, or a database) for large-scale storage.
- Not suitable for storing images or binary data (store file paths or references instead).

#### Best Practices
- Use UTF-8 encoding.
- Validate JSON structure with schemas (e.g., JSON Schema).
- For large-scale projects, consider line-delimited JSON (NDJSON) for easier streaming and processing.

JSON is thus a robust, flexible, and well-supported choice for this project's data storage needs.

---

### 5. Reporting and Distribution

#### Deliverables
- Medium article (methods and insights)
- Twitter thread (infographics)
- GitHub repo (scraper, NLP pipeline)
- Interactive dashboards (e.g., Streamlit, Plotly Dash)

#### Visuals
- Topic cluster plots
- Section-based emotion charts
- Filing trends by gender and region
- Amar outcome distributions
- Timeline of marriage, conflict, and separation

---

### 6. Ethics, Privacy, and Storage

#### Privacy
- Redact all personal identifiers
- No direct quotes or full documents in public outputs
- Provide takedown request form
- Automate redaction where possible and log redacted fields

#### Storage
- External PDF storage (e.g., Google Drive, Zenodo)
- Reference documents using content hashes

---

### 7. Dataset Publishing Plan

#### Platforms
- Publish processed datasets on Kaggle and HuggingFace Datasets
- Provide links and DOIs for citation

#### Data Documentation
- Include a detailed data dictionary and schema description
- Document extraction, cleaning, and redaction steps
- Provide sample code for loading and using the dataset

#### Privacy Considerations
- Ensure all personal identifiers are redacted
- Remove or anonymize sensitive fields
- Include a clear data usage and privacy statement
- Provide a takedown request mechanism

#### Versioning
- Use semantic versioning for dataset releases (e.g., v1.0, v1.1)
- Maintain a changelog documenting updates, corrections, and new features

#### Reproducibility
- Publish code and notebooks for data processing and analysis
- Include instructions for reproducing the dataset from raw sources

---

### Updated Checklist

- [ ] Scrape 50–100 sample cases and metadata
- [ ] Run OCR and evaluate accuracy (especially fragmented layout)
- [ ] Normalize section headers and demographic fields
- [ ] Run BERTopic and section-based emotion model
- [ ] Extract outcomes and demographic patterns
- [ ] Analyze and visualize page views and downloads
- [ ] Publish pilot report on GitHub and Medium
- [ ] Release dataset on Kaggle and HuggingFace with documentation

```