## 🧭 **Benang Merah Perpisahan: An Analysis of Indonesian Public Divorce Documents – Refined Implementation Plan**

### **🧱 Project Structure**

* **Goal**: Extract trends, emotions, and social signals from Indonesian public divorce documents.
* **Pipeline**:
  `Scraping → OCR → Preprocessing → NLP Analyses → Reporting`

---

### **🔹 1. Data Collection & Metadata Extraction**

#### **📌 Source**
* Website: [putusan3.mahkamahagung.go.id (Perceraian)](https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perceraian.html)

#### **📦 Method**
* **Tool**: [Scrapling](https://github.com/D4Vinci/Scrapling)
* **Enhancements**:
  - Parallel scraping
  - Extract accompanying case metadata (Nomor, Tanggal, Hakim, Amar)

```python
for page in pagination:
    links = get_all_case_links(page)
    for link in links:
        if has_pdf(link):
            download_pdf(link)
            extract_metadata(link)  # New addition
````

#### **🛡 Risk Mitigation**

* Add request delay & retry

---

### **🔹 2. OCR (for scanned PDFs)**

#### **📌 Tooling**

* Primary: [`surya`](https://github.com/VikParuchuri/surya)
* Fallback: `EasyOCR` (Indonesian support)
* Add: Tesseract with `--psm 6` for layout-aware fallback

#### **✅ Quality Check**

* Apply confidence scoring
* Manually review low-confidence files
* Fix fragmented text (e.g., "Mah\nka\nmah" → "Mahkamah")

---

### **🔹 3. Text Preprocessing**

#### **⚙️ Techniques**

* Remove legal boilerplate & headers
* Normalize:

  * Spelling
  * Legal/Islamic terms (`ba’da dukhul`, `verstek`, etc.)
  * Family roles (`Ibu Kandung`, `Sepupu`)
* Standardize demographic terms (e.g., education, occupation)

#### **📚 Language Support**

* Custom dictionary: legal, Arabic, Latin terms
* Tokenizer: IndoBERT or Indo-LegalBERT

---

### **🔹 4. NLP Analyses**

#### **1. Topic Modeling**

* **Goal**: Surface recurring case themes
* **Tool**: BERTopic
* **Enhancements**:

  * Sentence embeddings (IndoBERT/SBERT)
  * Manual topic labeling

---

#### **2. Emotion Analysis**

* **Goal**: Capture emotional tone by section
* **Approach**:

  * Split text by section (e.g., "DUDUK PERKARA", "PERTIMBANGAN")
  * Use fine-tuned Indonesian emotion classifier
* **Output**:

  * Section-level emotion heatmaps

---

#### **3. Gender-Based Filing & Role Analysis**

* **Goal**: Track who files more—husbands or wives
* **Add**: Demographic analysis

  * Education level
  * Occupation
* **Approach**: Rule-based extraction of "Pemohon" vs "Termohon" roles

---

#### **4. Outcome Classification**

* **Goal**: Predict or extract ruling outcomes
* **Target labels**:

  * `Dikabulkan`, `Ditolak`, `Verstek`, `Talak raj’i`
* **Method**: BERT-based text classifier or regex-based rule system
* **Challenge**: Amar section formatting

---

#### **5. Regional Mapping**

* **Goal**: Visualize geographic divorce patterns
* **Approach**:

  * Extract court & residence locations
  * Use gazetteer/geocoding (e.g., GeoNames)
  * Normalize district names (e.g., "Tangerang" vs "Kota Tangerang")

---

Based on the **sample divorce case PDF** and its **metadata**, here are **targeted Exploratory Data Analysis (EDA)** ideas you can implement for your project:

---

## 📊 Exploratory Data Analysis (EDA) Ideas

### 🔹 1. **Case Filing Trends Over Time**

* 📅 **Plot**: Number of divorce filings per month/year.
* ✅ **Data**: `Tanggal Register` and `Tanggal Putusan` from metadata.
* 📈 **Goal**: Understand seasonal or annual trends in divorce filings.

---

### 🔹 2. **Gender Roles in Divorce Filings**

* 👩‍⚖️ **Metric**: % of filings initiated by **husband (cerai talak)** vs **wife (cerai gugat)**.
* ✅ **Data**: Role labels from PDF text (`Pemohon`, `Termohon` + gender names).
* 📊 **Visualization**: Pie chart or bar chart by year or region.

---

### 🔹 3. **Geographic Distribution**

* 🗺 **Metric**: Case counts per court or city.
* ✅ **Data**: `Lembaga Peradilan`, addresses in PDF text.
* 📍 **Visualization**: Choropleth or dot map of Indonesia.

---

### 🔹 4. **Amar Putusan Outcome Frequency**

* ⚖️ **Metric**: Frequency of `Dikabulkan`, `Ditolak`, `Verstek`, `Talak Raj’i`.
* ✅ **Data**: `Amar` section from metadata or PDF.
* 📊 **Visualization**: Bar chart by outcome.

---

### 🔹 5. **Duration Analysis**

* ⏳ **Metric**: Days between register date and ruling date.
* ✅ **Data**: `Tanggal Register` → `Tanggal Dibacakan`
* 📈 **Goal**: Distribution of case durations.
* 📊 **Visualization**: Histogram + boxplot.

---

### 🔹 6. **Occupational Demographics**

* 👨‍🔧 **Metric**: Top occupations of Pemohon and Termohon.
* ✅ **Data**: Free-text fields in PDF (e.g., “Ojek Online”, “Karyawan Swasta”).
* 📊 **Visualization**: Word cloud or ranked bar chart.
* 💡 **Bonus**: Cross by gender/role.

---

### 🔹 7. **Educational Background Distribution**

* 🎓 **Metric**: Educational attainment of plaintiffs/defendants (e.g., SLTA, S1).
* ✅ **Data**: Extract from PDF profiles.
* 📊 **Visualization**: Bar chart by education level.

---

### 🔹 8. **Marriage Duration Before Divorce**

* ❤️‍🩹 **Metric**: Time from `Tanggal Nikah` to `Tanggal Register`.
* ✅ **Data**: `Tanggal Nikah` from PDF + register date.
* 📈 **Goal**: Histogram of marriage length before filing.

---

### 🔹 9. **Children Involved in Divorce**

* 👶 **Metric**: Number of children per case.
* ✅ **Data**: Text mentions like “dikaruniai 2 orang anak”.
* 📊 **Visualization**: Histogram of child count per case.
* 💡 **Bonus**: Slice by gender role or outcome.

---

### 🔹 10. **Reason/Conflict Type Frequency**

* 💥 **Metric**: Types of conflicts (e.g., “kurang perhatian”, “perselingkuhan”, “KDRT”).
* ✅ **Data**: Extract reasons from `Duduk Perkara`.
* 📊 **Visualization**: Word cloud or multi-label bar chart.

---


### **🔹 6. Reporting & Distribution**

#### **Deliverables**

* 📖 Medium article (methods + insights)
* 🧵 Twitter thread (infographics)
* 💻 GitHub repo (scraper, NLP pipeline)
* kaggle and huggingface for dataset publishing

#### **Visuals**

* Topic cluster plots
* Section-based emotion charts
* Filing trends by gender/region
* Amar outcome distributions
* Timeline of marriage–conflict–separation

---

### **🔹 6. Ethics & Storage**

#### **Privacy**

* Redact all personal identifiers
* No direct quotes or full documents
* Provide takedown request form

#### **Storage**

* External PDF storage (e.g., GDrive/Zenodo)
* Reference documents using content hashes

---

### ✅ **Updated Checklist**

* [ ] Scrape 50–100 sample cases + metadata
* [ ] Run OCR and evaluate accuracy (esp. fragmented layout)
* [ ] Normalize section headers and demographic fields
* [ ] Run BERTopic & section-based emotion model
* [ ] Extract outcomes and demographic patterns
* [ ] Publish pilot report on GitHub + Medium

```

