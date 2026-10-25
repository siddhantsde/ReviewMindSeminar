# ReviewMind — Product Review Intelligence

ReviewMind is a local-first product intelligence prototype that turns review text into sentiment, product comparisons, evidence-linked issue priorities, hidden-problem signals, trend analysis, and downloadable reports. The bundled 1,200-row dataset is reproducibly generated synthetic feedback about fictional products; it is not actual customer data.

## Problem and objectives

Manually reading large volumes of reviews makes it hard to spot recurring satisfaction drivers and complaints. ReviewMind demonstrates a local analytics workflow: validate review data, quantify ratings and sentiment, explore themes, compare products, and present findings with uncertainty clearly stated.

## Features

- Overview KPIs and Plotly charts for sentiment, ratings, time, products, and categories.
- CSV upload with required-column validation, accepted/rejected counts, duplicate exclusion, preview, cleaned CSV export, and a 20 MB limit.
- Review search and filters for category, product, rating, sentiment, verified purchase, plus date and helpful-vote sorting.
- Local VADER sentiment and TF-IDF/NMF topic discovery.
- Product comparisons with review counts and small-sample warning.
- Customer Pain Points with a transparent keyword taxonomy, eligible-review denominators, issue sentiment, rating impact, product coverage, trend, representative evidence, and configurable priority weights.
- Issue Priority Matrix and Hidden Problems investigation signals, explicitly labeled as heuristics rather than validated risk or defect claims.
- Monthly ratings, volume, sentiment and issue frequency; configurable recent-period issue comparisons require minimum sample counts and report percentage-point changes.
- Multi-product comparisons with issue evidence, rating–sentiment disagreements, sample-size warnings, and approximate mean-rating intervals.
- Review-level sentiment and issue phrase inspection, plus a mixed-opinion language cue.
- Dataset quality coverage, illustrative business-impact scenarios with user-entered assumptions, and grouped navigation for analysis pages.
- Optional local Ollama reports use a compact verified evidence package and system-level instructions treating customer review text as untrusted. When unavailable, a deterministic rule-based report uses the same calculated evidence.
- SQLite report history and downloadable print-ready HTML, simple Markdown, and metrics CSV exports.
- Deep analytics for data completeness, rating distribution and uncertainty, keyword-based review aspects, verified-purchase segments, and monthly trends.
- A copy-ready prompt for transforming reviews pasted from websites, tables, or plain text into import-ready CSV.
- Star ratings may be blank when the source did not provide them; ReviewMind never invents a score and still analyzes the review text.

## Technology and architecture

Python, Streamlit, pandas, NumPy, Plotly, VADER, scikit-learn, requests, and SQLite. `app.py` connects the modular services in `src/`: loading, preprocessing, sentiment, topics, analytics, recommendations, optional Ollama, persistence, and report rendering. All ordinary analytics operate locally; Ollama is contacted only to check status or when the user requests a report and has configured a model. No API key is required.

## Project structure

```text
ReviewMind/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── generate_sample.py
├── data/sample_reviews.csv
├── src/ (analytics, loading, preprocessing, sentiment, topics, issue intelligence, recommendations, LLM, database, reports)
├── prompts/review_text_to_csv.txt
├── tests/test_core.py
└── reports/.gitkeep
```

## Windows installation and startup

From the `ReviewMind` folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To regenerate the deterministic synthetic dataset, run `python generate_sample.py`.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for steps to publish on Streamlit Community Cloud and details about hosted uploads, local SQLite history, and Ollama availability.

## Using ReviewMind

The bundled synthetic dataset loads on startup. Use the sidebar to filter the active data or upload a CSV. Required columns are `product_name`, `rating`, and `review_text`; optional columns are `product_id`, `category`, `review_date`, `verified_purchase`, `helpful_votes`, and `review_id`. Invalid rows are reported and excluded from analysis; duplicate review text can be excluded. Upload errors are shown and never trigger an automatic switch to sample data. All pages use the active dataset and shared product/category filters.

### Converting pasted reviews to CSV

Open **Review-to-CSV Prompt** in the application to copy or download the prompt. It can also be embedded in a website assistant; the standalone text is in `prompts/review_text_to_csv.txt`. Give the prompt and pasted review content to the model, then save its CSV code block as a `.csv` file and upload it. The prompt handles copied page text and tables, separates individual reviews, preserves explicit ratings/dates, and leaves unknown fields blank. Blank ratings are retained and sentiment is still analyzed; they are left out of rating averages. Never trust an AI extraction as source truth—spot-check product names, row boundaries, and ratings against the copied text.

For a demonstration: show Overview; inspect Customer Pain Points and the priority components; investigate the potential hidden battery issue; explore monthly trends; compare products; generate a report (local LLM or clearly labeled rule-based fallback); download the polished HTML or Markdown report; then open Analysis History.

## Optional local Ollama setup

Install Ollama separately, start its local service, and download a compatible instruction model using Ollama's own model command, for example `ollama pull llama3.2`. Copy `.env.example` to `.env`, then configure `OLLAMA_MODEL` to the exact installed model name and optionally `OLLAMA_BASE_URL` (default `http://localhost:11434`). Set the variables in the shell before launching Streamlit, or load the `.env` values in your environment. The model must already be installed locally. Model download is a one-time network operation managed by Ollama; ReviewMind does not use a hosted inference API.

Without Ollama, all analysis remains available. Report generation produces explicitly labeled **Rule-based insights — local LLM unavailable** output. It never presents that fallback as generative output.

## Tests

```powershell
python -m pytest -q
```

Tests cover required columns, missing text, invalid ratings, duplicates, sentiment, aggregate metrics, empty/small input, topic-analysis failure, SQLite, HTML escaping, unavailable-LLM fallback, pain-point evidence and denominators, priority weights, trend thresholds, hidden-problem rules, rating–sentiment disagreement, product comparison, and scenario calculations. They do not require Ollama or internet access.

## Limitations and future work

The sample is synthetic. VADER is lexicon-based and can misread sarcasm, negation, mixed opinions, or domain language. NMF topics are keyword summaries, not verified business categories; the taxonomy can miss product-specific language. Priority weights and hidden-issue thresholds are configurable heuristics, not validated models. Near-duplicate text detection is not implemented. A website-to-CSV prompt depends on the external model used to run it; verify extracted data before upload. Results depend on supplied data quality and sample size. The prototype has no authentication or multi-user separation. Hosted SQLite history may be ephemeral depending on the hosting filesystem.
