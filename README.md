COMPANY: COTECH IT SOLUTIONS

NAME: SURABHI SANJAY KHATALE

INTERN ID: CT04DG167

DOMAIN: PYTHON PROGRAMMING

MENTOR: VAISHALI SHRIVASTAVA

DESCRIPTION--
This Python project automates the process of reading a CSV file, analyzing its contents (numeric and categorical), visualizing trends, and generating a professional PDF report using `ReportLab`. It's a useful tool for data analysts, students, or anyone who needs to quickly summarize datasets into clean, readable reports.

---

## Features

- Reads any structured CSV dataset
- Analyzes numeric and categorical data
- Highlights missing values and data types
- Generates charts (histograms, box plots, bar charts, heatmaps)
- Dynamic insights/conclusion generation based on dataset
- Outputs a professionally formatted multi-page PDF

---

## Project Structure

```
automated_report_generator/
├── report_generator.py      # Main script
├── sample_data.csv          # Sample dataset (optional)
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/automated-report-generator.git
cd automated-report-generator
```

2. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Option 1: With your own CSV file
```bash
python report_generator.py path/to/your/data.csv
```

### Option 2: Use the sample data (auto-generated)
```bash
python report_generator.py
# or simply run:
python report_generator.py
```
The script will create a sample dataset and generate the PDF.

The report will be saved as `report_YYYYMMDD_HHMMSS.pdf`.

---

## Output Screenshots

---

## Example Insights

- Most frequent categorical values
- Column with highest missing values
- Numeric column with the highest average
- Most strongly correlated variables (if applicable)

---

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `reportlab`

Install via:
```bash
pip install -r requirements.txt
```

---

## Notes

- Works with most structured CSV datasets
- Automatically adapts to available data columns
- All charts are generated dynamically and embedded in the PDF

