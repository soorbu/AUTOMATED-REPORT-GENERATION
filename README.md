# Task 2: AUTOMATED REPORT GENERATION 

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
The report will be saved as `report_YYYYMMDD_HHMMSS.pdf`.

---

## Output Screenshots

![Image](https://github.com/user-attachments/assets/a324f1fe-3e48-4773-88b5-e18f1d289ba3)
![Image](https://github.com/user-attachments/assets/332fa017-cc44-4ede-8b76-61fbc3d4952a)
![Image](https://github.com/user-attachments/assets/0b6d0d2c-4d84-49ab-a3d0-ae04987216ff)
![Image](https://github.com/user-attachments/assets/76dfa13d-06e4-4bf7-88b9-0affcab58732)
![Image](https://github.com/user-attachments/assets/3315a6fb-82c5-4fa6-8545-ab36f6b6c35f)
![Image](https://github.com/user-attachments/assets/41100118-8635-4cb7-a5dd-69547f4fda02)

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

