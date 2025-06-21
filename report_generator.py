"""
Automated CSV Data Analysis & PDF Report Generator

This script reads data from a CSV file, performs comprehensive analysis,
and generates a professional PDF report with charts, tables, and statistics.

Requirements:
pip install reportlab pandas matplotlib seaborn numpy

Run:
python report_generator.py [csv_file_path]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CSVReportGenerator:
    def __init__(self, csv_file_path, output_path=None):
        """
        Initialize the report generator
        
        Args:
            csv_file_path (str): Path to the CSV file
            output_path (str): Path for output PDF (optional)
        """
        self.csv_file_path = csv_file_path
        self.output_path = output_path or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        self.df = None
        self.styles = getSampleStyleSheet()
        self.story = []
        self.temp_chart_files = []  # Track temporary chart files for cleanup
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.blue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.blue
        )
        
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.darkgreen
        )

    def load_data(self):
        """Load and validate CSV data"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False

    def analyze_data(self):
        """Perform comprehensive data analysis"""
        if self.df is None:
            return None
            
        analysis = {
            'basic_info': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'column_names': list(self.df.columns),
                'data_types': self.df.dtypes.to_dict(),
                'missing_values': self.df.isnull().sum().to_dict(),
                'memory_usage': self.df.memory_usage(deep=True).sum()
            },
            'numeric_analysis': {},
            'categorical_analysis': {},
            'correlations': None
        }
        
        # Numeric columns analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['numeric_analysis'] = {
                'columns': list(numeric_cols),
                'statistics': self.df[numeric_cols].describe().to_dict(),
                'totals': self.df[numeric_cols].sum().to_dict(),
                'means': self.df[numeric_cols].mean().to_dict(),
                'medians': self.df[numeric_cols].median().to_dict(),
                'std_devs': self.df[numeric_cols].std().to_dict()
            }
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                analysis['correlations'] = self.df[numeric_cols].corr()
        
        # Categorical columns analysis
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_analysis = {}
            for col in categorical_cols:
                cat_analysis[col] = {
                    'unique_values': self.df[col].nunique(),
                    'value_counts': self.df[col].value_counts().head(10).to_dict(),
                    'most_frequent': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A'
                }
            analysis['categorical_analysis'] = cat_analysis
        
        return analysis

    def create_charts(self, analysis):
        """Create visualization charts"""
        charts = []
        
        # Set matplotlib backend and style
        plt.switch_backend('Agg')  # Using non-interactive backend
        try:
            plt.style.use('default')
        except:
            pass
        
        numeric_cols = analysis['numeric_analysis'].get('columns', [])
        categorical_cols = list(analysis['categorical_analysis'].keys())
        
        # Chart 1: Numeric data distribution
        if len(numeric_cols) > 0:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('Numeric Data Analysis', fontsize=16, fontweight='bold')
                
                # Histogram of first numeric column
                if len(numeric_cols) >= 1:
                    self.df[numeric_cols[0]].hist(bins=20, ax=axes[0, 0], color='skyblue', alpha=0.7)
                    axes[0, 0].set_title(f'Distribution of {numeric_cols[0]}')
                    axes[0, 0].set_xlabel(numeric_cols[0])
                    axes[0, 0].set_ylabel('Frequency')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, 0].transAxes)
                
                # Box plot of numeric columns
                if len(numeric_cols) >= 2:
                    box_data = [self.df[col].dropna() for col in numeric_cols[:4]]
                    axes[0, 1].boxplot(box_data, labels=numeric_cols[:4])
                    axes[0, 1].set_title('Box Plot of Numeric Columns')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                elif len(numeric_cols) == 1:
                    axes[0, 1].boxplot([self.df[numeric_cols[0]].dropna()], labels=[numeric_cols[0]])
                    axes[0, 1].set_title('Box Plot')
                else:
                    axes[0, 1].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, 1].transAxes)
                
                # Line plot showing trends
                if len(numeric_cols) >= 1:
                    sample_size = min(100, len(self.df))  # Limit points for readability
                    sample_data = self.df[numeric_cols[0]].head(sample_size)
                    axes[1, 0].plot(range(len(sample_data)), sample_data, marker='o', markersize=2)
                    axes[1, 0].set_title(f'Trend of {numeric_cols[0]} (First {sample_size} records)')
                    axes[1, 0].set_xlabel('Record Index')
                    axes[1, 0].set_ylabel(numeric_cols[0])
                else:
                    axes[1, 0].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[1, 0].transAxes)
                
                # Summary statistics bar chart
                if len(numeric_cols) >= 1:
                    cols_to_show = numeric_cols[:5]
                    means = [analysis['numeric_analysis']['means'][col] for col in cols_to_show]
                    axes[1, 1].bar(range(len(cols_to_show)), means, color='lightgreen', alpha=0.7)
                    axes[1, 1].set_title('Mean Values by Column')
                    axes[1, 1].set_xticks(range(len(cols_to_show)))
                    axes[1, 1].set_xticklabels(cols_to_show, rotation=45)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                
                plt.tight_layout()
                
                # Save chart to file directly
                chart_filename = f"temp_numeric_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_filename, format='PNG', dpi=150, bbox_inches='tight')
                charts.append(chart_filename)
                plt.close(fig)
                
            except Exception as e:
                print(f"Error creating numeric analysis chart: {e}")
                plt.close('all')
        
        # Chart 2: Categorical data analysis
        if len(categorical_cols) > 0:
            try:
                fig_size = (12, 8) if len(categorical_cols) <= 2 else (12, 10)
                rows = 1 if len(categorical_cols) <= 2 else 2
                cols = min(2, len(categorical_cols))
                
                fig, axes = plt.subplots(rows, cols, figsize=fig_size)
                if rows == 1 and cols == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes
                else:
                    axes = axes.flatten()
                
                fig.suptitle('Categorical Data Analysis', fontsize=16, fontweight='bold')
                
                for i, col in enumerate(categorical_cols[:4]):
                    if i >= len(axes):
                        break
                        
                    value_counts = self.df[col].value_counts().head(10)
                    
                    if len(value_counts) > 0:
                        # Bar chart
                        axes[i].bar(range(len(value_counts)), value_counts.values, color='coral', alpha=0.7)
                        axes[i].set_title(f'Top Values in {col}')
                        axes[i].set_xticks(range(len(value_counts)))
                        axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                        axes[i].set_ylabel('Count')
                
                # Hide unused subplots
                for i in range(len(categorical_cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                # Save chart to file directly
                chart_filename = f"temp_categorical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_filename, format='PNG', dpi=150, bbox_inches='tight')
                charts.append(chart_filename)
                plt.close(fig)
                
            except Exception as e:
                print(f"Error creating categorical analysis chart: {e}")
                plt.close('all')
        
        # Chart 3: Correlation heatmap
        if analysis.get('correlations') is not None and len(analysis['correlations']) > 1:
            try:
                plt.figure(figsize=(10, 8))
                
                # Create correlation matrix
                corr_matrix = analysis['correlations']
                
                # Create heatmap manually without seaborn dependency
                im = plt.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                
                # Add colorbar
                plt.colorbar(im)
                
                # Set ticks and labels
                plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
                plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
                
                # Add correlation values as text
                for i in range(len(corr_matrix.index)):
                    for j in range(len(corr_matrix.columns)):
                        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')
                
                plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Save chart to file directly
                chart_filename = f"temp_correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_filename, format='PNG', dpi=150, bbox_inches='tight')
                charts.append(chart_filename)
                plt.close()
                
            except Exception as e:
                print(f"Error creating correlation matrix chart: {e}")
                plt.close('all')
        
        return charts

    def add_title_page(self):
        """Add title page to the report"""
        self.story.append(Spacer(1, 2*inch))
        
        title = Paragraph("Data Analysis Report", self.title_style)
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))
        
        subtitle = Paragraph(f"Analysis of: {os.path.basename(self.csv_file_path)}", self.heading_style)
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.3*inch))
        
        date_info = Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", self.styles['Normal'])
        self.story.append(date_info)
        self.story.append(PageBreak())

    def add_summary_section(self, analysis):
        """Add summary statistics section"""
        self.story.append(Paragraph("Executive Summary", self.heading_style))
        
        basic_info = analysis['basic_info']
        
        summary_data = [
            ['Dataset Overview', ''],
            ['Total Records', f"{basic_info['total_rows']:,}"],
            ['Total Columns', str(basic_info['total_columns'])],
            ['Memory Usage', f"{basic_info['memory_usage'] / 1024:.2f} KB"],
            ['Missing Values', str(sum(basic_info['missing_values'].values()))],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        self.story.append(summary_table)
        self.story.append(Spacer(1, 0.5*inch))

    def add_column_info_section(self, analysis):
        """Add detailed column information"""
        self.story.append(Paragraph("Column Information", self.heading_style))
        
        basic_info = analysis['basic_info']
        
        column_data = [['Column Name', 'Data Type', 'Missing Values', 'Sample Value']]
        
        for col in basic_info['column_names']:
            dtype = str(basic_info['data_types'][col])
            missing = basic_info['missing_values'][col]
            sample_val = str(self.df[col].iloc[0]) if not self.df[col].empty else 'N/A'
            if len(sample_val) > 20:
                sample_val = sample_val[:20] + '...'
            
            column_data.append([col, dtype, str(missing), sample_val])
        
        column_table = Table(column_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
        column_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        self.story.append(column_table)
        self.story.append(Spacer(1, 0.5*inch))

    def add_numeric_analysis_section(self, analysis):
        """Add numeric data analysis section"""
        if not analysis['numeric_analysis']:
            return
            
        self.story.append(Paragraph("Numeric Data Analysis", self.heading_style))
        
        numeric_cols = analysis['numeric_analysis']['columns']
        statistics = analysis['numeric_analysis']['statistics']
        
        # Create statistics table
        stat_data = [['Column', 'Count', 'Mean', 'Std', 'Min', 'Max']]
        
        for col in numeric_cols:
            col_stats = statistics[col]
            stat_data.append([
                col,
                f"{col_stats['count']:.0f}",
                f"{col_stats['mean']:.2f}",
                f"{col_stats['std']:.2f}",
                f"{col_stats['min']:.2f}",
                f"{col_stats['max']:.2f}"
            ])
        
        stat_table = Table(stat_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        stat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        self.story.append(stat_table)
        self.story.append(Spacer(1, 0.3*inch))

    def add_categorical_analysis_section(self, analysis):
        """Add categorical data analysis section"""
        if not analysis['categorical_analysis']:
            return
            
        self.story.append(Paragraph("Categorical Data Analysis", self.heading_style))
        
        for col, info in analysis['categorical_analysis'].items():
            self.story.append(Paragraph(f"{col}", self.subheading_style))
            
            # Create value counts table
            value_data = [['Value', 'Count']]
            for value, count in list(info['value_counts'].items())[:10]:  # Top 10 values
                value_str = str(value)
                if len(value_str) > 30:
                    value_str = value_str[:30] + '...'
                value_data.append([value_str, str(count)])
            
            value_table = Table(value_data, colWidths=[3*inch, 1*inch])
            value_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            self.story.append(value_table)
            self.story.append(Spacer(1, 0.2*inch))
            
            # Add summary info
            summary_text = f"Unique values: {info['unique_values']}, Most frequent: {info['most_frequent']}"
            self.story.append(Paragraph(summary_text, self.styles['Normal']))
            self.story.append(Spacer(1, 0.3*inch))

    def add_charts_section(self, charts):
        """Add charts to the report"""
        if not charts:
            return
            
        self.story.append(PageBreak())
        self.story.append(Paragraph("Data Visualizations", self.heading_style))
        
        for chart_filename in charts:
            try:
                # Check if file exists before adding to report
                if os.path.exists(chart_filename):
                    # Add chart to report
                    img = Image(chart_filename, width=7*inch, height=5.25*inch)
                    self.story.append(img)
                    self.story.append(Spacer(1, 0.3*inch))
                else:
                    print(f"Warning: Chart file {chart_filename} not found")
            except Exception as e:
                print(f"Error adding chart {chart_filename}: {e}")
        
        # Clean up temporary chart files after PDF generation
        self.temp_chart_files = charts
    
    def add_conclusion_section(self, analysis):
        """Add conclusion/insights section with dynamic logic"""
        self.story.append(PageBreak())
        self.story.append(Paragraph("Conclusion & Insights", self.heading_style))

        insights = []

        # Insight 1: Most common categorical column + its top value
        if analysis['categorical_analysis']:
            top_col = max(analysis['categorical_analysis'].items(), key=lambda x: x[1]['unique_values'])[0]
            top_val = analysis['categorical_analysis'][top_col]['most_frequent']
            insights.append(f"In the column <b>{top_col}</b>, the most frequent value is <b>{top_val}</b>.")

        # Insight 2: Column with highest missing values
        missing = analysis['basic_info']['missing_values']
        if any(missing.values()):
            most_missing_col = max(missing, key=missing.get)
            insights.append(f"The column <b>{most_missing_col}</b> has the highest number of missing values: <b>{missing[most_missing_col]}</b>.")

        # Insight 3: Highest mean numeric column
        if analysis['numeric_analysis']:
            means = analysis['numeric_analysis']['means']
            top_mean_col = max(means, key=means.get)
            insights.append(f"Among numeric columns, <b>{top_mean_col}</b> has the highest mean value of <b>{means[top_mean_col]:.2f}</b>.")

        # Insight 4: Correlation comment
        if analysis['correlations'] is not None:
            corr_matrix = analysis['correlations'].abs()
            corr_matrix.values[np.tril_indices_from(corr_matrix)] = np.nan  # ignore lower triangle
            max_corr = corr_matrix.unstack().idxmax()
            val = corr_matrix.loc[max_corr]
            if not pd.isna(val):
                insights.append(f"The strongest correlation is between <b>{max_corr[0]}</b> and <b>{max_corr[1]}</b> (correlation: {val:.2f}).")

        # Print fallback message if no insights could be generated
        if not insights:
            insights.append("No specific insights could be generated for this dataset. Try enriching the data or analyzing manually.")

        # Add to report
        for insight in insights:
            self.story.append(Paragraph(insight, self.styles['Normal']))
            self.story.append(Spacer(1, 0.2*inch))
            
    def cleanup_temp_files(self):
        """Clean up temporary chart files"""
        for chart_file in self.temp_chart_files:
            try:
                if os.path.exists(chart_file):
                    os.remove(chart_file)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {chart_file}: {e}")

    def generate_report(self):
        """Generate the complete PDF report"""
        if not self.load_data():
            return False
        
        print("Analyzing data...")
        analysis = self.analyze_data()
        
        print("Creating visualizations...")
        charts = self.create_charts(analysis)
        
        print("Building PDF report...")
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(self.output_path, pagesize=letter,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            # Build report sections
            self.add_title_page()
            self.add_summary_section(analysis)
            self.add_column_info_section(analysis)
            self.add_numeric_analysis_section(analysis)
            self.add_categorical_analysis_section(analysis)
            self.add_charts_section(charts)
            self.add_conclusion_section(analysis)

            
            # Build PDF
            doc.build(self.story)
            
            print(f"Report generated successfully: {self.output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return False
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()


def main():
    """Main function"""
    print("CSV Data Analysis & PDF Report Generator")
    print("=" * 50)
    
    # Get CSV file path
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Ask user for file path or create sample data
        csv_file = input("Enter CSV file path: ").strip()
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return
    
    # Generate report
    generator = CSVReportGenerator(csv_file)
    success = generator.generate_report()
    
    if success:
        print(f"\n✓ Report generated successfully!")
        print(f"📄 Output file: {generator.output_path}")
        print(f"📊 Data analyzed: {csv_file}")
    else:
        print("\n✗ Failed to generate report!")

if __name__ == "__main__":
    main()