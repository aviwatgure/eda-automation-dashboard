# 📊 Automated EDA (Exploratory Data Analysis) Dashboard

An interactive web application built with Streamlit that automates the Exploratory Data Analysis process. This tool helps data scientists and analysts quickly analyze and visualize datasets with just a few clicks.

## 🌟 Features

### 📁 Data Import
- Supports multiple file formats:
  - CSV (.csv)
  - Excel (.xlsx, .xls)
  - JSON (.json)
  - Text files (.txt)
- Automatic file type detection and processing
- Error handling for corrupt or invalid files

### 📊 Analysis Features

1. **Basic Information**
   - Dataset shape and size
   - Data preview (head and tail)
   - Detailed info about columns

2. **Missing Values Analysis**
   - Detection of missing values
   - Missing value counts and percentages
   - Clear visualization of missing data

3. **Categorical Data Analysis**
   - Frequency tables
   - Bar charts with color gradients
   - Pie charts
   - Relative frequency analysis

4. **Numerical Data Analysis**
   - Statistical summaries
   - Distribution visualization with histograms
   - Box plots
   - Mean and median indicators
   - Density plots (KDE)

5. **Outlier Detection**
   - Box plot visualization
   - IQR-based outlier detection
   - Outlier statistics and percentages

6. **Bivariate Analysis**
   - Relationship analysis between variables
   - Scatter plots for numerical vs numerical
   - Box plots for numerical vs categorical
   - Bar charts for categorical vs categorical
   - Correlation coefficients

7. **Multivariate Analysis**
   - Correlation matrix heatmap
   - Detailed correlation analysis
   - Pair plots for numerical variables

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eda-automation-dashboard.git
cd eda-automation-dashboard
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run EDA_automation_app.py
```

## 📦 Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## 🎯 Usage

1. Launch the application
2. Upload your dataset using the file uploader
3. Select the analyses you want to perform using the checkboxes in the sidebar
4. Explore the automatically generated visualizations and statistics
5. Interact with the plots and select specific columns for detailed analysis

## 📋 Code Structure

- `EDA_automation_app.py`: Main application file
- `requirements.txt`: List of Python dependencies
- `README.md`: Documentation file

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](link-to-issues).

## 📝 License

This project is [MIT](link-to-license) licensed.

## 👥 Author

Your Name
