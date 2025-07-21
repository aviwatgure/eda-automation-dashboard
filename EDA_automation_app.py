# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="📊 EDA Automation Dashboard", layout="wide")
st.title("📊 Automated EDA Dashboard")

    # Sidebar - User controls
with st.sidebar:
    st.header("Hello, Lets analysis data ")
    
    # File upload section with multiple file types
    uploaded_file = st.file_uploader(
        "📂 Upload Dataset", 
        type=["csv", "xlsx", "xls", "json", "txt"]
    )
    
    if uploaded_file is not None:
        # Get the file type from the file extension
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            # Read different file types
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_type == 'json':
                df = pd.read_json(uploaded_file)
            elif file_type == 'txt':
                # Try to read txt file as CSV first
                try:
                    df = pd.read_csv(uploaded_file, sep='\t')
                except:
                    df = pd.read_csv(uploaded_file, sep=',')
            
            st.success(f"✅ {uploaded_file.name} uploaded and processed successfully!")
        except Exception as e:
            st.error(f"❌ Error reading the file: {str(e)}")
            st.stop()
            
    show_basic = st.checkbox("Show Basic Info", value=True)
    show_missing = st.checkbox("Missing Value Analysis")
    # Data Cleaning feature removed
    show_cat_analysis = st.checkbox("Categorical Analysis")
    show_num_analysis = st.checkbox("Numerical Analysis")
    show_outlier = st.checkbox("Outlier Analysis")
    show_bivariate = st.checkbox("Bivariate Analysis")
    show_multivariate = st.checkbox("Multi-variate Analysis")# Main panel - Data Preview
if uploaded_file:

    # Column separation
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    col1, col2 = st.columns([1, 3])  # 25% sidebar-like + 75% main display

    # ===============================
    # Step 3: Basic Info
    # ===============================
    if show_basic:
        with col2:
            st.subheader("📌 Basic Overview")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Size:** {df.size}")
            st.write(f"**Length (rows):** {len(df)}")
            st.write("**Head:**")
            st.dataframe(df.head())
            st.write("**Tail:**")
            st.dataframe(df.tail())
            st.write("**Info:**")
            st.text(df.info())

    # ===============================
    # Step 4: Missing Values + Imputation
    # ===============================
    if show_missing:
        st.subheader("🧱 Missing Values Analysis")
        # Calculate missing values and show only columns with missing values
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            st.write("Columns with missing values:")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df)
        else:
            st.success("✅ No missing values found in the dataset!")

    # Data Cleaning feature removed

    # ===============================
    # Step 6: Categorical Column Analysis
    # ===============================
    if show_cat_analysis:
        with col1:
            st.subheader("🔤 Categorical Analysis")
            cat_col = st.selectbox("Categorical Column", cat_cols)
            cat_chart = st.radio("Chart Type", ["Bar", "Pie"])
        with col2:
            st.write("🔢 Frequency Table")
            st.write(df[cat_col].value_counts())

            st.write("📊 Relative Frequency")
            st.write(df[cat_col].value_counts(normalize=True))

            fig = plt.figure(figsize=(10, 5))
            if cat_chart == "Bar":
                # Get value counts and create color gradient
                value_counts = df[cat_col].value_counts()
                colors = plt.cm.viridis(np.linspace(0, 0.8, len(value_counts)))
                ax = value_counts.plot(kind="bar", color=colors)
                plt.title(f"Distribution of {cat_col}")
                plt.xlabel(cat_col)
                plt.ylabel("Count")
                plt.xticks(rotation=45)
            else:
                df[cat_col].value_counts().plot.pie(autopct="%1.1f%%")
            plt.tight_layout()
            st.pyplot(fig)

    # ===============================
    # Step 7: Numerical Column Analysis
    # ===============================
    if show_num_analysis:
        with col1:
            st.subheader("🔢 Numerical Analysis")
            num_col = st.selectbox("Numerical Column", num_cols)
            dist_type = st.radio("Distribution", ["Histogram", "Boxplot"])
        with col2:
            st.write("📈 Statistical Summary")
            st.write(df[num_col].describe())

            fig = plt.figure(figsize=(10, 6))
            if dist_type == "Histogram":
                # Create histogram using seaborn for better default styling
                sns.histplot(data=df[num_col], stat='density', 
                           bins=min(50, int(np.sqrt(len(df[num_col])))),
                           color='skyblue', alpha=0.6)
                
                # Add KDE plot
                sns.kdeplot(data=df[num_col], color='darkred', linewidth=2)
                
                # Add mean and median lines
                plt.axvline(df[num_col].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df[num_col].mean():.2f}')
                plt.axvline(df[num_col].median(), color='green', linestyle='--', 
                          label=f'Median: {df[num_col].median():.2f}')
                
                plt.title(f"Distribution of {num_col}")
                plt.xlabel(num_col)
                plt.ylabel("Density")
                plt.legend()
            else:
                sns.boxplot(x=df[num_col], color='skyblue')
                plt.title(f"Boxplot of {num_col}")
            
            plt.tight_layout()
            st.pyplot(fig)

    # ===============================
    # Step 8: Outlier Treatment
    # ===============================
    if show_outlier:
        with col1:
            st.subheader("🚨 Outlier Detection")
            out_col = st.selectbox("Column to Check", num_cols)
        
            # Calculate outlier statistics
            Q1 = df[out_col].quantile(0.25)
            Q3 = df[out_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[out_col] < lower) | (df[out_col] > upper)][out_col]
            
            # Display outlier statistics
            st.write("**Outlier Statistics:**")
            if len(outliers) > 0:
                st.write(f"• Number of outliers: {len(outliers)}")
                st.write(f"• Percentage: {(len(outliers)/len(df)*100):.2f}%")
            else:
                st.success("✅ No outliers detected")
                
        with col2:
            # Display smaller boxplot for outlier visualization
            fig = plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[out_col])
            plt.title("Boxplot for Outlier Detection")
            plt.tight_layout()
            st.pyplot(fig)

    # ===============================
    # Step 9: Bivariate Analysis
    # ===============================
    if show_bivariate:
        st.markdown("---")
        st.header("📊 Bivariate Analysis")

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Select Columns")
            x_col = st.selectbox("Select X (Independent)", df.columns)
            y_col = st.selectbox("Select Y (Dependent)", df.columns)

            x_dtype = df[x_col].dtype
            y_dtype = df[y_col].dtype

        with col2:
            st.subheader("📈 Relationship Plot")

            # Numerical vs Numerical
            if np.issubdtype(x_dtype, np.number) and np.issubdtype(y_dtype, np.number):
                st.write("▶️ **Numerical vs Numerical** → Scatter Plot")
                fig = plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x=x_col, y=y_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # Add correlation coefficient
                correlation = df[x_col].corr(df[y_col])
                st.write(f"**Correlation coefficient:** {correlation:.2f}")

            # Numerical vs Categorical
            elif np.issubdtype(x_dtype, np.number) and y_dtype == object:
                st.write("▶️ **Numerical vs Categorical** → Box Plot")
                fig = plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[y_col], y=df[x_col])
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            elif np.issubdtype(y_dtype, np.number) and x_dtype == object:
                st.write("▶️ **Categorical vs Numerical** → Box Plot")
                fig = plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[x_col], y=df[y_col])
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            # Categorical vs Categorical
            elif x_dtype == object and y_dtype == object:
                st.write("▶️ **Categorical vs Categorical** → Grouped Bar Chart")
                cross_tab = pd.crosstab(df[x_col], df[y_col])
                fig = plt.figure(figsize=(10, 6))
                cross_tab.plot(kind='bar', stacked=True, ax=plt.gca())
                plt.xticks(rotation=45)
                plt.legend(title=y_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)

                # Display contingency table
                st.write("**Contingency Table:**")
                st.dataframe(cross_tab)

            else:
                st.warning("⚠️ Unsupported column type combination.")

    # ===============================
    # Step 10: Multivariate Analysis
    # ===============================
    if show_multivariate:
        with col2:
            st.subheader("🔗 Correlation Analysis")
            
            # Correlation heatmap
            try:
                plt.figure(figsize=(12, 8))
                corr_matrix = df.corr(numeric_only=True)
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
                plt.title("Correlation Matrix Heatmap")
                plt.tight_layout()
                st.pyplot(plt)
                
                # Show correlation values in a dataframe format
                st.write("**Detailed Correlation Matrix:**")
                st.dataframe(corr_matrix.round(2))
            except Exception as e:
                st.warning("Could not generate correlation heatmap.")

            st.subheader("📊 PairPlot")
            if st.button("Generate Pairplot"):
                try:
                    fig2 = sns.pairplot(df[num_cols].dropna())
                    st.pyplot(fig2)
                except Exception as e:
                    st.warning("Could not generate pairplot. Please check your data.")
