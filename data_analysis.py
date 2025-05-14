import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set the style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

CSV_PATH = r'C:/Users/Steve/Downloads/Compressed/archive/Iris.csv'

def load_and_explore_data():
    """Load and explore the Iris dataset from CSV"""
    print(f"Loading and exploring the Iris dataset from {CSV_PATH} ...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"File not found: {CSV_PATH}")
        raise
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

    # Show columns and adjust if necessary
    print("\nColumns in the dataset:", df.columns.tolist())
    # Try to standardize column names if needed
    col_map = {
        'Species': 'species',
        'sepal_length': 'sepal length (cm)',
        'sepal_width': 'sepal width (cm)',
        'petal_length': 'petal length (cm)',
        'petal_width': 'petal width (cm)',
        'SepalLengthCm': 'sepal length (cm)',
        'SepalWidthCm': 'sepal width (cm)',
        'PetalLengthCm': 'petal length (cm)',
        'PetalWidthCm': 'petal width (cm)',
    }
    df = df.rename(columns=col_map)
    # Remove Id column if present
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Check data types and missing values
    print("\nDataset information:")
    print(df.info())
    
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Clean missing values if any
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print("\nRows with missing values have been dropped.")
    
    return df

def basic_analysis(df):
    """Perform basic statistical analysis"""
    print("\nPerforming basic statistical analysis...")
    
    # Basic statistics for numerical columns
    print("\nBasic statistics for numerical columns:")
    print(df.describe())
    
    # Group by species and calculate mean for each feature
    print("\nMean values by species:")
    print(df.groupby('species').mean())
    
    return df

def create_visualizations(df):
    """Create various visualizations of the data"""
    print("\nCreating visualizations...")
    
    # Create a figure with subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Line chart - Mean feature values by species
    plt.subplot(2, 2, 1)
    species_means = df.groupby('species').mean()
    species_means.plot(kind='line', marker='o')
    plt.title('Mean Feature Values by Species')
    plt.xlabel('Species')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Bar chart - Mean sepal length by species
    plt.subplot(2, 2, 2)
    sns.barplot(x='species', y='sepal length (cm)', data=df)
    plt.title('Mean Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length (cm)')
    
    # 3. Histogram - Distribution of petal length
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='petal length (cm)', hue='species', multiple='stack')
    plt.title('Distribution of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Count')
    
    # 4. Scatter plot - Sepal length vs Petal length
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.savefig('iris_analysis.png')
    plt.close()

def main():
    try:
        # Load and explore the data
        df = load_and_explore_data()
        
        # Perform basic analysis
        df = basic_analysis(df)
        
        # Create visualizations
        create_visualizations(df)
        
        print("\nAnalysis complete! Visualizations have been saved as 'iris_analysis.png'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 