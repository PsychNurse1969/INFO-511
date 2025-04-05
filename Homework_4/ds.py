# HW-04: Exploring Relationships Between Education and Disease

"""
INFO 511 - Homework 4
Author: Todd Adams
Purpose: Analyze relationships between education level and disease prevalence
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Task 1: Load and Inspect the Dataset
def load_csv(file_path):
    """
    Loads a CSV file into a pandas DataFrame and prints basic information.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    # Check if the file exists
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.normpath(os.path.join(base_dir, file_path))  # Normalize path

        print(f"🔍 Attempting to load: {full_path}")

        df = pd.read_csv(full_path)
        print("✅ File loaded successfully!")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        df.info()
        return df

    except FileNotFoundError:
        print(f"❌ File not found: {full_path}")
        return None


# Task 2: Summarize Categorical Variables
def summarize_categorical(df, column):
    """
    Summarizes a categorical column by computing frequency counts.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The categorical column to summarize.

    Returns:
    pd.Series: A Series with frequency counts for each category.
    """
    # Compute value counts
    return df[column].value_counts()


# Task 3: Compare Disease Prevalence Across Education Levels
def compute_disease_prevalence(df, education_col, disease_col):
    """
    Computes the prevalence of a disease for each education level.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    education_col (str): The column representing education levels.
    disease_col (str): The column representing disease status.

    Returns:
    pd.DataFrame: A DataFrame with education levels and corresponding disease prevalence.
    """
    # Group by education level and calculate mean disease prevalence
    return df.groupby(education_col)[disease_col].mean().reset_index()


# Task 4: Visualize Disease Prevalence
def plot_disease_prevalence(
    df,
    education_col,
    prevalence_col,
    title="Disease Prevalence by Education Level"
):
    """
    Creates a bar plot to visualize disease prevalence across education levels.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    education_col (str): The column representing education levels.
    prevalence_col (str): The column representing disease prevalence.
    title (str): The title of the plot.

    Returns:
    None
    """
    # Create a seaborn bar plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=education_col, y=prevalence_col, data=df, palette="Blues_d")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Education Level", fontsize=12)
    ax.set_ylabel("Disease Prevalence", fontsize=12)
    plt.tight_layout()
    plt.show()


# Task 5: Statistical Analysis of Disease Prevalence
def perform_chi_squared_test(df, education_col, disease_col):
    """
    Performs a chi-squared test for independence between education levels and disease prevalence.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    education_col (str): The column representing education levels.
    disease_col (str): The column representing disease status.

    Returns:
    tuple: A tuple containing the chi-squared statistic, p-value, and degrees of freedom.
    """
    # Create a contingency table and perform chi-squared test
    # Create a contingency table for education levels and disease status
    contingency_table = pd.crosstab(df[education_col], df[disease_col])
    # Perform the chi-squared test
    chi2, p, dof, _ = stats.chi2_contingency(contingency_table)
    return chi2, p, dof


# Task 6: Analyze Relationships with Additional Variables
def analyze_relationship(df, numerical_col, disease_col):
    """
    Analyzes the relationship between a numerical variable and disease prevalence.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    numerical_col (str): The numerical column to analyze.
    disease_col (str): The column representing disease status.

    Returns:
        pd.DataFrame: A DataFrame with the mean and the standard deviation 
        of the numerical column for each disease status.
    """
    # Group by disease status and compute mean and standard deviation
    # Calculate mean and standard deviation of the numerical column for each disease status
    summary = (df.groupby(disease_col)[numerical_col]
            .agg(['mean', 'std'])
            .reset_index()
    )
    # Rename columns for clarity
    summary.columns = [disease_col, "Age Mean", "Age Std"]
    return summary
