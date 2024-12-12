import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from io import StringIO

# Setup OpenAI API
def setup_openai():
    try:
        openai.api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

# Load dataset
def load_dataset(filename):
    try:
        return pd.read_csv(filename)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

# Analyze dataset
def analyze_dataset(df):
    analysis = {}
    analysis['summary'] = df.describe(include='all').to_dict()
    analysis['missing_values'] = df.isnull().sum().to_dict()
    if df.select_dtypes(include=['number']).shape[1] > 0:
        analysis['correlation'] = df.corr().to_dict()
    return analysis

# Visualize data
def visualize_data(df, analysis):
    visualizations = []
    if 'correlation' in analysis:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        chart_path = "correlation_matrix.png"
        plt.savefig(chart_path)
        plt.close()
        visualizations.append(chart_path)
    
    # Histogram of numerical columns
    for column in df.select_dtypes(include=['number']).columns:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        chart_path = f"histogram_{column}.png"
        plt.savefig(chart_path)
        plt.close()
        visualizations.append(chart_path)

    return visualizations

# Generate story
def generate_story(filename, analysis, visualizations):
    try:
        prompt = (
            f"You are a data scientist. I analyzed the dataset '{filename}' and here are the results:\n"
            f"Analysis Summary:\n{analysis}\n"
            f"Visualizations available: {visualizations}\n"
            "Write a detailed story about the data, key findings, and implications."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating story: {e}")
        sys.exit(1)

# Save README
def save_readme(content, visualizations):
    with open("README.md", "w") as f:
        f.write(content)
        for vis in visualizations:
            f.write(f"\n![{vis}]({vis})")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Automated analysis of datasets.")
    parser.add_argument("filename", help="Path to the CSV dataset file")
    args = parser.parse_args()

    setup_openai()
    
    print("Loading dataset...")
    df = load_dataset(args.filename)

    print("Analyzing dataset...")
    analysis = analyze_dataset(df)

    print("Visualizing data...")
    visualizations = visualize_data(df, analysis)

    print("Generating story...")
    story = generate_story(args.filename, analysis, visualizations)

    print("Saving results...")
    save_readme(story, visualizations)

    print("Analysis complete. Results saved in README.md and visualizations saved as PNG files.")

if __name__ == "__main__":
    main()

