import pandas as pd

file_path = "data/books.csv"
try:
    df = pd.read_csv(file_path, nrows=1)
    print("Column names:", df.columns.tolist())
except Exception as e:
    print(f"Error reading CSV: {e}")