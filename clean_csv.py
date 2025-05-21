import pandas as pd
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

input_file = "data/books.csv"
output_file = "data/books_cleaned.csv"

def clean_csv():
    try:
        # Read CSV, handling potential errors
        df = pd.read_csv(input_file, quoting=csv.QUOTE_ALL, escapechar='\\')
        # Strip whitespace from column names
        df.columns = [col.strip() for col in df.columns]
        # Rename columns for consistency
        df = df.rename(columns={
            '  num_pages': 'num_pages',  # Fix spaced column
            'average_rating': 'average_rating',
            'language_code': 'language_code',
            'ratings_count': 'ratings_count',
            'text_reviews_count': 'text_reviews_count',
            'publication_date': 'publication_date'
        })
        # Save cleaned CSV
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Cleaned CSV saved as {output_file} with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error cleaning CSV: {e}")

if __name__ == "__main__":
    clean_csv()