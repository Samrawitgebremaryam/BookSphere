# 📚 BookSphere

**BookSphere** is a smart book recommendation web app that helps you discover your next favorite read! Powered by Neo4j, Streamlit, and Python, BookSphere offers personalized book recommendations using both collaborative and content-based filtering.

---

## 🚀 Features

- **Top Books:** Instantly see the most popular books in the database.
- **Personalized Recommendations:**  
  - **Similar to You:** Get recommendations based on what users with similar tastes enjoyed.
  - **Similar to This Book:** Find books similar in genre and description to a book you love.
- **Interactive UI:**  
  - Clean, modern interface with dark mode support.
  - Dropdowns with autocomplete for easy book selection.
  - Clickable learning modules in the sidebar to understand how recommendations work.
- **Book Details:**  
  - See book covers, authors, ratings, genres, and direct links to Goodreads.

---

## 🛠️ How It Works

1. **Data Storage:**  
   Book and rating data are stored in a Neo4j graph database.
2. **Recommendation Engine:**  
   - **Popularity-Based:** Ranks books by average rating and number of reviews.
   - **Collaborative Filtering:** Finds books liked by users with similar reading habits.
   - **Content-Based Filtering:** Finds books with similar genres and descriptions.
3. **Web App:**  
   Built with Streamlit for a fast, interactive user experience.

---

## 📂 Project Structure

```
BookSphere/
│
├── app.py                  # Main Streamlit web app
├── recommender.py          # Recommendation logic and data loading
├── fetch_books.py          # Scripts to import books/users/ratings into Neo4j
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🖥️ Usage

### 1. **Install Requirements**

```bash
pip install -r requirements.txt
```

### 2. **Set Up Neo4j**

- Make sure you have a running Neo4j instance (local or cloud).
- Add your connection details to a `.env` file:
    ```
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    ```

### 3. **Import Data (First Time Only)**

If you need to load books and ratings into Neo4j, run:
```bash
python fetch_books.py
```

### 4. **Run the App**

```bash
streamlit run app.py
```
Open the provided local URL in your browser.

---

## 🧠 How Recommendations Work

- **Popularity-Based:**  
  Shows books with the highest ratings and most reviews.
- **Similar to You (Collaborative Filtering):**  
  Recommends books liked by users who have similar reading habits to you.
- **Similar to This Book (Content-Based):**  
  Suggests books with similar genres, keywords, or descriptions.

**Example:**  
If you love "The Hobbit", "Similar to You" might recommend "The Lord of the Rings" (because many users liked both), while "Similar to This Book" might recommend other fantasy novels.

---

## 📝 Example User Flow

1. **See Top Books:**  
   Instantly view the most popular books.
2. **Get Recommendations:**  
   - Choose "Similar to You" and pick a book you liked.
   - Or, choose "Similar to This Book" for content-based suggestions.
3. **Learn:**  
   Use the sidebar to understand how each recommendation method works.

---

## 🧩 Technologies Used

- **Python 3**
- **Streamlit** (UI)
- **Neo4j** (Graph database)
- **scikit-learn** (Machine learning)
- **pandas** (Data manipulation)
- **requests, dotenv, nltk** (Utilities)

---

## 🙋 FAQ

**Q: Why do some books not appear for "Similar to You"?**  
A: Collaborative filtering only works for books with enough ratings. The dropdown only shows eligible books.

**Q: Can I use my own dataset?**  
A: Yes! Update the CSV and use `fetch_books.py` to import your data.

---


