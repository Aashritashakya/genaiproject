import os
from dotenv import load_dotenv

load_dotenv()

import sqlite3
import google.generativeai as genai
import streamlit as st

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found! Please set it in your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "models/gemini-1.5-pro"
model = genai.GenerativeModel(MODEL_NAME)

def create_db():
    conn = sqlite3.connect("students.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            gpa REAL
        )
    ''')
    c.execute("DELETE FROM students")
    data = [
        (1, "Aashri", "Computer", 3.8),
        (2, "Ravi", "Electronics", 3.2),
        (3, "Sita", "Computer", 3.9),
        (4, "Hari", "Mechanical", 2.8)
    ]
    c.executemany("INSERT INTO students VALUES (?, ?, ?, ?)", data)
    conn.commit()
    conn.close()

def question_to_sql(question):
    prompt = f"""
You are an assistant that converts natural language questions into valid SQLite SQL queries.

Table schema:
students(id, name, department, gpa)

Convert this question into SQL:
\"\"\"{question}\"\"\"

SQL:
"""
    try:
        response = model.generate_content(prompt)
        sql_text = response.text.strip()
        if "SQL:" in sql_text:
            sql_text = sql_text.split("SQL:")[-1].strip()
        return sql_text
    except Exception as e:
        print(f"API error in question_to_sql: {e}")
        # smarter fallback based on question content
        q = question.lower()
        if "department" in q and "gpa" in q:
            return "SELECT name, department, gpa FROM students WHERE gpa > 3.2;"
        elif "gpa" in q:
            return "SELECT name FROM students WHERE gpa > 3.2;"
        else:
            return "SELECT * FROM students LIMIT 5;"

def run_sql(sql):
    try:
        conn = sqlite3.connect("students.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        return f"SQL Error: {e}"

def local_answer_from_result(question, result):
    if isinstance(result, list) and all(isinstance(row, tuple) for row in result):
        # Format output nicely depending on columns in result
        if len(result) == 0:
            return "No records found."
        # Check if rows have 3 items (name, department, gpa)
        if all(len(row) == 3 for row in result):
            lines = [f"{row[0]} from {row[1]} department has GPA {row[2]}" for row in result]
            return "Students found:\n" + "\n".join(lines)
        # If only name column
        if all(len(row) == 1 for row in result):
            names = ", ".join(row[0] for row in result)
            return f"The students matching your query are: {names}."
    return "Sorry, I could not generate a proper answer."

def summarize(question, result):
    prompt = f"""
You are a helpful assistant.

Question: {question}
SQL Result: {result}

Provide a natural language answer:
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"API error in summarize: {e}")
        return local_answer_from_result(question, result)

st.set_page_config(page_title="Talk to My Database (Google Gemini)", layout="centered")
st.title("💬 Talk to My Database using Google Gemini")

create_db()

question = st.text_input("Ask a question about students:", placeholder="e.g. Who has GPA above 3.5?")

if st.button("Ask") and question:
    with st.spinner("Processing..."):
        try:
            sql = question_to_sql(question)
        except Exception as e:
            st.warning(f"API error during SQL generation: {e}\nUsing fallback SQL.")
            q = question.lower()
            if "department" in q and "gpa" in q:
                sql = "SELECT name, department, gpa FROM students WHERE gpa > 3.2;"
            elif "gpa" in q:
                sql = "SELECT name FROM students WHERE gpa > 3.2;"
            else:
                sql = "SELECT * FROM students LIMIT 5;"

        st.code(sql, language="sql")

        result = run_sql(sql)
        st.write("📊 SQL Result:", result)

        try:
            answer = summarize(question, result)
        except Exception as e:
            st.warning(f"API error during answer generation: {e}")
            answer = local_answer_from_result(question, result)

        st.success("🤖 Answer:")
        st.write(answer)
