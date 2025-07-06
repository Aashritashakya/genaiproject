import sqlite3
import streamlit as st
import requests
import re

# === Ollama Config ===
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  # You can use llama3, codellama, phi, etc.

# === Step 1: Create sample database ===
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
    c.execute("DELETE FROM students")  # reset each time
    data = [
        (1, "Aashri", "Computer", 3.8),
        (2, "Ravi", "Electronics", 3.2),
        (3, "Sita", "Computer", 3.9),
        (4, "Hari", "Mechanical", 2.8),
        (5, "Nina", "Civil", 2.1)
    ]
    c.executemany("INSERT INTO students VALUES (?, ?, ?, ?)", data)
    conn.commit()
    conn.close()

# === Step 2: Talk to Ollama API ===
def query_ollama(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })

        data = response.json()
        if "response" in data:
            return data["response"].strip()
        elif "error" in data:
            return f"Ollama Error: {data['error']}"
        else:
            return "Ollama returned unexpected format."
    except Exception as e:
        return f"Ollama Request Failed: {e}"

# === Step 3: Convert question to SQL safely ===
def question_to_sql(question):
    prompt = f"""
You are a helpful assistant that converts natural language questions into valid SQLite SQL.

Assume a table named 'students' with the following columns:
- id (integer)
- name (text)
- department (text)
- gpa (real)

Convert this question to a single SELECT SQL query:
\"\"\"{question}\"\"\"

SQL:
"""
    raw_sql = query_ollama(prompt)

    # Extract clean SQL from markdown code block
    if "```" in raw_sql:
        cleaned = re.findall(r"```(?:sql)?\s*(.*?)\s*```", raw_sql, re.DOTALL)
        if cleaned:
            return cleaned[0].strip()
    return raw_sql.strip()

# === Step 4: Run SQL on SQLite ===
def run_sql(sql):
    try:
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute(sql)
        result = c.fetchall()
        conn.close()
        return result
    except Exception as e:
        return f"SQL Error: {e}"

# === Step 5: Summarize result using LLM ===
def summarize(question, result):
    prompt = f"""
You are a helpful assistant. A user asked: "{question}".

Here is the SQL result: {result}

Explain the result in one short sentence.
"""
    return query_ollama(prompt)

# === UI ===
st.set_page_config(page_title="🧠 Talk to SQLite (Ollama)", layout="centered")
st.title("💬 Talk to Database with Ollama")

# Build DB
create_db()

question = st.text_input("Ask a question about students:", placeholder="e.g. who has GPA > 3.2")

if st.button("Ask") and question:
    with st.spinner("💡 Generating SQL..."):
        sql = question_to_sql(question)
        st.code(sql, language="sql")

        if "select" not in sql.lower():
            st.error("❌ Failed to generate valid SELECT query.")
        else:
            result = run_sql(sql)
            if isinstance(result, str) and result.startswith("SQL Error"):
                st.error(result)
            else:
                st.write("📊 SQL Result:", result)
                with st.spinner("🤖 Generating answer..."):
                    answer = summarize(question, result)
                    st.success("🧠 Answer:")
                    st.write(answer)
