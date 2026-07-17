"""Check PostgreSQL connectivity and database existence."""
import psycopg2

conn = psycopg2.connect(
    host='localhost', port=5432, user='postgres', password='1234', dbname='postgres'
)
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT 1 AS connected")
print("Connected:", cur.fetchone())

cur.execute("SELECT 1 FROM pg_database WHERE datname='genie_recruitment'")
exists = cur.fetchone()
print("Database genie_recruitment exists:", exists is not None)

# Check existing tables
if exists:
    cur.close()
    conn.close()
    conn = psycopg2.connect(
        host='localhost', port=5432, user='postgres', password='1234', dbname='genie_recruitment'
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public' ORDER BY table_name
    """)
    tables = [r[0] for r in cur.fetchall()]
    print("Existing tables:", tables)

conn.close()
