import sqlite3
import pandas as pd
import os

for name in ['games_lite','positions_lite']:
    # Connect to SQLite database
    conn = sqlite3.connect(os.path.join(os.getcwd(),'test.db'))
      
    # Load CSV data into Pandas DataFrame
    stud_data = pd.read_csv(os.path.join(os.getcwd(),name+'.csv'),delimiter = ";")
    # Write the data to a sqlite table
    stud_data.to_sql(name, conn, if_exists='replace', index=False)
    # Create a cursor object
    cur = conn.cursor()
    # Fetch and display result
    conn.close()