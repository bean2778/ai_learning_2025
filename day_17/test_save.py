import pandas as pd
import os

print(f"Current directory: {os.getcwd()}")
print(f"Data folder exists: {os.path.exists('data')}")

# Try to save a simple file
df = pd.DataFrame({'a': [1, 2, 3]})
df.to_csv('data/test.csv', index=False)

print(f"Test file exists: {os.path.exists('data/test.csv')}")
print(f"Contents of data folder: {os.listdir('data')}")