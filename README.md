# Exploratory-Data-Analysis-EDA-on-Retail-Sales-Data-
#Description:  In this project, you will work with a dataset containing information about retail sales. The goal is to perform exploratory data analysis (EDA) to uncover patterns, trends, and insights that can help the retail business make informed decisions.
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = kagglehub.dataset_download("mohammadtalib786/retail-sales-dataset")

print("Dataset downloaded at:", path)

files = os.listdir(path)
print("Files inside dataset folder:", files)

file_path = os.path.join(path, files[0])
df = pd.read_csv(file_path)

print("\nFirst 5 rows of dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

print("\nData after cleaning:")
print(df.head())


if 'Total Amount' in df.columns:
    sales_column = 'Total Amount'
elif 'Sales' in df.columns:
    sales_column = 'Sales'
else:
    sales_column = df.select_dtypes(include='number').columns[-1]

print("\nSales Statistics")
print("Mean:", df[sales_column].mean())
print("Median:", df[sales_column].median())
print("Mode:", df[sales_column].mode()[0])
print("Standard Deviation:", df[sales_column].std())


plt.figure()
plt.hist(df[sales_column], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()


if 'Product Category' in df.columns:

    product_sales = df.groupby('Product Category')[sales_column].sum()

    plt.figure()
    product_sales.plot(kind='bar')
    plt.title("Sales by Product Category")
    plt.xlabel("Product Category")
    plt.ylabel("Total Sales")
    plt.show()


if 'Gender' in df.columns:

    gender_sales = df.groupby('Gender')[sales_column].sum()

    plt.figure()
    gender_sales.plot(kind='bar')
    plt.title("Sales by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Total Sales")
    plt.show()


if 'Age' in df.columns:

    plt.figure()
    plt.hist(df['Age'], bins=10)
    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()


numeric_df = df.select_dtypes(include='number')

plt.figure()
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
