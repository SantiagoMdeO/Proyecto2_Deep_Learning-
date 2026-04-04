import pandas as pd

# 
df_original = pd.read_csv("../data/raw/train_clean.csv")

def clean_loans(text):
    if pd.isna(text):
        return []
    
    text = text.replace(" and ", ", ") # this might not work as they are classified with ", and"    | like it did work, but imma keep this comment just i case
    loans = [l.strip() for l in text.split(",")]

    loans = [l for l in loans if l != ""]
    return list(set(loans))


df_original["Loan_List"] = df_original["Type_of_Loan"].apply(clean_loans)


# Get all unique loan types
all_loans = set()
for row in df_original["Loan_List"]:
    all_loans.update(row)

print(all_loans)

# Create binary columns
for loan in all_loans:
    df_original[loan] = df_original["Loan_List"].apply(lambda x: int(loan in x))

# Drop original columns
df_original = df_original.drop(columns=["Type_of_Loan", "Loan_List"])

# Save new dataset
output_path = "./train_clean_type.csv"
df_original.to_csv(output_path, index=False)

print(f" File saved to: {output_path}")
print(f"shape: {df_original.shape}")
print("New columns addeeeeeddd:", list(all_loans))