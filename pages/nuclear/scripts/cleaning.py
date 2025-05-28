import pandas as pd
import re

def clean_reactor_types(input_csv: str, output_csv: str) -> pd.DataFrame:
    # 1) Load raw CSV
    df = pd.read_csv(input_csv)
    
    # 2) Rename columns
    df = df.rename(columns={
        'vteTypes of nuclear fission reactor': 'category',
        'vteTypes of nuclear fission reactor.1': 'models'
    })
    
    # 3) If models is blank, use the category (to avoid NaNs)
    df['models'] = df['models'].fillna(df['category'])
    
    # 4) Drop the fusion row (and any other rows you don’t need)
    df = df[~df['category'].str.contains('fusion', case=False, na=False)]
    
    # 5) Drop exact duplicates
    df = df.drop_duplicates()
    
    # 6) (Optional) split the models string into a list of model names
    #    Here we split on one-or-more whitespace or commas—adjust to your delimiter
    df['model_list'] = df['models'].apply(lambda s: re.split(r'[\s,]+', s.strip()))
    
    # 7) Save a cleaned version for downstream use
    df.to_csv(output_csv, index=False)
    
    return df

if __name__ == "__main__":
    cleaned = clean_reactor_types(
        input_csv="data/reactors_raw.csv",
        output_csv="data/reactors_clean.csv"
    )
    print(cleaned.head())
