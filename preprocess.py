import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, for_training=True):
    df = df.copy()

    # 1. Drop duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)

    # 2. Drop unnecessary columns
    drop_cols = [
        'Claim_ID', 'Bind_Date1', 'Policy_Num', 'Policy_Start_Date',
        'Policy_Expiry_Date', 'Accident_Date', 'DL_Expiry_Date',
        'Claims_Date', 'Vehicle_Registration', 'Check_Point'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # 3. Count and Drop missing values
    missing_values = df.isnull().sum().sum()
    df.dropna(inplace=True)

    # 4. Remove outliers (IQR method)
    outliers_removed = 0
    if for_training:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].nunique() > 10:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                before = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                outliers_removed += before - len(df)

    # 5. Encode target
    if 'Fraud_Ind' in df.columns:
        df['Fraud_Ind'] = df['Fraud_Ind'].map({'Y': 1, 'N': 0})

    # 6. Label encode object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # 7. Print stats
    if for_training:
        print(f"✅ Preprocessing Summary:")
        print(f" - Duplicates removed: {duplicates_removed}")
        print(f" - Missing values dropped: {missing_values}")
        print(f" - Outliers removed: {outliers_removed}")
        print(f" - Final rows: {len(df)}")

    return df
