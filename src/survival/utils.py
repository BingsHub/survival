import pandas as pd
def show_all():
    return pd.option_context('display.max_rows', None, 'display.max_columns', None)

def lower_case(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.lower()
    return data
def geonames_cleaner(df, cols):
    for col in cols:
        df[col] = df[col].str.lower().str.replace('  ', ' ').str.replace('-', ' ').str.replace("'", '')\
            .str.replace('é', 'e').str.replace('ë', 'e').str.replace('ö', 'o').str.replace('ü', 'u')\
            .str.replace('ï', 'i').str.replace('î', 'i').str.replace('ç', 'c').str.replace('à', 'a')\
            .str.replace('â', 'a').str.replace('ê', 'e').str.replace('ô', 'o').str.replace('û', 'u')\
            .str.replace('è', 'e').str.replace('.', '').str.replace('(', '').str.replace(')', '')\
            .str.replace(',', '').str.strip()
    return df