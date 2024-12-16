def show_all():
    return pd.option_context('display.max_rows', None, 'display.max_columns', None)

def lower_case(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.lower()
    return data