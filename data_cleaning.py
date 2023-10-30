

def convert_to_int(column):
    try:
        return int(column)
    except ValueError:
        return column
    
def fix_int_type(df):
    return df.apply(lambda col: col.apply(convert_to_int) if col.dtype == 'object' else col)