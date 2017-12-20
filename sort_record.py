import pandas as pd

file_name = 'nn_record.csv'

record = pd.read_csv(file_name)

column = 'val_auc'

record = record.sort_values(by=column, ascending=False)
if len(record) > 25:
    record = record[:25]
    
record.to_csv(file_name, index=False)
