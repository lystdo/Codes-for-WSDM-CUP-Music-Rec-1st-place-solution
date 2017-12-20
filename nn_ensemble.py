import os
import pandas as pd

cnt = 0.0
score = 0.0

for item in os.listdir('./temp_nn/'):
    score += float(item.split('_')[1])
    tmp = pd.read_csv('./temp_nn/'+item)
    if cnt == 0:
        preds = tmp
    else:
        preds['target'] += tmp['target']
    cnt += 1.0

score /= cnt
preds['target'] /= cnt
preds.to_csv('./submission/%.5f_%d_ensemble_add.csv.gz'%(score, cnt), index=False, \
        compression='gzip')

