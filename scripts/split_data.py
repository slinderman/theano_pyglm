import cPickle
import os
from utils.io import segment_data

data_dir = '/group/hips/scott/pyglm/data/synth/dist/N16T300/2014_07_16-17_07'

with open(os.path.join(data_dir, 'data.pkl')) as f:
    data = cPickle.load(f)
  

data_test = segment_data(data, (240,300))
with open(os.path.join(data_dir, 'data_test.pkl'), 'w') as f:
    cPickle.dump(data_test, f, protocol=-1)

ts = [15,30,60,120,180, 240]
datas = []

data_test
for t in ts:
    datas.append(segment_data(data, (0,t)))
        
for (t,d) in zip(ts, datas):
    with open(os.path.join(data_dir, 'data_%d.pkl' % t), 'w') as f:
        cPickle.dump(d, f, protocol=-1)
        

