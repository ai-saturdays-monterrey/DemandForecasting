from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH='/home/adrianrdzv/Documentos/fastai/fastai/data/forecasting/'

table_names = ['train', 'test']
tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]
from IPython.display import HTML
for t in tables: display(t.head())
for t in tables: display(DataFrameSummary(t).summary())
train,test = tables
len(train),len(test)

add_datepart(train, "date", drop=False)
add_datepart(test, "date", drop=False)

columns = ["date"]
df = train[columns]
df = df.set_index("date")
df.reset_index(inplace=True)

test = test.set_index("date")
test.reset_index(inplace=True)

cat_vars = ['store','item', 'Year', 'Month', 'Week', 'Day','Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start','Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
contin_vars = []
n = len(test); n

dep = 'sales'
test[dep]=0

for v in cat_vars: test[v] = test[v].astype('category').cat.as_ordered()
for v in cat_vars: train[v] = train[v].astype('category').cat.as_ordered()

for v in contin_vars:
    train[v] = train[v].astype('float32')
    test[v] = test[v].astype('float32')

samp_size = len(train)
joined_samp = train.set_index("date")
joined_test = test.set_index("date")

df, y, nas, mapper = proc_df(joined_samp, 'sales', do_scale=True)
yl = np.log(y)

df_test, _, nas, mapper = proc_df(joined_test, 'sales', do_scale=True, skip_flds=['id'],mapper=mapper, na_dict=nas)

val_idx = np.flatnonzero(
    (df.index<=datetime.datetime(2017,12,31)) & (df.index>=datetime.datetime(2017,10,1)))

def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)

max_y = np.max(y)
y_range=(0,max_y*1.2)

def SMAPE(y_pred,targ):
    return (np.abs(y_pred-targ)/((np.fabs(y_pred)+np.fabs(targ))/2)).mean()                     

md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=256,test_df=df_test)

cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]

m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [2000,800], [0.001,0.01], y_range=y_range)
lr = 1e-3

m.lr_find()
m.sched.plot()

m.fit(lr, 3, metrics=[SMAPE])

x,y=m.predict_with_targs()
SMAPE(x,y)

pred_test=m.predict(True)
joined_test['sales'] = pred_test

#We can save our results in the format specified, with the next chunk of code
csv_fn=f'{PATH}tmp/submission_4agosto.csv'
joined_test[['id','sales']].to_csv(csv_fn, index=False)