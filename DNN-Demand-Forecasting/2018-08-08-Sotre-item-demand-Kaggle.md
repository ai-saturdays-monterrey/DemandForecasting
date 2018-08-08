
# Store item Demand Forecasting Challenge


```python
%matplotlib inline
%reload_ext autoreload
%autoreload 2
```

Import fastai libraries to use columnar data in DNN


```python
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH='/home/adrianrdzv/Documentos/fastai/fastai/data/forecasting/'
```

Feature Space:
* train: Los datos de entranamiento proporcionados por Kaggle
* submmission: ejemplo de salida
* test: los datos de salida que generaremos

## Analyze data previous to data cleansing


```python
table_names = ['train', 'test']
```


```python
tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]
```


```python
from IPython.display import HTML
```


```python
for t in tables: display(t.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>store</th>
      <th>item</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-02</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-03</td>
      <td>1</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-04</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-05</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>store</th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2018-01-01</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2018-01-02</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2018-01-03</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2018-01-04</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2018-01-05</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
for t in tables: display(DataFrameSummary(t).summary())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>store</th>
      <th>item</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>NaN</td>
      <td>913000</td>
      <td>913000</td>
      <td>913000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>5.5</td>
      <td>25.5</td>
      <td>52.2503</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>2.87228</td>
      <td>14.4309</td>
      <td>28.8011</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>3</td>
      <td>13</td>
      <td>30</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>5.5</td>
      <td>25.5</td>
      <td>47</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>8</td>
      <td>38</td>
      <td>70</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>10</td>
      <td>50</td>
      <td>231</td>
    </tr>
    <tr>
      <th>counts</th>
      <td>913000</td>
      <td>913000</td>
      <td>913000</td>
      <td>913000</td>
    </tr>
    <tr>
      <th>uniques</th>
      <td>1826</td>
      <td>10</td>
      <td>50</td>
      <td>213</td>
    </tr>
    <tr>
      <th>missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>missing_perc</th>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
    </tr>
    <tr>
      <th>types</th>
      <td>categorical</td>
      <td>numeric</td>
      <td>numeric</td>
      <td>numeric</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>store</th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45000</td>
      <td>NaN</td>
      <td>45000</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22499.5</td>
      <td>NaN</td>
      <td>5.5</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12990.5</td>
      <td>NaN</td>
      <td>2.87231</td>
      <td>14.431</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11249.8</td>
      <td>NaN</td>
      <td>3</td>
      <td>13</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22499.5</td>
      <td>NaN</td>
      <td>5.5</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33749.2</td>
      <td>NaN</td>
      <td>8</td>
      <td>38</td>
    </tr>
    <tr>
      <th>max</th>
      <td>44999</td>
      <td>NaN</td>
      <td>10</td>
      <td>50</td>
    </tr>
    <tr>
      <th>counts</th>
      <td>45000</td>
      <td>45000</td>
      <td>45000</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>uniques</th>
      <td>45000</td>
      <td>90</td>
      <td>10</td>
      <td>50</td>
    </tr>
    <tr>
      <th>missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>missing_perc</th>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
    </tr>
    <tr>
      <th>types</th>
      <td>numeric</td>
      <td>categorical</td>
      <td>numeric</td>
      <td>numeric</td>
    </tr>
  </tbody>
</table>
</div>


## Data Cleaning / Feature Engineering

Adjusting the data to be use in the DNN


```python
train,test = tables
```


```python
len(train),len(test)
```




    (913000, 45000)



La siguiente función "add_datepart" nos proporciona una gran variedad de variables temporales que pueden capturar comportamientos de estacioanlidad y demas caracteristicas temporales en los datos

The next function "add_datepart" will give us a variety of temporary variables who captures seasonality behaviours and stuff related to time series


```python
add_datepart(train, "date", drop=False)
add_datepart(test, "date", drop=False)
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>store</th>
      <th>item</th>
      <th>sales</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>1356998400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-02</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357084800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-03</td>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357171200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-04</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357257600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-05</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357344000</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>store</th>
      <th>item</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2018-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>1514764800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2018-01-02</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1514851200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2018-01-03</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1514937600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2018-01-04</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515024000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2018-01-05</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515110400</td>
    </tr>
  </tbody>
</table>
</div>




```python
columns = ["date"]
df = train[columns]
df = df.set_index("date")
df.reset_index(inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-05</td>
    </tr>
  </tbody>
</table>
</div>




```python
test = test.set_index("date")
test.reset_index(inplace=True)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>id</th>
      <th>store</th>
      <th>item</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>1514764800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1514851200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1514937600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515024000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-05</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515110400</td>
    </tr>
  </tbody>
</table>
</div>



Para poder hacer uso de las funciones de la red neuronal debemos identificar que variables trataremos como categoricas y cuales como continuas, para este caso todas las trataremos como categoricas


```python
#cat_vars = ['store', 'item', 'Dayofweek', 'Year', 'Month', 'Day']
cat_vars = ['store','item', 'Year', 'Month', 'Week', 'Day','Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start','Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
contin_vars = []
n = len(test); n
```




    45000




```python
dep = 'sales'
test[dep]=0
join_test=test.copy() # Pendiente eliminar
```

Debemos modificar explicitamente el dataframe para que cada variable categorica se exprese como tal, dado que asi espera las funciones

We need to modify explictly the dataframe in order to every categorical variable be tagged as categorical, and the same for continuos


```python
for v in cat_vars: test[v] = test[v].astype('category').cat.as_ordered()
```


```python
for v in cat_vars: train[v] = train[v].astype('category').cat.as_ordered()
```


```python
for v in contin_vars:
    train[v] = train[v].astype('float32')
    test[v] = test[v].astype('float32')
```

We are going to use the full dataset to train our model


```python
samp_size = len(train)
joined_samp = train.set_index("date")

```


```python
#joined_samp = joined_samp.set_index("date")
joined_test = test.set_index("date")
```

We can now process our data...


```python
joined_samp.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store</th>
      <th>item</th>
      <th>sales</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>1356998400</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357084800</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357171200</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357257600</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357344000</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>2013</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357430400</td>
    </tr>
    <tr>
      <th>2013-01-07</th>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>2013</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357516800</td>
    </tr>
    <tr>
      <th>2013-01-08</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>2013</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>8</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357603200</td>
    </tr>
    <tr>
      <th>2013-01-09</th>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>2013</td>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>9</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357689600</td>
    </tr>
    <tr>
      <th>2013-01-10</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>2013</td>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>3</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1357776000</td>
    </tr>
  </tbody>
</table>
</div>




```python
joined_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>store</th>
      <th>item</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
      <th>sales</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>1514764800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1514851200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1514937600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515024000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515110400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-06</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515196800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-07</th>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515283200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515369600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-09</th>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515456000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-10</th>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>2</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515542400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-11</th>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>3</td>
      <td>11</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515628800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-12</th>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>12</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515715200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-13</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>5</td>
      <td>13</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515801600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-14</th>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>14</td>
      <td>6</td>
      <td>14</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515888000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-15</th>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>15</td>
      <td>0</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515974400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-16</th>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>1</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516060800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>17</td>
      <td>2</td>
      <td>17</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516147200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>18</td>
      <td>3</td>
      <td>18</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516233600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-19</th>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>19</td>
      <td>4</td>
      <td>19</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516320000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-20</th>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>20</td>
      <td>5</td>
      <td>20</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516406400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-21</th>
      <td>20</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>21</td>
      <td>6</td>
      <td>21</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516492800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-22</th>
      <td>21</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>22</td>
      <td>0</td>
      <td>22</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516579200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-23</th>
      <td>22</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>23</td>
      <td>1</td>
      <td>23</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516665600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-24</th>
      <td>23</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>24</td>
      <td>2</td>
      <td>24</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516752000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-25</th>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>25</td>
      <td>3</td>
      <td>25</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516838400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-26</th>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>26</td>
      <td>4</td>
      <td>26</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516924800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-27</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>27</td>
      <td>5</td>
      <td>27</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1517011200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-28</th>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>28</td>
      <td>6</td>
      <td>28</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1517097600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-29</th>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>5</td>
      <td>29</td>
      <td>0</td>
      <td>29</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1517184000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-30</th>
      <td>29</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>5</td>
      <td>30</td>
      <td>1</td>
      <td>30</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1517270400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-03-02</th>
      <td>44970</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
      <td>61</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1519948800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-03</th>
      <td>44971</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>9</td>
      <td>3</td>
      <td>5</td>
      <td>62</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520035200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-04</th>
      <td>44972</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>9</td>
      <td>4</td>
      <td>6</td>
      <td>63</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520121600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-05</th>
      <td>44973</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>64</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520208000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-06</th>
      <td>44974</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>6</td>
      <td>1</td>
      <td>65</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520294400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-07</th>
      <td>44975</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>2</td>
      <td>66</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520380800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-08</th>
      <td>44976</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>67</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520467200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-09</th>
      <td>44977</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>9</td>
      <td>4</td>
      <td>68</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520553600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-10</th>
      <td>44978</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>69</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520640000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-11</th>
      <td>44979</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>11</td>
      <td>6</td>
      <td>70</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520726400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-12</th>
      <td>44980</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>12</td>
      <td>0</td>
      <td>71</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520812800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-13</th>
      <td>44981</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>13</td>
      <td>1</td>
      <td>72</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520899200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-14</th>
      <td>44982</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>14</td>
      <td>2</td>
      <td>73</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520985600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-15</th>
      <td>44983</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>15</td>
      <td>3</td>
      <td>74</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521072000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-16</th>
      <td>44984</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>16</td>
      <td>4</td>
      <td>75</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521158400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-17</th>
      <td>44985</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>17</td>
      <td>5</td>
      <td>76</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521244800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-18</th>
      <td>44986</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>18</td>
      <td>6</td>
      <td>77</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521331200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-19</th>
      <td>44987</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>19</td>
      <td>0</td>
      <td>78</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521417600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-20</th>
      <td>44988</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>20</td>
      <td>1</td>
      <td>79</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521504000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-21</th>
      <td>44989</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>21</td>
      <td>2</td>
      <td>80</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521590400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-22</th>
      <td>44990</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>22</td>
      <td>3</td>
      <td>81</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521676800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-23</th>
      <td>44991</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>23</td>
      <td>4</td>
      <td>82</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521763200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-24</th>
      <td>44992</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>24</td>
      <td>5</td>
      <td>83</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521849600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-25</th>
      <td>44993</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>25</td>
      <td>6</td>
      <td>84</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521936000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-26</th>
      <td>44994</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>26</td>
      <td>0</td>
      <td>85</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522022400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-27</th>
      <td>44995</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>27</td>
      <td>1</td>
      <td>86</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522108800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-28</th>
      <td>44996</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>28</td>
      <td>2</td>
      <td>87</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522195200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-29</th>
      <td>44997</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>29</td>
      <td>3</td>
      <td>88</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522281600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-30</th>
      <td>44998</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>30</td>
      <td>4</td>
      <td>89</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522368000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-31</th>
      <td>44999</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>31</td>
      <td>5</td>
      <td>90</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522454400</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>45000 rows × 17 columns</p>
</div>



The next function process the dataframe, to convert the data to the final representation to be used in the Neural network, and separates the response variable from the predictors.


```python
df, y, nas, mapper = proc_df(joined_samp, 'sales', do_scale=True)
yl = np.log(y)
```

    /home/adrianrdzv/anaconda3/envs/fastai/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in log
      



```python
#df_test, _, nas, mapper = proc_df(test, 'sales', do_scale=True, skip_flds=['id'], mapper=mapper, na_dict=nas)

df_test, _, nas, mapper = proc_df(joined_test, 'sales', do_scale=True, skip_flds=['id'],mapper=mapper, na_dict=nas)
```

As we can see in the next chunks of code the data is now all continuos


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store</th>
      <th>item</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>-1.731103</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1.729205</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1.727308</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1.725411</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1.723514</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store</th>
      <th>item</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1.733000</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.734897</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.736794</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.738691</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.740588</td>
    </tr>
  </tbody>
</table>
</div>



The validation set to be used, will be accord to the expected test set, so we will use the last three months as a validation set


```python
val_idx = np.flatnonzero(
    (df.index<=datetime.datetime(2017,12,31)) & (df.index>=datetime.datetime(2017,10,1)))
```


```python
len(df.iloc[val_idx])

```




    46000




```python
val_idx
```




    array([  1734,   1735,   1736,   1737,   1738,   1739,   1740,   1741,   1742,   1743,   1744,   1745,
             1746,   1747,   1748,   1749,   1750,   1751,   1752,   1753, ..., 912980, 912981, 912982, 912983,
           912984, 912985, 912986, 912987, 912988, 912989, 912990, 912991, 912992, 912993, 912994, 912995,
           912996, 912997, 912998, 912999])




```python
#prueba de datos de validacion en rango correcto
df.iloc[912984]
```




    store                10.000000
    item                 50.000000
    Year                  5.000000
    Month                12.000000
    Week                 50.000000
    Day                  16.000000
    Dayofweek             6.000000
    Dayofyear           350.000000
    Is_month_end          1.000000
    Is_month_start        1.000000
    Is_quarter_end        1.000000
    Is_quarter_start      1.000000
    Is_year_end           1.000000
    Is_year_start         1.000000
    Elapsed               1.702646
    Name: 2017-12-16 00:00:00, dtype: float64



## DL

We are going to create a new metric **SMAPE** 
https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error


```python
#Eliminar la metrica de rossman (NA's)
def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)

max_y = np.max(y)
y_range=(0,max_y*1.2)
##F_t = y_pred A_t = targ
def SMAPE(y_pred,targ):
    return (np.abs(y_pred-targ)/((np.fabs(y_pred)+np.fabs(targ))/2)).mean()                     
#return (math.fabs(y_pred-targ)/((math.fabs(y_pred)+math.fabs(targ))/2))
```

We can create a ModelData object directly from out data frame.


```python
#md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128,
#                                       test_df=df_test)
#md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128)
#md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=128)
#md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=128,test_df=df_test)
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=256,test_df=df_test)
```


```python
cat_vars
```




    ['store',
     'item',
     'Year',
     'Month',
     'Week',
     'Day',
     'Dayofweek',
     'Dayofyear',
     'Is_month_end',
     'Is_month_start',
     'Is_quarter_end',
     'Is_quarter_start',
     'Is_year_end',
     'Is_year_start']



Some categorical variables have a lot more levels than others. Store, in particular, has over a thousand!


```python
cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]
```


```python
cat_sz
```




    [('store', 11),
     ('item', 51),
     ('Year', 6),
     ('Month', 13),
     ('Week', 54),
     ('Day', 32),
     ('Dayofweek', 8),
     ('Dayofyear', 367),
     ('Is_month_end', 3),
     ('Is_month_start', 3),
     ('Is_quarter_end', 3),
     ('Is_quarter_start', 3),
     ('Is_year_end', 3),
     ('Is_year_start', 3)]



We use the *cardinality* of each variable (that is, its number of unique values) to decide how large to make its *embeddings*. Each level will be associated with a vector with length defined as below.


```python
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
```


```python
emb_szs
```




    [(11, 6),
     (51, 26),
     (6, 3),
     (13, 7),
     (54, 27),
     (32, 16),
     (8, 4),
     (367, 50),
     (3, 2),
     (3, 2),
     (3, 2),
     (3, 2),
     (3, 2),
     (3, 2)]




```python
y
```




    array([13, 11, 14, 13, 10, 12, 10,  9, 12,  9,  9,  7, 10, 12,  5,  7, 16,  7, 18, 15, ..., 67, 67, 72, 72,
           52, 86, 53, 54, 51, 63, 75, 70, 76, 51, 41, 63, 59, 74, 62, 82])




```python
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [2000,800], [0.001,0.01], y_range=y_range)
lr = 1e-3
```

    /home/adrianrdzv/Documentos/fastai/fastai/courses/dl1/fastai/column_data.py:101: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
      for o in self.lins: kaiming_normal(o.weight.data)
    /home/adrianrdzv/Documentos/fastai/fastai/courses/dl1/fastai/column_data.py:103: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
      kaiming_normal(self.outp.weight.data)



```python
m.lr_find()
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


     61%|██████    | 2053/3387 [02:50<01:50, 12.08it/s, loss=435]   


```python
m.sched.plot()
```


![png](/images/output_60_0.png)


With the fit function and passing the metric SMAPE we could see how our model is working in the validation set, and seeing this we could then generate our prediction for the test range


```python
m.fit(lr, 3, metrics=[SMAPE])
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=3), HTML(value='')))


      1%|          | 31/3387 [00:02<04:33, 12.28it/s, loss=3.51e+03]
    epoch      trn_loss   val_loss   SMAPE                           
        0      58.461612  62.077307  0.12835   
        1      57.60579   61.226561  0.128534                     
        2      55.84363   60.653265  0.127744                     
    





    [60.65326502791695, 0.12774361452849015]




```python
#Calar mañana con este en kaggle
#m.fit(lr, 2, metrics=[exp_rmspe,SMAPE], cycle_len=3)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=6), HTML(value='')))


    epoch      trn_loss   val_loss   exp_rmspe  SMAPE                
        0      58.428253  60.687587  nan        0.125747  
        1      55.664219  58.901838  nan        0.123808          
        2      53.808453  56.613611  nan        0.122914          
        3      56.226725  56.70169   nan        0.122738          
        4      53.166681  56.525577  nan        0.122837          
        5      53.49262   56.550598  nan        0.122943          
    





    [56.550598181683085, nan, 0.1229434128092683]




```python
#m.load_cycle()
#m.load
```

We could calculate the metric again to validate our model before being use to predict in the test dataset


```python
x,y=m.predict_with_targs()
SMAPE(x,y)
```




    0.12774362



Using the predict function with True parameter we predict in the test dataset, so we can save our results


```python
pred_test=m.predict(True)
```


```python
len(pred_test)
pred_test
```




    array([[12.4353 ],
           [15.99211],
           [15.5996 ],
           [16.17989],
           [17.26732],
           [18.59948],
           [19.75835],
           [13.07455],
           [15.55011],
           [15.71422],
           [16.48752],
           [17.36207],
           [18.38673],
           [19.4954 ],
           [13.12858],
           [16.1863 ],
           [15.82948],
           [16.28306],
           [17.44085],
           [18.91442],
           ...,
           [64.77862],
           [75.88193],
           [76.84612],
           [80.93729],
           [85.14046],
           [90.38946],
           [96.48158],
           [65.06002],
           [76.08002],
           [76.95865],
           [80.9269 ],
           [84.26495],
           [89.71136],
           [97.39009],
           [64.57227],
           [76.35717],
           [75.9097 ],
           [82.60927],
           [86.11958],
           [91.05175]], dtype=float32)




```python
joined_test['sales'] = pred_test
```


```python
joined_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>store</th>
      <th>item</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
      <th>sales</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>1514764800</td>
      <td>12.435302</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1514851200</td>
      <td>15.992112</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1514937600</td>
      <td>15.599597</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515024000</td>
      <td>16.179892</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515110400</td>
      <td>17.267321</td>
    </tr>
    <tr>
      <th>2018-01-06</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515196800</td>
      <td>18.599476</td>
    </tr>
    <tr>
      <th>2018-01-07</th>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515283200</td>
      <td>19.758350</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515369600</td>
      <td>13.074553</td>
    </tr>
    <tr>
      <th>2018-01-09</th>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515456000</td>
      <td>15.550114</td>
    </tr>
    <tr>
      <th>2018-01-10</th>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>10</td>
      <td>2</td>
      <td>10</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515542400</td>
      <td>15.714221</td>
    </tr>
    <tr>
      <th>2018-01-11</th>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>3</td>
      <td>11</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515628800</td>
      <td>16.487524</td>
    </tr>
    <tr>
      <th>2018-01-12</th>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>12</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515715200</td>
      <td>17.362072</td>
    </tr>
    <tr>
      <th>2018-01-13</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>5</td>
      <td>13</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515801600</td>
      <td>18.386734</td>
    </tr>
    <tr>
      <th>2018-01-14</th>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>2</td>
      <td>14</td>
      <td>6</td>
      <td>14</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515888000</td>
      <td>19.495398</td>
    </tr>
    <tr>
      <th>2018-01-15</th>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>15</td>
      <td>0</td>
      <td>15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1515974400</td>
      <td>13.128580</td>
    </tr>
    <tr>
      <th>2018-01-16</th>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>1</td>
      <td>16</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516060800</td>
      <td>16.186298</td>
    </tr>
    <tr>
      <th>2018-01-17</th>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>17</td>
      <td>2</td>
      <td>17</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516147200</td>
      <td>15.829476</td>
    </tr>
    <tr>
      <th>2018-01-18</th>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>18</td>
      <td>3</td>
      <td>18</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516233600</td>
      <td>16.283060</td>
    </tr>
    <tr>
      <th>2018-01-19</th>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>19</td>
      <td>4</td>
      <td>19</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516320000</td>
      <td>17.440847</td>
    </tr>
    <tr>
      <th>2018-01-20</th>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>20</td>
      <td>5</td>
      <td>20</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516406400</td>
      <td>18.914425</td>
    </tr>
    <tr>
      <th>2018-01-21</th>
      <td>20</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>3</td>
      <td>21</td>
      <td>6</td>
      <td>21</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516492800</td>
      <td>19.885489</td>
    </tr>
    <tr>
      <th>2018-01-22</th>
      <td>21</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>22</td>
      <td>0</td>
      <td>22</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516579200</td>
      <td>13.303565</td>
    </tr>
    <tr>
      <th>2018-01-23</th>
      <td>22</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>23</td>
      <td>1</td>
      <td>23</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516665600</td>
      <td>16.030056</td>
    </tr>
    <tr>
      <th>2018-01-24</th>
      <td>23</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>24</td>
      <td>2</td>
      <td>24</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516752000</td>
      <td>16.096254</td>
    </tr>
    <tr>
      <th>2018-01-25</th>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>25</td>
      <td>3</td>
      <td>25</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516838400</td>
      <td>16.395996</td>
    </tr>
    <tr>
      <th>2018-01-26</th>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>26</td>
      <td>4</td>
      <td>26</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1516924800</td>
      <td>17.516056</td>
    </tr>
    <tr>
      <th>2018-01-27</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>27</td>
      <td>5</td>
      <td>27</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1517011200</td>
      <td>18.376472</td>
    </tr>
    <tr>
      <th>2018-01-28</th>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>4</td>
      <td>28</td>
      <td>6</td>
      <td>28</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1517097600</td>
      <td>19.416647</td>
    </tr>
    <tr>
      <th>2018-01-29</th>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>5</td>
      <td>29</td>
      <td>0</td>
      <td>29</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1517184000</td>
      <td>13.390797</td>
    </tr>
    <tr>
      <th>2018-01-30</th>
      <td>29</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>5</td>
      <td>30</td>
      <td>1</td>
      <td>30</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1517270400</td>
      <td>15.469314</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-03-02</th>
      <td>44970</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
      <td>61</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1519948800</td>
      <td>84.911552</td>
    </tr>
    <tr>
      <th>2018-03-03</th>
      <td>44971</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>9</td>
      <td>3</td>
      <td>5</td>
      <td>62</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520035200</td>
      <td>90.113197</td>
    </tr>
    <tr>
      <th>2018-03-04</th>
      <td>44972</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>9</td>
      <td>4</td>
      <td>6</td>
      <td>63</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520121600</td>
      <td>94.746307</td>
    </tr>
    <tr>
      <th>2018-03-05</th>
      <td>44973</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>64</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520208000</td>
      <td>65.446701</td>
    </tr>
    <tr>
      <th>2018-03-06</th>
      <td>44974</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>6</td>
      <td>1</td>
      <td>65</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520294400</td>
      <td>76.037209</td>
    </tr>
    <tr>
      <th>2018-03-07</th>
      <td>44975</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>2</td>
      <td>66</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520380800</td>
      <td>75.994850</td>
    </tr>
    <tr>
      <th>2018-03-08</th>
      <td>44976</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>8</td>
      <td>3</td>
      <td>67</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520467200</td>
      <td>80.185951</td>
    </tr>
    <tr>
      <th>2018-03-09</th>
      <td>44977</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>9</td>
      <td>4</td>
      <td>68</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520553600</td>
      <td>84.384804</td>
    </tr>
    <tr>
      <th>2018-03-10</th>
      <td>44978</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>69</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520640000</td>
      <td>91.049042</td>
    </tr>
    <tr>
      <th>2018-03-11</th>
      <td>44979</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>10</td>
      <td>11</td>
      <td>6</td>
      <td>70</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520726400</td>
      <td>96.319534</td>
    </tr>
    <tr>
      <th>2018-03-12</th>
      <td>44980</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>12</td>
      <td>0</td>
      <td>71</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520812800</td>
      <td>64.778618</td>
    </tr>
    <tr>
      <th>2018-03-13</th>
      <td>44981</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>13</td>
      <td>1</td>
      <td>72</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520899200</td>
      <td>75.881927</td>
    </tr>
    <tr>
      <th>2018-03-14</th>
      <td>44982</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>14</td>
      <td>2</td>
      <td>73</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1520985600</td>
      <td>76.846123</td>
    </tr>
    <tr>
      <th>2018-03-15</th>
      <td>44983</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>15</td>
      <td>3</td>
      <td>74</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521072000</td>
      <td>80.937286</td>
    </tr>
    <tr>
      <th>2018-03-16</th>
      <td>44984</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>16</td>
      <td>4</td>
      <td>75</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521158400</td>
      <td>85.140465</td>
    </tr>
    <tr>
      <th>2018-03-17</th>
      <td>44985</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>17</td>
      <td>5</td>
      <td>76</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521244800</td>
      <td>90.389458</td>
    </tr>
    <tr>
      <th>2018-03-18</th>
      <td>44986</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>11</td>
      <td>18</td>
      <td>6</td>
      <td>77</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521331200</td>
      <td>96.481575</td>
    </tr>
    <tr>
      <th>2018-03-19</th>
      <td>44987</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>19</td>
      <td>0</td>
      <td>78</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521417600</td>
      <td>65.060020</td>
    </tr>
    <tr>
      <th>2018-03-20</th>
      <td>44988</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>20</td>
      <td>1</td>
      <td>79</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521504000</td>
      <td>76.080025</td>
    </tr>
    <tr>
      <th>2018-03-21</th>
      <td>44989</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>21</td>
      <td>2</td>
      <td>80</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521590400</td>
      <td>76.958649</td>
    </tr>
    <tr>
      <th>2018-03-22</th>
      <td>44990</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>22</td>
      <td>3</td>
      <td>81</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521676800</td>
      <td>80.926903</td>
    </tr>
    <tr>
      <th>2018-03-23</th>
      <td>44991</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>23</td>
      <td>4</td>
      <td>82</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521763200</td>
      <td>84.264954</td>
    </tr>
    <tr>
      <th>2018-03-24</th>
      <td>44992</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>24</td>
      <td>5</td>
      <td>83</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521849600</td>
      <td>89.711365</td>
    </tr>
    <tr>
      <th>2018-03-25</th>
      <td>44993</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>12</td>
      <td>25</td>
      <td>6</td>
      <td>84</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1521936000</td>
      <td>97.390091</td>
    </tr>
    <tr>
      <th>2018-03-26</th>
      <td>44994</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>26</td>
      <td>0</td>
      <td>85</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522022400</td>
      <td>64.572266</td>
    </tr>
    <tr>
      <th>2018-03-27</th>
      <td>44995</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>27</td>
      <td>1</td>
      <td>86</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522108800</td>
      <td>76.357170</td>
    </tr>
    <tr>
      <th>2018-03-28</th>
      <td>44996</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>28</td>
      <td>2</td>
      <td>87</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522195200</td>
      <td>75.909698</td>
    </tr>
    <tr>
      <th>2018-03-29</th>
      <td>44997</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>29</td>
      <td>3</td>
      <td>88</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522281600</td>
      <td>82.609268</td>
    </tr>
    <tr>
      <th>2018-03-30</th>
      <td>44998</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>30</td>
      <td>4</td>
      <td>89</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522368000</td>
      <td>86.119576</td>
    </tr>
    <tr>
      <th>2018-03-31</th>
      <td>44999</td>
      <td>10</td>
      <td>50</td>
      <td>2018</td>
      <td>3</td>
      <td>13</td>
      <td>31</td>
      <td>5</td>
      <td>90</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1522454400</td>
      <td>91.051750</td>
    </tr>
  </tbody>
</table>
<p>45000 rows × 17 columns</p>
</div>



We can save our results in the format specified, with the next chunk of code


```python
csv_fn=f'{PATH}tmp/submission_4agosto.csv'
joined_test[['id','sales']].to_csv(csv_fn, index=False)
```
