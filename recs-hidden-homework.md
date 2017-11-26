

```python
import pandas as pd
import numpy as np
```


```python
from os import path
dir_cos = '/home/asigachev/recs-hidden/recs-cos'
dir_als = '/home/asigachev/recs-hidden/recs-als'
```


```python
col_names = ['user', 'artist-mbid', 'artist-name', 'total-plays']
data = pd.read_csv(
    'lastfm_small.tsv',
    sep='\t',
    header=None,
    names=col_names)
```


```python
data.headad()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>artist-mbid</th>
      <th>artist-name</th>
      <th>total-plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>3bd73256-3905-4f3a-97e2-8b341527f805</td>
      <td>betty blowtorch</td>
      <td>2137</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>f2fb0ff0-5679-42ec-a55c-15109ce6e320</td>
      <td>die Ärzte</td>
      <td>1099</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>b3ae82c2-e60b-4551-a76d-6620f1b456aa</td>
      <td>melissa etheridge</td>
      <td>897</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>3d6bbeb7-f90e-4d10-b440-e153c0d10b53</td>
      <td>elvenking</td>
      <td>717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00000c289a1829a808ac09c00daf10bc3c4e223b</td>
      <td>bbd2ffd7-17f4-4506-8572-c1ea58c3f9a8</td>
      <td>juliette &amp; the licks</td>
      <td>706</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.fillna("None", inplace=True)
data["user_id"] = data["user"].astype("category").cat.codes.copy() + 1
data["artist_id"] = data["artist-mbid"].astype("category").cat.codes.copy() + 1
data.drop(["artist-name", "artist-mbid", "user"], axis=1, inplace=True)
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total-plays</th>
      <th>user_id</th>
      <th>artist_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2137</td>
      <td>1</td>
      <td>15531</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1099</td>
      <td>1</td>
      <td>63469</td>
    </tr>
    <tr>
      <th>2</th>
      <td>897</td>
      <td>1</td>
      <td>46858</td>
    </tr>
    <tr>
      <th>3</th>
      <td>717</td>
      <td>1</td>
      <td>15968</td>
    </tr>
    <tr>
      <th>4</th>
      <td>706</td>
      <td>1</td>
      <td>48969</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_indices = np.random.choice(
    data.index.values,
    replace=False,
    size=int(len(data.index.values) * 0.2)
)
test_data = data.iloc[test_indices]
train_data = data.drop(test_indices)
```


```python
test_file_name = "lastfm.test.0"
test_data[["user_id", "artist_id", "total-plays"]].to_csv(
    test_file_name,
    sep="\t",
    header=False,
    index=False
)
train_file_name = "train/lastfm.train.0"
train_data[["user_id", "artist_id", "total-plays"]].to_csv(
    train_file_name,
    sep="\t",
    header=False,
    index=False
)
```


```python
from scipy.sparse import coo_matrix
import numpy as np

plays = coo_matrix((
    train_data["total-plays"].astype(np.float32),
    (
        train_data["artist_id"],
        train_data["user_id"]
    )
))
```


```python
import time
from implicit.nearest_neighbours import CosineRecommender

model_cos = CosineRecommender()
print("строим матрицу схожести по косинусной мере")
start = time.time()
model_cos.fit(plays)
print("построили матрицу схожести по косинусной мере за {} секунд".format(
        time.time() - start))
```

    строим матрицу схожести по косинусной мере
    построили матрицу схожести по косинусной мере за 1.2766621112823486 секунд



```python
print("получаем рекомендации для всех пользователей")
start = time.time()
user_plays = plays.T.tocsr()
with open(path.join(dir_cos, test_file_name + ".recs.tsv"), "w") as output_file:
    for user_id in test_data["user_id"].unique():
        for artist_id, score in model.recommend(user_id, user_plays):
                output_file.write("%s\t%s\t%s\n" % (user_id, artist_id, score))
print("получили рекомендации для всех пользователей за {} секнуд".format(
        time.time() - start))
```

    получаем рекомендации для всех пользователей
    получили рекомендации для всех пользователей за 21.19092583656311 секнуд


## ALS


```python
plays2 = plays.astype(np.float64)
```


```python
from implicit.als import AlternatingLeastSquares

model_als = AlternatingLeastSquares(factors=50)
print("строим матрицу схожести по ALS")
start = time.time()
model_als.fit(plays2)
print("построили матрицу схожести по ALS мере за {} секунд".format(
        time.time() - start))
```

    строим матрицу схожести по ALS
    построили матрицу схожести по ALS мере за 12.927585363388062 секунд



```python
print("получаем рекомендации для всех пользователей")
start = time.time()
user_plays = plays.T.tocsr()
with open(path.join(dir_als, test_file_name + ".recs.tsv"), "w") as output_file:
    for user_id in test_data["user_id"].unique():
        for artist_id, score in model_als.recommend(user_id, user_plays):
                output_file.write("%s\t%s\t%s\n" % (user_id, artist_id, score))
print("получили рекомендации для всех пользователей за {} секнуд".format(
        time.time() - start))
```

    получаем рекомендации для всех пользователей
    получили рекомендации для всех пользователей за 99.7573082447052 секнуд


## Оценка результатов

### kNN


```python
!/home/asigachev/mrec/sbin/mrec_evaluate \
    --input_format=tsv --test_input_format=tsv \
    --train /home/asigachev/recs-hidden/lastfm.test.0 \
    --recsdir /home/asigachev/recs-hidden/recs-cos
```

    [2017-11-26 08:13:33,075] INFO: processing /home/asigachev/recs-hidden/lastfm.test.0...
    None
    mrr            0.0518 +/- 0.0000
    prec@5         0.0170 +/- 0.0000
    prec@10        0.0152 +/- 0.0000
    prec@15        0.0101 +/- 0.0000
    prec@20        0.0076 +/- 0.0000


### ALS


```python
!/home/asigachev/mrec/sbin/mrec_evaluate \
    --input_format=tsv --test_input_format=tsv \
    --train /home/asigachev/recs-hidden/lastfm.test.0 \
    --recsdir /home/asigachev/recs-hidden/recs-als
```

    [2017-11-26 08:17:14,838] INFO: processing /home/asigachev/recs-hidden/lastfm.test.0...
    None
    mrr            0.3009 +/- 0.0000
    prec@5         0.1302 +/- 0.0000
    prec@10        0.1077 +/- 0.0000
    prec@15        0.0718 +/- 0.0000
    prec@20        0.0539 +/- 0.0000

