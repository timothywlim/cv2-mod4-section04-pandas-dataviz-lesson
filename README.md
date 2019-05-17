
### Questions


### Objectives
YWBAT
* change functions from 'def' format to 'lambda' format
* create a list using list comprehension
* apply lambda functions within list comprehension
* create a dictionary using zip
* Create a pivot table in pandas (this will be done on learn.co)


```python
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.datasets import load_wine

import matplotlib.pyplot as plt
```

### Rewrite the following functions as lambda functions


```python
def summ(a, b):
    return a+b
```


```python
summ = lambda a, b: a + b
```


```python
summ(3, 5)
```




    8




```python
def get_excited(string):
    return string + "!"*10
```


```python
get_excited("It's Friday, y'all")
```




    "It's Friday, y'all!!!!!!!!!!"




```python
get_excited =  lambda string: string + '!'*10
```


```python
tweet = "My Twitter is pretty much complete nonsense at this point"
```


```python
tweet_list = tweet.split(" ")
tweet_list
```




    ['My',
     'Twitter',
     'is',
     'pretty',
     'much',
     'complete',
     'nonsense',
     'at',
     'this',
     'point']




```python
tweet_list_lower = []
for word in tweet_list:
    new_word = word.lower()
    tweet_list_lower.append(new_word)
tweet_list_lower
```




    ['my',
     'twitter',
     'is',
     'pretty',
     'much',
     'complete',
     'nonsense',
     'at',
     'this',
     'point']




```python
tweet_list_lower = [word.lower() if 'i' in word else word.upper() for word in tweet_list]
tweet_list_lower
```




    ['MY',
     'twitter',
     'is',
     'PRETTY',
     'MUCH',
     'COMPLETE',
     'NONSENSE',
     'AT',
     'this',
     'point']




```python
x = 2
```


```python
y = 5 if x == 2 else 10
y
```




    5




```python
wine = load_wine()
data = wine.data
feature_names = wine.feature_names
```


```python
df = pd.DataFrame(data, columns=feature_names)
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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
for index, value in enumerate(df.ash[5:30]):
    print(index, value)
```

    0 2.45
    1 2.45
    2 2.61
    3 2.17
    4 2.27
    5 2.3
    6 2.32
    7 2.41
    8 2.39
    9 2.38
    10 2.7
    11 2.72
    12 2.62
    13 2.48
    14 2.56
    15 2.28
    16 2.65
    17 2.36
    18 2.52
    19 2.61
    20 3.22
    21 2.62
    22 2.14
    23 2.8
    24 2.21



```python
x = list(range(20, 50))
y = list(range(30, 40))
m = np.zeros((len(x), len(y)))
```


```python
for i, value_x in enumerate(x):
    for j, value_y in enumerate(y):
        m[i, j] = value_x + value_y
m
```




    array([[50., 51., 52., 53., 54., 55., 56., 57., 58., 59.],
           [51., 52., 53., 54., 55., 56., 57., 58., 59., 60.],
           [52., 53., 54., 55., 56., 57., 58., 59., 60., 61.],
           [53., 54., 55., 56., 57., 58., 59., 60., 61., 62.],
           [54., 55., 56., 57., 58., 59., 60., 61., 62., 63.],
           [55., 56., 57., 58., 59., 60., 61., 62., 63., 64.],
           [56., 57., 58., 59., 60., 61., 62., 63., 64., 65.],
           [57., 58., 59., 60., 61., 62., 63., 64., 65., 66.],
           [58., 59., 60., 61., 62., 63., 64., 65., 66., 67.],
           [59., 60., 61., 62., 63., 64., 65., 66., 67., 68.],
           [60., 61., 62., 63., 64., 65., 66., 67., 68., 69.],
           [61., 62., 63., 64., 65., 66., 67., 68., 69., 70.],
           [62., 63., 64., 65., 66., 67., 68., 69., 70., 71.],
           [63., 64., 65., 66., 67., 68., 69., 70., 71., 72.],
           [64., 65., 66., 67., 68., 69., 70., 71., 72., 73.],
           [65., 66., 67., 68., 69., 70., 71., 72., 73., 74.],
           [66., 67., 68., 69., 70., 71., 72., 73., 74., 75.],
           [67., 68., 69., 70., 71., 72., 73., 74., 75., 76.],
           [68., 69., 70., 71., 72., 73., 74., 75., 76., 77.],
           [69., 70., 71., 72., 73., 74., 75., 76., 77., 78.],
           [70., 71., 72., 73., 74., 75., 76., 77., 78., 79.],
           [71., 72., 73., 74., 75., 76., 77., 78., 79., 80.],
           [72., 73., 74., 75., 76., 77., 78., 79., 80., 81.],
           [73., 74., 75., 76., 77., 78., 79., 80., 81., 82.],
           [74., 75., 76., 77., 78., 79., 80., 81., 82., 83.],
           [75., 76., 77., 78., 79., 80., 81., 82., 83., 84.],
           [76., 77., 78., 79., 80., 81., 82., 83., 84., 85.],
           [77., 78., 79., 80., 81., 82., 83., 84., 85., 86.],
           [78., 79., 80., 81., 82., 83., 84., 85., 86., 87.],
           [79., 80., 81., 82., 83., 84., 85., 86., 87., 88.]])




```python
y = np.random.randint(20, 100, 30)
w = np.random.randint(50, 1000, 30)
y
```




    array([63, 72, 82, 98, 57, 66, 63, 20, 76, 26, 72, 82, 37, 64, 98, 40, 22,
           80, 20, 29, 48, 64, 99, 37, 90, 70, 64, 72, 25, 58])




```python
z_score = (y[0] - y.mean())/y.std()
z_score2 = (w[0] - w.mean())/w.std()

z_score, z_score2
```




    (0.1337722192412354, -0.6871339897090867)


