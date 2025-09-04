### Spot Rate Forecasting
These forecast consists of the following currency pairs: S(  $/pound  ), S(  $/Euro  ), S(  $/peso  ), S (  $/AUD  ), S(  $/YEN  ) I will quote the currency in American Terms. Example: 
```text
S($/pound) = 1.33 -> 1 pound = 1.33 $
S(pound/$) = 1 / S($/pound) = 1 / 1.33 $ -> .75 pound = 1 $
S($/pound) = 1 / S(pound/$) = 1 / .75 -> 1 pound = 1.33 $
```

The 1:1 parity always holds. There are examples of arbitrage being available for investors to make profit. For example, the Uncovered Interest Rate Parity, however this does not hold long and has to do with taking advantage of a domestic countries interest rates.



### Requirements
```bash
pip install matplotlib seaborn pandas scikit-learn torch torchvision torchaudio numpy fred
```



### LSTM
```python
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(LSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size,output_size)
        
    def forward(self,X):
        h0 = torch.zeros(1,X.size(0),self.hidden_size)
        c0 = torch.zeros(1,X.size(0),self.hidden_size)
        out,_ = self.lstm(X,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out

```
### GRU
```python
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        h0 = torch.zeros(1, X.size(0), self.hidden_size)
        out, _ = self.gru(X, h0)
        out = self.fc(out[:,-1,:])
        return out
```
### Bi-LSTM
```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, X):
        h0 = torch.zeros(2 * self.num_layers, X.size(0), self.hidden_size)
        c0 = torch.zeros(2 * self.num_layers, X.size(0), self.hidden_size)
        out, _ = self.lstm(X, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```



### S($/GBP) Results

![predicted-vs-actual-exchange-rate-usd-gbp-bi-lstm](images/predicted_vs_actual_SpotRate_bidirectional_lstm.png)

```text
Mean Absolute Percentage Error: 0.0580
R2 Score: 96.216959%
Root Mean Squared Error: 0.0079
            Date  Actual Spot Rate  Predicted Spot Rate
12338 2020-03-10            1.2933               1.3186
12339 2020-03-11            1.2887               1.3027
12340 2020-03-12            1.2541               1.2983
12341 2020-03-13            1.2406               1.2656
12342 2020-03-16            1.2278               1.2528
12343 2020-03-17            1.2017               1.2408
12344 2020-03-18            1.1760               1.2162
12345 2020-03-19            1.1662               1.1921
12346 2020-03-20            1.1743               1.1829
12347 2020-03-23            1.1492               1.1905
12348 2020-03-24            1.1784               1.1670
12349 2020-03-25            1.1763               1.1943
12350 2020-03-26            1.2140               1.1924
12351 2020-03-27            1.2360               1.2278
12352 2020-03-30            1.2392               1.2485
12353 2020-03-31            1.2454               1.2515
12354 2020-04-01            1.2394               1.2574
12355 2020-04-02            1.2380               1.2517
12356 2020-04-03            1.2228               1.2504
12357 2020-04-06            1.2298               1.2361
            Date  Actual Spot Rate  Predicted Spot Rate
13688 2025-08-04            1.3290               1.3337
13689 2025-08-05            1.3309               1.3366
13690 2025-08-06            1.3360               1.3384
13691 2025-08-07            1.3434               1.3433
13692 2025-08-08            1.3444               1.3503
13693 2025-08-11            1.3414               1.3513
13694 2025-08-12            1.3509               1.3484
13695 2025-08-13            1.3572               1.3575
13696 2025-08-14            1.3532               1.3635
13697 2025-08-15            1.3565               1.3597
13698 2025-08-18            1.3521               1.3628
13699 2025-08-19            1.3496               1.3586
13700 2025-08-20            1.3450               1.3562
13701 2025-08-21            1.3417               1.3519
13702 2025-08-22            1.3528               1.3487
13703 2025-08-25            1.3491               1.3593
13704 2025-08-26            1.3485               1.3558
13705 2025-08-27            1.3468               1.3552
13706 2025-08-28            1.3516               1.3536
13707 2025-08-29            1.3511               1.3582
```



### GRU Results S(USD/PESO)

![gru-forecast-usd-peso](images/predicted_vs_actual_SpotRate_gru-usd-peso.png)

```
text
Date  Actual Spot Rate  Predicted Spot Rate
6381 2019-04-16            0.0527               0.0537
6382 2019-04-17            0.0531               0.0535
6383 2019-04-18            0.0533               0.0538
6384 2019-04-19            0.0533               0.0540
6385 2019-04-22            0.0531               0.0540
6386 2019-04-23            0.0528               0.0538
6387 2019-04-24            0.0526               0.0535
6388 2019-04-25            0.0524               0.0533
6389 2019-04-26            0.0528               0.0531
6390 2019-04-29            0.0526               0.0535
6391 2019-04-30            0.0526               0.0533
6392 2019-05-01            0.0530               0.0534
6393 2019-05-02            0.0523               0.0538
6394 2019-05-03            0.0527               0.0530
6395 2019-05-06            0.0527               0.0534
6396 2019-05-07            0.0525               0.0534
6397 2019-05-08            0.0524               0.0532
6398 2019-05-09            0.0519               0.0532
6399 2019-05-10            0.0523               0.0526
6400 2019-05-13            0.0521               0.0530
           Date  Actual Spot Rate  Predicted Spot Rate
7956 2025-08-04            0.0530               0.0537
7957 2025-08-05            0.0533               0.0538
7958 2025-08-06            0.0536               0.0540
7959 2025-08-07            0.0535               0.0543
7960 2025-08-08            0.0539               0.0542
7961 2025-08-11            0.0536               0.0546
7962 2025-08-12            0.0539               0.0543
7963 2025-08-13            0.0536               0.0546
7964 2025-08-14            0.0531               0.0543
7965 2025-08-15            0.0534               0.0539
7966 2025-08-18            0.0532               0.0541
7967 2025-08-19            0.0532               0.0539
7968 2025-08-20            0.0532               0.0539
7969 2025-08-21            0.0532               0.0539
7970 2025-08-22            0.0538               0.0539
7971 2025-08-25            0.0537               0.0545
7972 2025-08-26            0.0536               0.0544
7973 2025-08-27            0.0535               0.0543
7974 2025-08-28            0.0536               0.0542
7975 2025-08-29            0.0537               0.0543
```
### Bidirectional LSTM S($/YEN)
![bidrectional_lstm_image](images/predicted_vs_actual_SpotRate_usdjpn.png)
```text
R2-Score: 97.90%
Mean Absolute Percentage Error: 0.02807
Root Mean Squared Error: 0.006378
            Date  Actual Spot Rate  Predicted Spot Rate
12331 2020-03-10            0.6470             0.669608
12332 2020-03-11            0.6512             0.652951
12333 2020-03-12            0.6280             0.656742
12334 2020-03-13            0.6161             0.635883
12335 2020-03-16            0.6132             0.625263
12336 2020-03-17            0.5976             0.622683
12337 2020-03-18            0.5820             0.608861
12338 2020-03-19            0.5859             0.595132
12339 2020-03-20            0.5841             0.598555
12340 2020-03-23            0.5755             0.596974
12341 2020-03-24            0.5927             0.589439
12342 2020-03-25            0.5957             0.604538
12343 2020-03-26            0.6054             0.607184
12344 2020-03-27            0.6126             0.615760
12345 2020-03-30            0.6145             0.622150
12346 2020-03-31            0.6139             0.623839
12347 2020-04-01            0.6107             0.623306
12348 2020-04-02            0.6047             0.620462
12349 2020-04-03            0.6001             0.615140
12350 2020-04-06            0.6096             0.611070
            Date  Actual Spot Rate  Predicted Spot Rate
13681 2025-08-04            0.6467             0.652770
13682 2025-08-05            0.6472             0.652680
13683 2025-08-06            0.6504             0.653131
13684 2025-08-07            0.6502             0.656019
13685 2025-08-08            0.6528             0.655838
13686 2025-08-11            0.6512             0.658188
13687 2025-08-12            0.6532             0.656742
13688 2025-08-13            0.6546             0.658549
13689 2025-08-14            0.6490             0.659816
13690 2025-08-15            0.6515             0.654755
13691 2025-08-18            0.6494             0.657013
13692 2025-08-19            0.6459             0.655116
13693 2025-08-20            0.6428             0.651959
13694 2025-08-21            0.6421             0.649166
13695 2025-08-22            0.6485             0.648536
13696 2025-08-25            0.6500             0.654304
13697 2025-08-26            0.6500             0.655658
13698 2025-08-27            0.6496             0.655658
13699 2025-08-28            0.6532             0.655297
13700 2025-08-29            0.6545             0.658549
```

### BI-LSTM S($/AUD)
![bi-lstm-spot-rate-usd-aud](images/predicted_vs_actual-usd-aud.png)
```text
R2-Score: 98.39%
Mean Absolute Percentage Error: 0.02362
Root Mean Squared Error: 0.005585
            Date  Actual Spot Rate  Predicted Spot Rate
12331 2020-03-10            0.6470             0.668288
12332 2020-03-11            0.6512             0.651179
12333 2020-03-12            0.6280             0.655075
12334 2020-03-13            0.6161             0.633618
12335 2020-03-16            0.6132             0.622675
12336 2020-03-17            0.5976             0.620015
12337 2020-03-18            0.5820             0.605749
12338 2020-03-19            0.5859             0.591558
12339 2020-03-20            0.5841             0.595099
12340 2020-03-23            0.5755             0.593464
12341 2020-03-24            0.5927             0.585668
12342 2020-03-25            0.5957             0.601283
12343 2020-03-26            0.6054             0.604016
12344 2020-03-27            0.6126             0.612872
12345 2020-03-30            0.6145             0.619465
12346 2020-03-31            0.6139             0.621207
12347 2020-04-01            0.6107             0.620657
12348 2020-04-02            0.6047             0.617723
12349 2020-04-03            0.6001             0.612232
12350 2020-04-06            0.6096             0.608030
            Date  Actual Spot Rate  Predicted Spot Rate
13681 2025-08-04            0.6467             0.650994
13682 2025-08-05            0.6472             0.650901
13683 2025-08-06            0.6504             0.651364
13684 2025-08-07            0.6502             0.654333
13685 2025-08-08            0.6528             0.654147
13686 2025-08-11            0.6512             0.656561
13687 2025-08-12            0.6532             0.655075
13688 2025-08-13            0.6546             0.656933
13689 2025-08-14            0.6490             0.658233
13690 2025-08-15            0.6515             0.653034
13691 2025-08-18            0.6494             0.655354
13692 2025-08-19            0.6459             0.653405
13693 2025-08-20            0.6428             0.650159
13694 2025-08-21            0.6421             0.647288
13695 2025-08-22            0.6485             0.646640
13696 2025-08-25            0.6500             0.652570
13697 2025-08-26            0.6500             0.653962
13698 2025-08-27            0.6496             0.653962
13699 2025-08-28            0.6532             0.653590
13700 2025-08-29            0.6545             0.656933
```




















