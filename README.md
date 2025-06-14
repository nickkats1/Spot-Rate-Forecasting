### Spot Rate Forecasting
This is a forecast between the US Dollar and UK Pound. There is no arbitrage in the Foreign Exchange Market. Since, I am American, I will quote the currency in American Terms. Example: 
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

![Spot_Rate_USDGBP](images/spotrate.png)

### Example of S(pound/$)
![Spot_Rate_GBP](images/sportrate_GBPUSD.png)

Notice, the two graphs are completely inverse.

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


model = LSTM(input_size=1,hidden_size=228,num_layers=1,output_size=1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_fn = nn.MSELoss()
epochs = 200


for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred.float(),y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rsme = np.sqrt(loss_fn(y_pred,y_train))
        y_pred_test =  model(X_test)
        test_rsme = np.sqrt(loss_fn(y_pred_test,y_test))
        print(f'Epoch: {epoch}; train_RSEM: {train_rsme:.4}; Test RSME: {test_rsme:.4}')
```

After the train/test split, scaling, sliding window etc., the LSTM class is defined and the training and testing data is evaluated

### The Results

![predicted_vs_actual_spot_rate](images/predicted_vs_actual_SpotRate.png)

```text

            Date  Actual Spot Rate  Predicted Spot Rate
12292 2020-01-02            1.3128               1.3135
12293 2020-01-03            1.3091               1.3140
12294 2020-01-06            1.3163               1.3121
12295 2020-01-07            1.3127               1.3155
12296 2020-01-08            1.3110               1.3161
12297 2020-01-09            1.3069               1.3138
12298 2020-01-10            1.3060               1.3126
12299 2020-01-13            1.2983               1.3105
12300 2020-01-14            1.3018               1.3070
12301 2020-01-15            1.3030               1.3056
12302 2020-01-16            1.3076               1.3055
12303 2020-01-17            1.3029               1.3060
12304 2020-01-21            1.3047               1.3052
12305 2020-01-22            1.3136               1.3050
12306 2020-01-23            1.3104               1.3084
12307 2020-01-24            1.3071               1.3099
12308 2020-01-27            1.3054               1.3087
12309 2020-01-28            1.2996               1.3086
12310 2020-01-29            1.3012               1.3065
12311 2020-01-30            1.3106               1.3050
            Date  Actual Spot Rate  Predicted Spot Rate
13629 2025-05-08            1.3287               1.3343
13630 2025-05-09            1.3318               1.3320
13631 2025-05-12            1.3194               1.3322
13632 2025-05-13            1.3280               1.3293
13633 2025-05-14            1.3303               1.3284
13634 2025-05-15            1.3292               1.3306
13635 2025-05-16            1.3257               1.3297
13636 2025-05-19            1.3357               1.3281
13637 2025-05-20            1.3371               1.3312
13638 2025-05-21            1.3447               1.3337
13639 2025-05-22            1.3426               1.3372
13640 2025-05-23            1.3511               1.3396
13641 2025-05-27            1.3505               1.3434
13642 2025-05-28            1.3462               1.3467
13643 2025-05-29            1.3489               1.3469
13644 2025-05-30            1.3468               1.3480
13645 2025-06-02            1.3549               1.3489
13646 2025-06-03            1.3520               1.3507
13647 2025-06-04            1.3570               1.3520
13648 2025-06-05            1.3584               1.3533
```












