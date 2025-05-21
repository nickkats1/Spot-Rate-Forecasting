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


model = LSTM(input_size=1,hidden_size=512,num_layers=1,output_size=1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_fn = nn.MSELoss()
epochs = 100


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
13616 2025-04-21              1.34                 1.34
13617 2025-04-22              1.34                 1.35
13618 2025-04-23              1.33                 1.35
13619 2025-04-24              1.33                 1.34
13620 2025-04-25              1.33                 1.35
13621 2025-04-28              1.34                 1.35
13622 2025-04-29              1.34                 1.35
13623 2025-04-30              1.33                 1.35
13624 2025-05-01              1.33                 1.35
13625 2025-05-02              1.33                 1.34
13626 2025-05-05              1.33                 1.34
13627 2025-05-06              1.34                 1.34
13628 2025-05-07              1.33                 1.35
13629 2025-05-08              1.33                 1.35
13630 2025-05-09              1.33                 1.34
13631 2025-05-12              1.32                 1.35
13632 2025-05-13              1.33                 1.33
13633 2025-05-14              1.33                 1.34
13634 2025-05-15              1.33                 1.34
13635 2025-05-16              1.33                 1.34
```












