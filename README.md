# Estimating-Beta-Using-High-Frequency-Factors
## A demonstration of calculation factors and betas for year 2018

### 1. Portfolio_RCC.py
Set the startyear as 2018 and endyear as 2019, where startdate is set to be smaller than t-2 and enddate being slightly greater than t+1
  `startyear = 2018`  `endyear = 2019`
```
  startdate = 20160101
  enddate = 20201231
```


### 2. NewAPI_Save_TAQ.py
Set the startyear as 2018 and endyear as 2019. 
`startyear = 2018`  `endyear = 2019`

Note if this is for the first time of calculation make sure to uncomment both for loops to get daily sliced matchingtable and TAQ data saved as intermediate output.

### 3. NewAPI_RCC_concat.py
Set the startyear as 2018 and endyear as 2019. Also be aware of potential overwriting on my filepath. Please adjust the path info carefully.
```
  DATA_DIR = '/project2/dachxiu/xinyu/thesis/DATA'
  RESULT_DIR = '/project2/dachxiu/xinyu/thesis/RESULT'
  TEMP_DIR = '/project2/dachxiu/xinyu/thesis/TEMP'

  startyear=2018
  endyear=2019
```
### 4. pre2step.py
Run this code to arrange output from NewAPI_RCC_concat.py in a monthly manner.
  
### 5. 003_submit.py
Set calculation range for beta in the following manner, can do part of it or from 201807-201906. Again **watch out the output path infomation in tralpha and tralpha_hf function in main.py ` out_df.to_csv('./Results4Xinyu/lf/cbeta/%d.csv' % yrmth)`**. Don't run before adjust the path.
```
for year in range(2018, 2019):
    if year == 2018:
        for month in range(7, 13):
            ym = '%d%02d' % (year, month)
            ...
     if year == 2019:
        for month in range(1, 7):
            ym = '%d%02d' % (year, month)
            ...
```

### More to see for the code explanation:
https://github.com/Wentworthliu123/Fama_French_Selfstudy/blob/master/five_factor_model/HF_Package/RCC_version/Guide_Note_for_High_Frequency_FF_Factor_Construction_in_Python%20.pdf
