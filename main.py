import pandas as pd
import numpy as np
import math
import scipy.stats
from matplotlib import pyplot as plt

stocks = pd.read_excel('stock2.0.xlsx',sheet_name='Sheet1',header=0)
stocks = stocks.loc[:,"AAPL":"r_mkt"]
stocks.astype(float)
r_AAPL = stocks.loc[:,"AAPL"]
r_WMT = stocks.loc[:,"WMT"]
r_Fi = stocks.loc[:,"F"]
r_DIS = stocks.loc[:,"DIS"]
r_JNJ = stocks.loc[:,"JNJ"]
r_XOM = stocks.loc[:,"XOM"]
r_CAT = stocks.loc[:,"CAT"]
r_YUM = stocks.loc[:,"YUM"]
r_NUE = stocks.loc[:,"NUE"]
r_AXP = stocks.loc[:,"AXP"]
r_f = stocks.loc[:,"r_f"]
r_mkt = stocks.loc[:,"r_mkt"]

market=pd.read_excel('Market.xlsx',sheet_name='Sheet1',header=0)
market = market.loc[:,"AAPL":]
market.astype(float)
market.to_excel("1.xlsx",index=False)
#book value per share
b_AAPL = market.loc[:,"AAPL"]
b_WMT = market.loc[:,"WMT"]
b_Fi = market.loc[:,"F"]
b_DIS = market.loc[:,"DIS"]
b_JNJ = market.loc[:,"JNJ"]
b_XOM = market.loc[:,"XOM"]
b_CAT = market.loc[:,"CAT"]
b_YUM = market.loc[:,"YUM"]
b_NUE = market.loc[:,"NUE"]
b_AXP = market.loc[:,"AXP"]
b = pd.concat([b_AAPL,b_WMT,b_Fi,b_DIS,b_JNJ,b_XOM,b_CAT,b_YUM,b_NUE,b_AXP],axis=1)
b_m = np.matrix(b)
#share number
s_AAPL = market.loc[:,"AAPL_S"]
s_WMT = market.loc[:,"WMT_S"]
s_Fi = market.loc[:,"F_S"]
s_DIS = market.loc[:,"DIS_S"]
s_JNJ = market.loc[:,"JNJ_S"]
s_XOM = market.loc[:,"XOM_S"]
s_CAT = market.loc[:,"CAT_S"]
s_YUM = market.loc[:,"YUM_S"]
s_NUE = market.loc[:,"NUE_S"]
s_AXP = market.loc[:,"AXP_S"]
s = pd.concat([s_AAPL,s_WMT,s_Fi,s_DIS,s_JNJ,s_XOM,s_CAT,s_YUM,s_NUE,s_AXP],axis=1)
s_m = np.matrix(s)
#mkt value per share
m_AAPL = market.loc[:,"AAPL_MP"]
m_WMT = market.loc[:,"WMT_MP"]
m_Fi = market.loc[:,"F_MP"]
m_DIS = market.loc[:,"DIS_MP"]
m_JNJ = market.loc[:,"JNJ_MP"]
m_XOM = market.loc[:,"XOM_MP"]
m_CAT = market.loc[:,"CAT_MP"]
m_YUM = market.loc[:,"YUM_MP"]
m_NUE = market.loc[:,"NUE_MP"]
m_AXP = market.loc[:,"AXP_MP"]
m = pd.concat([m_AAPL,m_WMT,m_Fi,m_DIS,m_JNJ,m_XOM,m_CAT,m_YUM,m_NUE,m_AXP],axis=1)
m_m = np.matrix(m)
#log firm size
market_tv = np.multiply(m_m,s_m)
lgmarket = np.log(market_tv)

#b/m ratio
bm = np.divide(b_m,m_m)

t1 = 60 # moving window size, 60 months
t2 = 180 #  test window size, 15 years
i1=0
i2=0
ri_rf = np.zeros(shape=(240,10))
ri_mkt = np.zeros(shape=(240,10))
for a in range(240):
    for b in range(10):
        ri_rf[a][b]=stocks.iat[a,b]-stocks.iat[a,10]
        ri_mkt[a][b]=stocks.iat[a,11]-stocks.iat[a,10]
ones = np.ones((60,1))
g1=np.zeros((180,2))
g2=np.zeros((180,4))
g3=np.zeros((180,2))
g4=np.zeros((180,2))
g5=np.zeros((180,3))
while i1<t2:
    while i2<10:
        y = ri_rf[i1:(i1+60),i2]
        y = np.reshape(y,(60,1))
        x = ri_mkt[i1:i1+60,i2]
        x = np.reshape(x, (60, 1))
        x = np.append(ones,x,axis=1)
        temp1 = np.transpose(x)
        temp2 = np.dot(temp1,x)
        if i2 == 0:
            beta=np.dot(np.dot(np.linalg.inv(temp2),temp1),y)
        else:
            beta = np.append(beta,(np.dot(np.dot(np.linalg.inv(temp2), temp1), y)),axis=1)
        i2+=1
    y1 = ri_rf[60+i1][:]
    i2=0
    ones1 = np.ones((10, 1))
    beta1 = np.reshape((beta[1][:]),(10,1))
    x1 = np.append(ones1,beta1,axis=1)
    temp3 = np.transpose(x1)
    temp4 = np.dot(temp3, x1)

    g1[i1][:] = np.dot(np.dot(np.linalg.inv(temp4),temp3),y1)
    x2 = lgmarket[i1+60]
    x2 = np.transpose(x2)
    x2 = np.append(x1,x2,axis=1)
    tempx = bm[i1+60]
    tempx = np.transpose(tempx)
    x2 = np.append(x2,tempx,axis=1)
    temp3 = np.transpose(x2)
    temp4 = np.dot(temp3, x2)
    g2[i1][:] = np.dot(np.dot(np.linalg.inv(temp4), temp3), y1)

    x3 = lgmarket[i1+60]
    x3 = np.transpose(x3)
    x3 = np.append(ones1,x3,axis=1)
    temp3 = np.transpose(x3)
    temp4 = np.dot(temp3, x3)
    g3[i1][:] = np.dot(np.dot(np.linalg.inv(temp4), temp3), y1)

    x4 = bm[i1+60]
    x4 = np.transpose(x4)
    x4 = np.append(ones1, x4, axis=1)
    temp3 = np.transpose(x4)
    temp4 = np.dot(temp3, x4)
    g4[i1][:] = np.dot(np.dot(np.linalg.inv(temp4), temp3), y1)

    x5 = lgmarket[i1 + 60]
    x5 = np.transpose(x5)
    x5 = np.append(ones1, x5, axis=1)
    tempx = bm[i1 + 60]
    tempx = np.transpose(tempx)
    x5 = np.append(x5, tempx, axis=1)
    temp3 = np.transpose(x5)
    temp4 = np.dot(temp3, x5)
    g5[i1][:] = np.dot(np.dot(np.linalg.inv(temp4), temp3), y1)


    i1+=1
np.set_printoptions(precision=5,suppress=True)
df = pd.DataFrame(g1)
df2 = pd.DataFrame(g2)
df3 = pd.DataFrame(g3)
df4 = pd.DataFrame(g4)
df5 = pd.DataFrame(g5)
#OUTPUT Q1
print("Q1: MEAN: ",df[0].mean(),df[1].mean())
t1_1 = df[0].mean()/df[0].std()*math.sqrt(180)
t1_2 = df[1].mean()/df[1].std()*math.sqrt(180)
p1= scipy.stats.t.sf(abs(t1_1), df=180)
p2 = scipy.stats.t.sf(abs(t1_2), df=180)
print("Q1: T-Score: ",t1_1 ,t1_2," P-value: ",p1,p2)

#OUTPUT Q2
print("Q2: MEAN: ",df2[0].mean(),df2[1].mean(),df2[2].mean(),df2[3].mean())
t2_1 = df2[0].mean()/df2[0].std()*math.sqrt(180)
t2_2 = df2[1].mean()/df2[1].std()*math.sqrt(180)
t2_3= df2[2].mean()/df2[2].std()*math.sqrt(180)
t2_4 = df2[3].mean()/df2[3].std()*math.sqrt(180)
p1= scipy.stats.t.sf(abs(t2_1), df=180)
p2 = scipy.stats.t.sf(abs(t2_2), df=180)
p3 = scipy.stats.t.sf(abs(t2_3), df=180)
p4 = scipy.stats.t.sf(abs(t2_4), df=180)
print("Q2: T-Score: ",t2_1 ,t2_2,t2_3,t2_4," P-value: ",p1,p2,p3,p4)

#Q3
print("Q3: MEAN: ",df3[0].mean(),df3[1].mean())
t3_1 = df3[0].mean()/df3[0].std()*math.sqrt(180)
t3_2 = df3[1].mean()/df3[1].std()*math.sqrt(180)
p1= scipy.stats.t.sf(abs(t3_1), df=180)
p2 = scipy.stats.t.sf(abs(t3_2), df=180)
print("Q3: T-Score: ",t3_1 ,t3_2," P-value: ",p1,p2)

#Q4
print("Q4: MEAN: ",df4[0].mean(),df4[1].mean())
t4_1 = df4[0].mean()/df4[0].std()*math.sqrt(180)
t4_2 = df4[1].mean()/df4[1].std()*math.sqrt(180)
p1= scipy.stats.t.sf(abs(t4_1), df=180)
p2 = scipy.stats.t.sf(abs(t4_2), df=180)
print("Q4: T-Score: ",t4_1 ,t4_2," P-value: ",p1,p2)

#Q5
print("Q5: MEAN: ",df5[0].mean(),df5[1].mean(),df5[2].mean())
t5_1 = df5[0].mean()/df5[0].std()*math.sqrt(180)
t5_2 = df5[1].mean()/df5[1].std()*math.sqrt(180)
t5_3= df5[2].mean()/df5[2].std()*math.sqrt(180)
p1= scipy.stats.t.sf(abs(t5_1), df=180)
p2 = scipy.stats.t.sf(abs(t5_2), df=180)
p3 = scipy.stats.t.sf(abs(t5_3), df=180)

print("Q5: T-Score: ",t5_1 ,t5_2,t5_3," P-value: ",p1,p2,p3)

