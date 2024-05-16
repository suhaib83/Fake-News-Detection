import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

df = pd.read_csv("../Data/multimodal_train.csv")  #(625783, 86)



print(df['6_way_label'].value_counts())
data_count = df['6_way_label'].value_counts()
plt.figure(figsize=(7,6))
ax=plt.subplot(111)
plt.bar(data_count.index, data_count.values)
plt.grid()
plt.ylabel('Number of Occurrences', fontsize=20)
#plt.xlabel('Normality', fontsize=20)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure(figsize=(8,8))
plt.pie(df['6_way_label'].value_counts(),labels=data_count.index,autopct='%0.2f%%')
plt.title('Pie Chart Distribution of Multi-class Labels')
plt.show()

df['6_way_label'] = pd.Categorical(df['6_way_label']).codes
#Min Max Normlization