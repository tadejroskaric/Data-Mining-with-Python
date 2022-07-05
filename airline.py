import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

# Importing the data
df = pd.read_csv("airline.csv", index_col="id")

# Data cleaning
df = df.drop(columns=["Unnamed: 0"])
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(value=df["Arrival Delay in Minutes"].mean())

# Seaborn visualisations
sns.set(font_scale=1)
sns.catplot(x="Gender", data=df, kind="count", hue="Satisfaction")
sns.catplot(x="Class", data=df, kind="count", hue="Satisfaction")
sns.catplot(x="Age", data=df, kind="count", hue="Satisfaction", aspect=3.0)
sns.catplot(x="Age", data=df, kind="count", hue="Customer Type", aspect=3.0)
sns.catplot(x=["Neutral or dissatisfied", "Satisfied"], data=df, y=df["Satisfaction"].value_counts(), kind="bar")
plt.ylabel("count")
plt.show()

# Data transformation
df["Satisfaction"] = df["Satisfaction"].apply(lambda x: 1 if x == "Satisfied" else 0)
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Female" else 0)
df["Customer Type"] = df["Customer Type"].apply(lambda x: 1 if x == "Loyal Customer" else 0)
df["Type of Travel"] = df["Type of Travel"].apply(lambda x: 1 if x == "Business travel" else 0)
df["Class"] = df["Class"].apply(lambda x: 1 if x == "Business" else 0)

# Correlation matrix
corr = df.corr(method="spearman")
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr, annot=True, mask=mask)
# Formatting of x labels
plt.xticks(rotation=35, ha="right")
plt.tick_params(labelsize=8)
plt.show()

# Train and test split
df1 = df.drop(columns=["Satisfaction"])
x_train, x_test, y_train, y_test = train_test_split(df1, df["Satisfaction"], test_size=0.2)

# Decision tree model
model = DecisionTreeClassifier()
model = model.fit(x_train, y_train)
y_predict = model.predict(x_test)
matrix = confusion_matrix(y_test, y_predict)

# Displaying the confusion matrix (numbers will be a bit different with each iteration)
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
display.plot(values_format=",.0f", colorbar=False)
plt.grid(False)
plt.show()

# Extracting importance of features
importance = model.feature_importances_
importance = pd.Series(importance, index=x_train.columns)
importance.nlargest(23).plot(kind="bar")
plt.xticks(rotation=35, ha="right")
plt.tick_params(labelsize=8)
plt.show()
