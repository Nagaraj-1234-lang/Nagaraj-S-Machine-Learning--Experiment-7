# Nagaraj-S-Machine-Learning--Experiment-7
## DATE : 13/10/2026
### AIM :
Build a ColumnTransformer pipeline to preprocess numeric and categorical features safely.



### ALGORITHM :
Apply median imputation & standard scaling to numeric features; one-hot encode categorical features — all inside a pipeline to avoid leakage.

### THEORY:
Pipelines ensure transformations are fit on training data only; critical for valid evaluation.



### INPUT : 


```
# Experiment 7
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
df = pd.DataFrame({'age':[25,35,
None, 45],'income':[50,40,30,60],'city':['X','Y','X','Z'],'target':[0,1,0,1]})
X = df[['age','income','city']]; y = df['target']
numeric_features = ['age','income']
numeric_transformer = Pipeline([('imputer',
SimpleImputer(strategy='median')), ('scalar', StandardScaler())])
cat_features = ['city']; cat_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', numeric_transformer,
numeric_features), ('cat', cat_transformer, cat_features)])
clf = Pipeline([('pre', preprocessor), ('rf', RandomForestClassifier(random_state=42))])
clf.fit(X,y)
print('Preprocessor output shape:',
preprocessor.transform(X).shape)
print('Pipeline trainded successfully.')
```

### OUTPUT: 
```
Preprocessor output shape: (4, 5)
Pipeline trainded successfully.

```
### RESULT :
Students learn to encapsulate preprocessing to avoid leakage and ensure correct application during training and prediction.
