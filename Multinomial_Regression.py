### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mode = pd.read_csv("DSProject.csv")

mode.columns

#Index(['Unnamed: 0', 'AppName', 'Company', 'StarRating', 'TotalReviews',
#       'TotalDownloads', 'ContainsAds', 'Editors', 'RatedAgeGroup', 'Category',
#       'Score(Avg)', 'TopFeatures'],
#      dtype='object')

mode = mode.drop(['Unnamed: 0', 'AppName', 'Company', 'StarRating', 'TotalReviews', 'TotalDownloads', 'ContainsAds', 'Editors', 'RatedAgeGroup', 'Score(Avg)'], axis=1)

mode.head()


#   Category                                        TopFeatures
#0  Arts&design  ('urdu designer', 356), ('logo maker', 171), (...
#1  Arts&design  ('urdu designer', 356), ('logo maker', 171), (...
#2  Arts&design  ('urdu designer', 356), ('logo maker', 171), (...
#3  Arts&design  ('urdu designer', 356), ('logo maker', 171), (...
#4  Arts&design  ('urdu designer', 356), ('logo maker', 171), (...

from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()

mode['Category'] = labelencoder.fit_transform(mode['Category'])


train, test = train_test_split(mode)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, :1], train.iloc[:, 1])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, :1]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:, 1], test_predict)
#0.6354

train_predict = model.predict(train.iloc[:, :1]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:, 1], train_predict) 
#0.6785

X = mode.iloc[:, :1]

y = mode.iloc[:, 1]

#Fitting model with trainig data
regressor.fit(X, y)

regressor = LogisticRegression(multi_class = "multinomial", solver = "newton-cg")

import pickle

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[8]]))
["('khan academy', 201), ('learn language', 180), ('learn dance', 177), ('learn code', 174), ('english language', 164), ('brain training', 140), ('read book', 129), ('learn programming', 121), ('learn python', 110), ('fun game', 97), ('daily workout', 69), ('night sky', 49), ('people musician', 39), ('trial charge', 36), ('astronomy lover', 35), ('learning platform', 33), ('video lecture', 33), ('dark mode', 30), ('user friendly', 28), ('people babel', 26), ('dark mode', 25), ('star gaze', 24), ('gain knowledge', 24), ('instructor class', 22), ('elevate lab', 20), ('star planet', 19), ('user interface', 18), ('learning platform', 18), ('code basic', 17), ('preach incorporate', 15), ('sky watch', 15), ('video player', 14), ('people stone', 14), ('star constellation', 13), ('word language', 12), ('financial aid', 12), ('stellar object', 11), ('sleek design', 11), ('online learning', 11), ('user friendly', 9), ('forum song', 9), ('desktop version', 9), ('premium version', 8), ('assist touch', 7), ('word click', 7),  ('neuronation sharp', 5), ('pedagogical approach', 5), ('dynamic immersion', 5), ('landscape mode', 4), ('playing luminosity', 3), ('desktop version', 3), ('pro version', 3), ('grade system', 2), ('native accent', 2), ('share knowledge', 2), ('brain fog', 2)"]
