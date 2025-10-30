import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

# Task: of importing and loading the dataset into a pandas DataFrame for this exercise is to
df = pd.read_csv("IMDB Dataset.csv")
print(df.head())
'''
                                                  review sentiment
    0  One of the other reviewers has mentioned that ...  positive
    1  A wonderful little production. <br /><br />The...  positive
    2  I thought this was a wonderful way to spend ti...  positive
    3  Basically there's a family where a little boy ...  negative
    4  Petter Mattei's "Love in the Time of Money" is...  positive
'''


# Task: to check for missing values ​​in a data frame
print(df.isna().sum())
'''
    review       0
    sentiment    0
    dtype: int64  
'''


# Task: Check if any comment string is empty
empty = df[df['review'] == ""]
print(empty)

e = df['review'].str.isspace().sum()
print(e)

print(df.info())
'''
    RangeIndex: 50000 entries, 0 to 49999 
    Data columns (total 2 columns):       
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   review     50000 non-null  object
     1   sentiment  50000 non-null  object
    dtypes: object(2)
    memory usage: 781.4+ KB
'''


# Task: Determine the number for each label
print(df['sentiment'].value_counts())
'''
    sentiment
    positive    25000        
    negative    25000        
    Name: count, dtype: int64
'''


# Task: Find the top 20 words that are not English stop words in each tag type using the CountVectorizer model.
cv = CountVectorizer(stop_words='english')

matrix_neg = cv.fit_transform(df[df['sentiment'] == 'negative']['review'])
feature_name_neg = cv.get_feature_names_out()
sum_matrix_neg = matrix_neg.sum(axis=0).tolist()[0]

s_f = zip(feature_name_neg, sum_matrix_neg)
print("Top 20 words used for Negative reviews:")
print(sorted(s_f, key=lambda x: -x[1])[:20])
'''
    Top 20 words used for Negative reviews:
    [('br', 103997), ('movie', 50117), ('film', 37595), 
    ('like', 22458), ('just', 21075), ('good', 14728), 
    ('bad', 14726), ('time', 12358), ('really', 12355), 
    ('don', 10622), ('story', 10185), ('people', 9469), 
    ('make', 9355), ('movies', 8313), ('plot', 8214), 
    ('acting', 8087), ('way', 7780), ('characters', 7353), 
    ('watch', 7220), ('think', 7129)]
'''


matrix_pos = cv.fit_transform(df[df['sentiment'] == 'positive']['review'])
feature_name_pos = cv.get_feature_names_out()
sum_matrix_pos = matrix_pos.sum(axis=0).tolist()[0]

print("\n\nTop 20 words used for positive reviews:")
s_f = zip(feature_name_pos, sum_matrix_pos)
print(sorted(s_f, key=lambda x: -x[1])[:20])
'''
    Top 20 words used for positive reviews:
    [('br', 97954), ('film', 42110), ('movie', 37854), 
    ('like', 17714), ('good', 15025), ('just', 14109), 
    ('great', 12964), ('story', 12934), ('time', 12752), 
    ('really', 10739), ('people', 8719), ('love', 8692), 
    ('best', 8510), ('life', 8137), ('way', 7865), 
    ('films', 7601), ('think', 7208), ('characters', 7103), 
    ('don', 7001), ('movies', 6996)]
'''

 
# Task: is to split the data into features and a label (X) and (Y) and then perform the training-test split.
df = df.sample(frac=0.04, random_state=42).reset_index(drop=True)
print(df.count())
'''
    review       1000
    sentiment    1000
    dtype: int64     
'''

X = df['review']
Y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)


# Task: Create a pipeline that generates a TF-IDF vector from raw text data, then fit that pipeline to the training data.
pipe1 = Pipeline(steps=[
    ('tfidf', TfidfVectorizer()),
    ('svc', LinearSVC(dual='auto'))
])
pipe1.fit(x_train, y_train)


pipe2 = Pipeline(steps=[
    ('tfidf', TfidfVectorizer()),
    ('mnb', MultinomialNB())
])
pipe2.fit(x_train, y_train)


# Task: Create a classification report and draw a clutter matrix based on the results of your pipeline.
pred_pipe1 = pipe1.predict(x_test)
print(f"classification_report:\n{classification_report(y_test, pred_pipe1)}")
'''
    classification_report:
                  precision    recall  f1-score   support

        negative       0.86      0.86      0.86       205
        positive       0.85      0.85      0.85       195

        accuracy                           0.85       400
       macro avg       0.85      0.85      0.85       400
    weighted avg       0.85      0.85      0.85       400
'''
ConfusionMatrixDisplay.from_predictions(y_test, pred_pipe1)
plt.show()


pred_pipe2 = pipe2.predict(x_test)
print(f"classification_report:\n{classification_report(y_test, pred_pipe2)}")
'''
    classification_report:
                  precision    recall  f1-score   support

        negative       0.67      0.95      0.79       205
        positive       0.91      0.51      0.65       195

        accuracy                           0.73       400
       macro avg       0.79      0.73      0.72       400
    weighted avg       0.79      0.73      0.72       400
'''
ConfusionMatrixDisplay.from_predictions(y_test, pred_pipe2)
plt.show()


acc1 = accuracy_score(y_test, pred_pipe1)
acc2 = accuracy_score(y_test, pred_pipe2)

print(f"\nModel Comparison:\nLinearSVC Accuracy: {acc1:.3f}\nMultinomialNB Accuracy: {acc2:.3f}")

if acc1 > acc2:
    print("LinearSVC performed better overall.")
else:
    print("MultinomialNB performed better overall.")
'''
    Model Comparison:
    LinearSVC Accuracy: 0.855
    MultinomialNB Accuracy: 0.735
    LinearSVC performed better overall.
'''

'''
    Final Conclusion:
    The *LinearSVC + TF-IDF* pipeline achieved the most stable and accurate performance.
    This project demonstrates a complete NLP workflow — from raw text to sentiment prediction — and provides a foundation for future deep learning sentiment models.
'''
