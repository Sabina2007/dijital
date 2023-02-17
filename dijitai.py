#создай здесь свой индивидуальный проект!
import pandas as pd 
df = pd.read_csv('train.csv')
#df.info()

df.drop(['followers_count','id', 'people_main','has_photo', 'city', 'occupation_name', 'life_main', 'last_seen', 'career_start', 'career_end','bdate'], axis = 1, inplace = True)
     
#print(df.info())
#print (df['sex'].value_counts())
def sex_apply(sex):
    if sex == 2:
        return 0
    return 1
df['sex'] = df['sex'].apply(sex_apply)
#print(df['sex'].value_counts())        

df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)

#print(df['langs'].value_counts())
def langs_apply(langs):
    if langs.find('Русский') != -1:
        return 0
    return 1
df['langs'] = df['langs'].apply(langs_apply)
#print(df['education_status'].value_counts())        
#df.info()

def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    if edu_status == "Student (Bachelor's)" or edu_status == "Student (Specialist)" or edu_status == "Student (Master's)" :
        return 1
    if edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Specialist)" or edu_status == "Alumnus (Master's)" :
        return 2
    if edu_status == "PhD" or edu_status == "Candidate of Sciences" :
        return 3
df['education_status'] = df['education_status'].apply(edu_status_apply)


df['occupation_type'].fillna('university', inplace = True)
def ocu_type_apply(ocu_type):
    if ocu_type == 'university':
        return 1
    return 0
df['occupation_type'] = df['occupation_type'].apply(ocu_type_apply)       
df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
x = df.drop('result', axis = 1) 
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_test)
print(y_pred)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred) * 100, 2))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
