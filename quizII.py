import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv

ratings = pd.read_csv("gameratings.csv")
testgame = pd.read_csv("test_esrb.csv")
targetnames = pd.read_csv("target_names.csv")

r = ratings[
    [
        "alcohol_reference",
        "animated_blood",
        "blood",
        "blood_and_gore",
        "cartoon_violence",
        "crude_humor",
        "drug_reference",
        "fantasy_violence",
        "intense_violence",
        "language",
        "lyrics",
        "mature_humor",
        "mild_blood",
        "mild_cartoon_violence",
        "mild_fantasy_violence",
        "mild_language",
        "mild_lyrics",
        "mild_suggestive_themes",
        "mild_violence",
        "no_decriptors",
        "nudity",
        "partial_nudity",
        "sexual_content",
        "sexual_themes",
        "simulated_gambling",
        "strong_janguage",
        "strong_sexual_content",
        "suggestive_themes",
        "use_of_alcohol",
        "use_of_drugs_and_alcohol",
        "violence",
    ]
].values

x = ratings[["Target"]].values.reshape(-1,1)

testtrain = testgame[
    [
        "alcohol_reference",
        "animated_blood",
        "blood",
        "blood_and_gore",
        "cartoon_violence",
        "crude_humor",
        "drug_reference",
        "fantasy_violence",
        "intense_violence",
        "language",
        "lyrics",
        "mature_humor",
        "mild_blood",
        "mild_cartoon_violence",
        "mild_fantasy_violence",
        "mild_language",
        "mild_lyrics",
        "mild_suggestive_themes",
        "mild_violence",
        "no_decriptors",
        "nudity",
        "partial_nudity",
        "sexual_content",
        "sexual_themes",
        "simulated_gambling",
        "strong_janguage",
        "strong_sexual_content",
        "suggestive_themes",
        "use_of_alcohol",
        "use_of_drugs_and_alcohol",
        "violence",
    ]
].values

testtarget = testgame[["Target"]].values.ravel()

game = testgame[["Title"]].values.ravel()

print(r.shape)
print(x.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier

knn.fit(X=r, y=x.ravel())

predicted = knn.predict(X=testtrain)
expected = testtarget
list2 = []
list3 = []
dictionary = {}

with open("target_names.csv",mode="r") as ai:
    reader = csv.reader(ai)
    csvdictionary = {rows[0]: rows[2] for rows in reader}

print(csvdictionary)
print(csvdictionary.get('1'))

for a in predicted:
    list2.append(csvdictionary.get(str(a)))

for n in expected:
    list3.append(csvdictionary.get(str(n)))

total = []

for b in zip(game,list2,list3):
    total.append(b)

print('************************')
print("Correct")
print(format(knn.score(testtrain,testtarget),".2%"))
print('************************')
wrong = [(a,b,c) for (a,b,c) in zip (game,list2,list3) if b != c] 

pd.set_option('display.maxrows',None)
pd.set_option('display.maxcol',None)
pd.set_option('display.width',None)
pd.set_option('display.maxcolwidth',None)

wrongdata = pd.DataFrame(wrong, columns=['title','predicted','expected'])
print('************************')
print('Incorrect')

print(wrongdata)

header = ['Title','Predicted','Expected']

with open('game_output.csv','w',newline='') as k:
    writer = csv.writed(k)
    writer.writerow(header)
    writer.writerows(total)

print(' ')
print('Done')