<h1 align="center"> Intel_Unnati_Project-Whitecaps

## DESCRIPTION
This project on fake news detection using machine learning and python  proposes a system for fake news detection that uses machine learning techniques 
implemented in python.This project uses machine learning algorithms to detect fake news which can be of great usage in the identification of correct 
infoormation.In this paper it is seeked to produce a model that can accurately predict the likelihood that a given article is fake news harnessing the
power of machine language and python and  thus prevent the disemmination of fake news to an extent.
 
 

 ### TIMELINE OF THE PROJECT
 
 ***24 May  2023*** - First Interaction with mentor 
 
 ***30 May  2023*** - Submitted project work plan
 
 ***1 June  2023*** - Basic interaction inteldevcloud and OneApi
 
***2 June  2023*** - Hands on session on inteldevcloud and OneApi
 
 ***7 June  2023*** - Interaction with mentor for doubt clearance
 
 ***8 June  2023*** - Started coding part
 
 ***14 June 2023*** - Second live training session by intel
 
 ***17 June 2023*** - Interaction meeting with industry mentor
 
 ***19 June 2023*** - Started report making
 
 ***24 June 2023*** - Interction meeting with industry mentor
 
 ***1  July 2023*** - Interaction meeting with industry mentor
 
 ***7  July 2023*** - Interaction meeting with industry  mentor 
 
 ***10 July 2023*** - Made a video demo
 
 ***14 July 2023*** - Final project submssion

 ### MAJOR  CODE SECTIONS
  **User defined function for data visulatisation**
```
 def visualize(dataFile,feature):          
       plt.figure(figsize = (6,4))
       sns.set(style = "whitegrid",font_scale = 1.0)
       chart = sns.countplot(x = feature, data = data)
       chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
       plt.show()
```
**Data cleaning and pre-processing**
```
def Check_forNAN(data):                                                       
    print("Wait...Checking for NANs in the Dataset is progressing...")
    print("Total NANs:",data.isnull().sum())
    print("Checking is completed successfully...\n")
    print(10*"--","\n")
    print("Summary of the dataframe:......\n")
    data.info()

        print("check finished.")
 # tokenization
    def tokenize(column):
        tokens = nltk.word_tokenize(column)
        return [w for w in tokens if w.isalpha()]
    # stopwords removal
    def remove_stopwords(tokenized_column):
        stops = set(stopwords.words("english"))
        return [word for word in tokenized_column if not word in stops]
    # stemming
    def apply_stemming(tokenized_column):
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokenized_column]
    # creating words bag
    def rejoin_words(tokenized_column):
        return ( " ".join(tokenized_column))
```
**Preprocess function**
```
 ## Creating Data cleaning and pre-processing function
def PreProcess(data):
    data['tokenized'] = data.apply(lambda x: tokenize(x['text']), axis=1)
    data['stopwords_removed'] = data.apply(lambda x: remove_stopwords(x['tokenized']), axis=1)
    data['stemmed'] = data.apply(lambda x: apply_stemming(x['stopwords_removed']), axis=1)
    data['rejoined'] = data.apply(lambda x: rejoin_words(x['stemmed']), axis=1)

```
 **Splitting the data frame into data and label**
```
                                                      
data.label = data.label.astype(str)                              
dict = { 'REAL' : 1 , 'FAKE' : '0'}
data['label'].head()
X = data['rejoined']
y = data['label']        
count_vectorizer = CountVectorizer()                           //vectorisation
count_vectorizer.fit_transform(X)
freq_term_matrix = count_vectorizer.transform(X)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
print(tf_idf_matrix)
```
**Visulisation of bag of words and word count**
```
fake_data = data[data["label"] == "1"]
all_words = ' '.join([text for text in fake_data.rejoined])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))                                                          
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()
    counter(data[data['label'] == '1'], 'rejoined', 20)
```
 **Dataset preperation**
```
x_train, x_test, y_train, y_test = train_test_split(tf_idf_matrix,y, random_state=21)     

```

**Model training and evaluation**
```
Model training and evaluation
#logistic regrssion
accuracy_values=[]
logitmodel = LogisticRegression()
logitmodel.fit(x_train, y_train)
y_pred=logitmodel.predict(x_test)
Accuracy = logitmodel.score(x_test, y_test)
accuracy_values.append((Accuracy*100))
print(Accuracy*100)
 
#Naive bayes classification
NB = MultinomialNB()
NB.fit(x_train, y_train)
y_pred=NB.predict(x_test)
Accuracy = NB.score(x_test, y_test)
accuracy_values.append((Accuracy*100))
print(Accuracy*100)
 
#Decision tree
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
Accuracy = clf.score(x_test, y_test)
accuracy_values.append((Accuracy*100))
print(Accuracy*100)

#Passive aggresive clasiifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(x_train,y_train)
y_pred=pac.predict(x_test)
score=accuracy_score(y_test,y_pred)
accuracy_values.append((score*100))
print(f'Accuracy: {round(score*100,2)}%')

#Random Forest regression
rf = RandomForestRegressor(**params).fit(x_train, y_train)
train_patched = timer() - start
y_pred = rf.predict(x_test)
mse_opt = metrics.mean_squared_error(y_test, y_pred)
rf = RandomForestRegressor(**params).fit(x_train, y_train)

```
 
 
