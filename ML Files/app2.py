import streamlit as st
def app():
    import mysql.connector
    import pickle
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors
    import pandas as pd
    from PIL import Image

    vec = pickle.load(open("df1.pkl", 'rb'))
    loaded_model = pickle.load(open("tip.sav", 'rb'))
    products = pd.read_csv("products.csv")
    purchase = pd.read_csv("Purchase_His.csv")
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="root",
      database="xenathon"
    )
    st.write("CUSTOMER TRACKING")
    mycursor = mydb.cursor()
    a = st.text_input("Please enter customer id")
    if a:
        f="SELECT userid,review,mailid,prodname FROM customers WHERE userid="+"'"+str(a)+"'"
        mycursor.execute(f)

        myresult = mycursor.fetchall()
        if len(myresult)!=0:
            l = []
            pos = []
            neg = []
            posr = []
            negr = []
            for x in myresult:
                rv = [x[1]]
                rv = vec.transform(rv)
                predct = loaded_model.predict(rv)
                l.append(predct[0])
                if (predct[0] == 0):
                    neg.append(x[2])
                    negr.append(x[1])
                else:
                    pos.append(x[2])
                    posr.append(x[1])
            sz = len(l)
            positive_percent = ((sz - len(neg)) / sz) * 100
            st.write(positive_percent)
            comment_words = ''
            stopwords = set(STOPWORDS)
            for val in posr:
                val = str(val)
                tokens = val.split()
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()
                comment_words += " ".join(tokens) + " "

            wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate(comment_words)

            plt.figure(figsize=(8, 8), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad=0)

            plt.savefig('positive.png')
            image = Image.open('positive.png')
            st.image(image, caption='Positive Review Word cloud')

            for val in negr:
                val = str(val)
                tokens = val.split()
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()
                comment_words += " ".join(tokens) + " "

            wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate(comment_words)

            plt.figure(figsize=(8, 8), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad=0)

            plt.savefig('negative.png')
            image = Image.open('negative.png')
            st.image(image, caption='negative Review Word cloud')


            final_dataset = purchase.pivot(index='StockCode', columns='CustomerID', values='Purchase')
            final_dataset.fillna(0, inplace=True)
            no_user_purchased = purchase.groupby('StockCode')['Purchase'].agg('count')
            no_products_purchased = purchase.groupby('CustomerID')['Purchase'].agg('count')
            final_dataset = final_dataset.loc[no_user_purchased[no_user_purchased > 10].index, :]
            final_dataset = final_dataset.loc[:, no_products_purchased[no_products_purchased > 50].index]
            csr_data = csr_matrix(final_dataset.values)
            final_dataset.reset_index(inplace=True)
            knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
            knn.fit(csr_data)

            st.write('ALL POSITIVE REVIEWS')
            for i in posr:
                st.write(i)

            st.write('ALL NEGATIVE REVIEWS')
            for i in negr:
                st.write(i)

        else:
            st.write("Records Not Found")