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
    st.write("PRODUCT PERFORMANCE CHECKER")
    mycursor = mydb.cursor()
    a = st.text_input("Please enter product id")
    if a:
        f="SELECT userid,review,mailid,prodname FROM customers WHERE prodid="+"'"+str(a)+"'"
        mycursor.execute(f)

        myresult = mycursor.fetchall()
        if len(myresult)!=0:
            data = myresult[1][3]
            qwerty="Your Chosen Product is: "+str(data)
            st.write(qwerty)
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


            def get_product_recommendation(product_name):
                n_products_to_reccomend = 10
                product_list = products[products['Description'].str.contains(product_name)]
                if len(product_list):
                    product_idx = product_list.iloc[0]['StockCode']
                    product_idx = final_dataset[final_dataset['StockCode'] == product_idx].index[0]
                    distances, indices = knn.kneighbors(csr_data[product_idx], n_neighbors=n_products_to_reccomend + 1)
                    rec_product_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                             key=lambda x: x[1])[:0:-1]
                    recommend_frame = []
                    for val in rec_product_indices:
                        product_idx = final_dataset.iloc[val[0]]['StockCode']
                        idx = products[products['StockCode'] == product_idx].index
                        recommend_frame.append({'Description': products.iloc[idx]['Description'].values[0],
                                            'StockCode': products.iloc[idx]['StockCode'].values[0], 'Distance': val[1]})
                    df = pd.DataFrame(recommend_frame, index=range(1, n_products_to_reccomend + 1))
                    return st.write(df)
                else:
                    return "Nothing To Recommend"


            data = myresult[1][3]
            print(data)
            get_product_recommendation(data)


            st.write('ALL POSITIVE REVIEWS')
            for i in posr:
                st.write(i)

            st.write('ALL NEGATIVE REVIEWS')
            for i in negr:
                st.write(i)

        else:
            st.write("Records Not Found")