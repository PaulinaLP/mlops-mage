import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import vstack

def vectorize_train(df, chunksize=10000):
    vectorizer = DictVectorizer(sparse=True)
    first_chunk = True
    feature_matrix = None 
    num_row=len(df)
    print(num_row)
    row_fin=chunksize
    row_st=0

    # Reading the CSV file in chunks
    while row_fin<(num_row+chunksize):
        row_fin=num_row if num_row<row_fin else row_fin
        print(str(row_fin))
        chunk=df.iloc[row_st:row_fin]
        data = chunk[['PULocationID', 'DOLocationID']].astype(str).to_dict(orient='records')
        if first_chunk:
            feature_matrix = vectorizer.fit_transform(data)
            first_chunk = False
        else:
            partial_matrix = vectorizer.transform(data)
            feature_matrix = vstack([feature_matrix, partial_matrix])
        row_fin=row_fin+chunksize
        row_st=row_st+chunksize

    dimensionality = len(vectorizer.vocabulary_)
    print("Dimensionality of feature matrix (train):", dimensionality)
    return [feature_matrix, vectorizer]