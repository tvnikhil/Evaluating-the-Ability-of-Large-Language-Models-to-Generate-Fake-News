import pandas as pd

class DataOperations:
    def preProcess(self):
        dfPath = "/Users/thanikella_nikhil/Projects-Courses/ML/Paper1/data/original/test.pkl"
        toDrop = ["content_emotion", "comments_emotion", "emotion_gap", "style_feature"]
        temp = pd.read_pickle(dfPath)
        df = temp.drop(toDrop, axis=1, inplace=False)
        nfdf = df[df['label'] == 0] #True
        fdf = df[df['label'] == 1] #False (Fake)
        fdfNoLabel = fdf.drop(columns=['label'], inplace=False)

        self.nfdf, self.fdf = nfdf, fdf
        self.fdfNoLabel = fdfNoLabel

        return self.fdfNoLabel