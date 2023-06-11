import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import array
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import altair as alt
from sklearn.utils.validation import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image


primaryColor="#6eb52f"
backgroundColor="yellow"
secondaryBackgroundColor="#e0e0ef"
textColor="#262730"
font="sans serif"



st.title("MACHINE LEARNING")
st.title("Aplikasi Untuk Memprediksi Penyakit Liver")
st.write("By: Aisyah Meta Sari Putri - 200411100031, dan Isnaini - 200411100038")
st.write("Machine Learning B")
data,preprocessing, modeling, implementation = st.tabs(["Data", "Prepocessing", "Modeling", "Implementation"])

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv")
# Memberikan nama fitur pada tiap kolom dataset
df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                'Albumin', 'Albumin_and_Globulin_Ratio', 'Liver_disease']


with data:
    st.write("""# Tentang Dataset dan Aplikasi""")

    image = Image.open('liver.jpeg')

    st.image(image, caption='liver')

    st.write("Dataset yang digunakan adalah Indian Patient Liver dataset yang diambil dari https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)")
    st.write("Total datanya adalah 582 dengan data training 80% (415) dan data testing 20% (167)")


with preprocessing:
    
    st.write("""# Preprocessing""")
    df.describe()
    df = df.fillna(df.mean(numeric_only=True))
    
    df.Liver_disease.value_counts()
    # Buat dictionary untuk melakukan mapping variabel Liver_disease
    liver = {'Liver_disease': {1: 1, 2: 0}}

    df = df.replace(liver)
    df

   # Buat dictionary untuk melakukan mapping variabel kategorikal menjadi variabel numerikal
    gender = {'Gender': {"Male": 1, "Female": 0}}

    df = df.replace(gender)
    df
    
    X = df.drop(columns="Liver_disease")
    y = df.Liver_disease
   
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.Liver_disease).columns.values.tolist()
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X
 
    le.inverse_transform(y)
    X.shape, y.shape
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    showtraintest = st.button("Data Training and Testing")
    if showtraintest:
        """## Data Training"""
        X_train
        """## Data Testing"""
        X_test
    
with modeling:

    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    SVM = st.checkbox('SVM')
    Random_Forest = st.checkbox('Random Forest')
    mod = st.button("Modeling")
    
    # SVM
    kernel = 'linear'
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # RF

    rf_model = RandomForestClassifier()

    # Melatih model dengan data latih
    rf_model.fit(X_train, y_train)
    # prediction
    rf_model.score(X_test, y_test)
    y_pred = rf_model.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if SVM :
        if mod:
            st.write("Model SVM accuracy score : {0:0.2f}" . format(skor_akurasi))
    if Random_Forest :
        if mod :
            st.write("Model Random Forest accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [skor_akurasi,akurasiii],
            'Nama Model' : ['SVM','Random Forest']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)


with implementation:
    st.write("# Implementation")
    Age = st.number_input('Masukkan Umur Pasien')

    # GENDER
    gender = st.radio("Gender",('Male', 'Female'))
    if gender == "Male":
        Gender = 1
    elif gender == "Female" :
        Gender = 0

    Total_Bilirubin = st.number_input('Masukkan Hasil Test Total_Bilirubin (Contoh : 10.9)')
    Direct_Bilirubin = st.number_input('Masukkan Hasil Test Direct_Bilirubin (Contoh : 5.5)')
    Alkaline_Phosphotase = st.number_input('Masukkan Hasil Test Alkaline_Phosphotase (Contoh : 699)')
    Alamine_Aminotransferase = st.number_input('Masukkan Hasil Test Alamine_Aminotransferase (Contoh : 64)')
    Aspartate_Aminotransferase = st.number_input('Masukkan Hasil Test Aspartate_Aminotransferase (Contoh : 100)')
    Total_Protiens = st.number_input('Masukkan Hasil Test Total_Protiens (Contoh : 7.5)')
    Albumin = st.number_input('Masukkan Hasil Test Albumin (Contoh : 3.2)')
    Albumin_and_Globulin_Ratio = st.number_input('Masukkan Hasil Albumin_and_Globulin_Ratio (Contoh : 0.74)')
   



    def submit():
        # input
        inputs = np.array([[
            Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
                Albumin, Albumin_and_Globulin_Ratio
            ]])
        le = joblib.load("le.save")

        if skor_akurasi > akurasiii:
            model = joblib.load("svm.joblib")

        elif akurasiii > skor_akurasi:
            model = joblib.load("rf.joblib")

        y_pred3 = model.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda diprediksi cenderung : {le.inverse_transform(y_pred3)[0]}")
        st.write("0 = Tidak Terkena Liver")
        st.write("1 = Terkena Liver")
    all = st.button("Submit")
    if all :
        st.balloons()
        submit()


