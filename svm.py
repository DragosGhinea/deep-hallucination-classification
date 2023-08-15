import time
import numpy as np
from sklearn import svm, metrics, preprocessing
from PIL import Image
from sklearn.utils import shuffle, class_weight
from skimage.feature import hog


# Preia imaginea si o face un np_array (shape: 64, 64, 3 = 64x64 pixeli RGB)
def load_img(folder, nume):
    img = Image.open(f"{folder}/{nume}")
    img_array = np.array(img)
    return img_array


# Pentru numele unui fisier csv, returnam tuplu (imagini (ca np_array), labeluri (int))
# In cazul datelor de test, se returneaza doar imaginile
def load_data(tip_data):
    date = np.genfromtxt(f"{tip_data}.csv", delimiter=',', dtype=None, names=True, encoding='utf8')
    imagini = date['Image']
    imagini = [load_img(f"{tip_data}_images", nume_img) for nume_img in imagini]

    if tip_data == "test":
        return imagini

    labeluri = date['Class']
    return imagini, labeluri.astype(int)


# Histogram of Oriented Gradients
def extract_hog_features(imagini):
    hog_features = []
    for img in imagini:
        hog_feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1)
        hog_features.append(hog_feature)
    return np.array(hog_features)


# Normalizare standard
def normalizeaza_data(train_data, test_data):
    normalizer = preprocessing.StandardScaler()
    normalizer.fit(train_data)
    return normalizer.transform(train_data), normalizer.transform(test_data)


# Afisam informatiile relevante
def afisare_statistici(labeluri_adevarate, predictii):
    accuracy = metrics.accuracy_score(labeluri_adevarate, predictii)
    print(f"Accuracy: {accuracy}")

    print(f"Recall (micro): {metrics.recall_score(labeluri_adevarate, predictii, average='micro')}")
    print(f"Recall (macro): {metrics.recall_score(labeluri_adevarate, predictii, average='macro')}")
    print(f"Recall (weighted): {metrics.recall_score(labeluri_adevarate, predictii, average='weighted')}")
    print(f"F1_score (micro): {metrics.f1_score(labeluri_adevarate, predictii, average='micro')}")
    print(f"F1_score (macro): {metrics.f1_score(labeluri_adevarate, predictii, average='macro')}")
    print(f"F1_score (weighted): {metrics.f1_score(labeluri_adevarate, predictii, average='weighted')}")

    matrice_confuzie = metrics.confusion_matrix(labeluri_adevarate, predictii)
    np.set_printoptions(threshold=np.inf)
    print("Matrice confuzie:")
    for linie in matrice_confuzie:
        print(*linie)

    print()
    print(metrics.classification_report(labeluri_adevarate, predictii, target_names=[f'Clasa {i}' for i in range(96)],
                                        zero_division=0))


# Rulare model pe datele de validare
def rulare_validare(C_var, kernel_var, gamma_var):
    imagini_antrenare, labeluri_antrenare = load_data("train")
    imagini_antrenare, labeluri_antrenare = shuffle(imagini_antrenare, labeluri_antrenare)
    hog_features_antrenare = extract_hog_features(imagini_antrenare)

    # raw pixel representation (12288) + hog (1764) = total features (14052)
    imagini_antrenare_unidimensional = np.hstack(
        (np.array([img.flatten() for img in imagini_antrenare]), hog_features_antrenare))

    # doar hog
    # imagini_antrenare_unidimensional = hog_features_antrenare

    # doar raw pixel representation
    # imagini_antrenare_unidimensional = np.array([img.flatten() for img in imagini_antrenare])

    imagini_validare, labeluri_validare = load_data("val")
    hog_features_validare = extract_hog_features(imagini_validare)

    # raw pixel representation (12288) + hog (1764) = total features (14052)
    imagini_validare_unidimensional = np.hstack(
        (np.array([img.flatten() for img in imagini_validare]), hog_features_validare))

    # doar hog
    # imagini_validare_unidimensional = hog_features_validare

    # doar raw pixel representation
    # imagini_validare_unidimensional = np.array([img.flatten() for img in imagini_validare])

    imagini_antrenare_unidimensional, imagini_validare_unidimensional = normalizeaza_data(imagini_antrenare_unidimensional,
                                                                                          imagini_validare_unidimensional)

    # Initializarea modelului Support Vector Classifier
    svm_model = svm.SVC(C=C_var, kernel=kernel_var, gamma=gamma_var)

    # Rularea modelului
    start_time = time.time()
    svm_model.fit(imagini_antrenare_unidimensional, labeluri_antrenare)
    predictii = svm_model.predict(imagini_validare_unidimensional)
    final_time = time.time()

    timp_executie = final_time - start_time
    print(f"Timp executie: {timp_executie} secunde")

    afisare_statistici(labeluri_validare, predictii)


# Rulare model pe datele de test
def rulare_submit(C_var, kernel_var, gamma_var):
    imagini_antrenare, labeluri_antrenare = load_data("train")
    imagini_antrenare, labeluri_antrenare = shuffle(imagini_antrenare, labeluri_antrenare)
    hog_features_antrenare = extract_hog_features(imagini_antrenare)
    imagini_antrenare_unidimensional = np.hstack(
        (np.array([img.flatten() for img in imagini_antrenare]), hog_features_antrenare))

    imagini_testare = load_data("test")
    hog_features_testare = extract_hog_features(imagini_testare)
    imagini_testare_unidimensional = np.hstack(
        (np.array([img.flatten() for img in imagini_testare]), hog_features_testare))

    imagini_antrenare_unidimensional, imagini_testare_unidimensional = normalizeaza_data(imagini_antrenare_unidimensional,
                                                                                         imagini_testare_unidimensional)

    # Initializarea modelului Support Vector Classifier
    svm_model = svm.SVC(C=C_var, kernel=kernel_var, gamma=gamma_var)

    # Rularea modelului
    start_time = time.time()
    svm_model.fit(imagini_antrenare_unidimensional, labeluri_antrenare)
    predictii = svm_model.predict(imagini_testare_unidimensional)
    final_time = time.time()

    timp_executie = final_time - start_time
    print(f"Timp executie: {timp_executie} secunde")

    with open("submission.csv", 'w') as out:
        out.write("Image,Class\n")
        for i in range(len(imagini_testare_unidimensional)):
            out.write(f"{imagini_testare[i][0]},{predictii[i]}\n")


rulare_validare(10, 'rbf', 'auto')
