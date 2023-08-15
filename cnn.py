import time
import numpy as np
import torch
from sklearn import metrics
from PIL import Image
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# In functie de nevoie si disponibilitate
device = "cuda"
# device = "cpu"


def load_img(folder, nume):
    img = Image.open(f"{folder}/{nume}")
    return np.array(img)


def load_data(tip_data):
    date = np.genfromtxt(f"../{tip_data}.csv", delimiter=',', dtype=None, names=True, encoding='utf8')
    nume_imagini = date['Image']

    imagini = np.array([load_img(f"../{tip_data}_images", image_name) for image_name in nume_imagini])

    imagini = imagini.transpose((0, 3, 1, 2))  # Reordonare astfel incat culorile sa fie ultimele
    imagini = imagini / 255.0  # Normalizarea pixelilor

    imagini = torch.tensor(imagini, dtype=torch.float32)

    if tip_data == "test":
        return nume_imagini, imagini

    labeluri = date['Class']

    labeluri = labeluri.astype(int)
    labeluri = torch.tensor(labeluri, dtype=torch.long)

    return imagini, labeluri


class DatasetCuTransforms(Dataset):
    def __init__(self, data, labeluri, transformari=None):
        self.data = data
        self.labeluri = labeluri

        if transformari is None:
            self.transformari = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transformari = transformari

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        labeluri = self.labeluri[index]

        return self.transformari(data), labeluri


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Din comoditate, pentru a scrie mai rapid
        convolutieComuna = lambda in_chs, out_chs: nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=1)
        maxpool = lambda: nn.MaxPool2d(2)
        bn2d = lambda channels: nn.BatchNorm2d(channels)
        bn1d = lambda channels: nn.BatchNorm1d(channels)

        self.conv1 = convolutieComuna(3, 32)
        self.bn1 = bn2d(32)

        self.conv2 = convolutieComuna(32, 64)
        self.bn2 = bn2d(64)
        self.maxpool2 = maxpool()

        self.conv3 = convolutieComuna(64, 64)
        self.bn3 = bn2d(64)

        self.conv4 = convolutieComuna(64, 64)
        self.bn4 = bn2d(64)
        self.maxpool4 = maxpool()

        self.conv5 = convolutieComuna(64, 64)
        self.bn5 = bn2d(64)

        self.conv6 = convolutieComuna(64, 128)
        self.bn6 = bn2d(128)
        self.maxpool6 = maxpool()

        self.conv7 = convolutieComuna(128, 128)
        self.bn7 = bn2d(128)

        self.conv8 = convolutieComuna(128, 256)
        self.bn8 = bn2d(256)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=1)
        self.bn9 = bn2d(512)
        self.maxpool9 = maxpool()

        self.flatten = nn.Flatten()

        self.fc0 = nn.Linear(3 * 3 * 512, 512)
        self.bnF0 = bn1d(512)

        self.fc1 = nn.Linear(512, 64)
        self.bnF1 = bn1d(64)

        self.classifier = nn.Linear(64, 96)

    def forward(self, x):
        activation = nn.PReLU(device=device)
        #Din comoditate, pentru a scrie mai rapid
        pas_convolutie_comuna = lambda x_, conv, bn: activation(bn(conv(x_)))
        pas_convolutie_comuna_maxpool = lambda x_, conv, bn, maxpool: activation(maxpool(bn(conv(x_))))

        x = self.conv1(x)
        skip_x = x
        x = self.bn1(x)
        x = activation(x)

        x = pas_convolutie_comuna_maxpool(x, self.conv2, self.bn2, self.maxpool2)

        x = pas_convolutie_comuna(x, self.conv3, self.bn3)

        x = pas_convolutie_comuna_maxpool(x, self.conv4, self.bn4, self.maxpool4)

        x = pas_convolutie_comuna(x, self.conv5, self.bn5)

        x = pas_convolutie_comuna_maxpool(x, self.conv6, self.bn6, self.maxpool6)

        x = pas_convolutie_comuna(x, self.conv7, self.bn7)

        x = self.conv8(x)

        skip_x = nn.AdaptiveAvgPool2d((x.size(2) + 2, x.size(3) + 2))(skip_x)
        skip_x = nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=0, device=device)(skip_x)
        x += skip_x

        x = self.bn8(x)
        x = activation(x)

        x = pas_convolutie_comuna_maxpool(x, self.conv9, self.bn9, self.maxpool9)

        x = self.flatten(x)

        x = self.fc0(x)
        x = self.bnF0(x)
        x = activation(x)

        x = self.fc1(x)
        x = self.bnF1(x)
        x = activation(x)

        x = self.classifier(x)
        return x


def training(train_images, train_labels, verify_images, verify_labels):
    batch_size = 64
    n_epoci = 250
    date_antrenare = DatasetCuTransforms(train_images, train_labels,
                                         transformari=transforms.Compose([
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),
                                             transforms.RandomAffine(degrees=15, scale=(0.75, 1.25), shear=0.25,
                                                                     translate=(0.25, 0.25)),
                                             transforms.RandomErasing(),
                                         ]))

    data_loader = DataLoader(date_antrenare, batch_size=batch_size, shuffle=True)

    cnn_model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=0.03)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoci, eta_min=0.000013)

    # Pentru early stopping
    best_valid_acuratete = 0
    fara_imbunatatiri = 0
    patience = 30

    for index_epoca in range(1, n_epoci + 1):
        loss_antrenare = 0.0
        corecte = 0
        total = 0

        cnn_model.train()

        for imagini, labeluri_adevarate in data_loader:
            imagini = imagini.to(device)
            labeluri_adevarate = labeluri_adevarate.to(device)

            optimizer.zero_grad()

            returnat = cnn_model(imagini)
            loss = criterion(returnat, labeluri_adevarate)
            loss.backward()
            optimizer.step()

            loss_antrenare += loss.item() * imagini.size(0)

            _, predictie = torch.max(returnat.data, 1)
            total += labeluri_adevarate.size(0)
            corecte += (predictie == labeluri_adevarate).sum().item()

        scheduler.step()

        loss_antrenare /= len(data_loader.dataset)
        acuratete_antrenare = corecte / total
        acuratete_valid = predict_accuracy(cnn_model, verify_images, verify_labels)

        print(
            f"Epoca {index_epoca}/{n_epoci}, Loss Antrenare: {loss_antrenare:.4f}, Acuratete Antrenare: {acuratete_antrenare:.4f}, Acuratete Validare {acuratete_valid:.4f} Learning Rate: {scheduler.get_last_lr()}")

        if acuratete_valid > best_valid_acuratete:
            best_valid_acuratete = acuratete_valid
            fara_imbunatatiri = 0
            torch.save(cnn_model.state_dict(), "../best_models/best_model.pt")
        else:
            fara_imbunatatiri += 1
            if fara_imbunatatiri >= patience:
                print("Early stopping.")
                break


def load_best_model():
    best_model = CNN().to(device)
    best_model.load_state_dict(torch.load("../best_models/best_model.pt"))
    return best_model


def predict_accuracy(model, imagini_de_verificat, labeluri_de_verificat):
    model.eval()
    imagini_de_verificat = imagini_de_verificat.to(device)
    with torch.no_grad():  # no_grad dezactiveaza computatii extra care nu sunt necesare pentru evaluare
        evaluare = model(imagini_de_verificat)
        _, predictii = torch.max(evaluare.data, 1)

    acuratete = metrics.accuracy_score(labeluri_de_verificat, predictii.cpu())
    return acuratete


def statistici(labeluri_adevarate, predictii):
    accuracy = metrics.accuracy_score(labeluri_adevarate, predictii)
    print(f"Acuratete: {accuracy}")

    print(f"Recall (micro): {metrics.recall_score(labeluri_adevarate, predictii, average='micro')}")
    print(f"Recall (macro): {metrics.recall_score(labeluri_adevarate, predictii, average='macro')}")
    print(f"Recall (weighted): {metrics.recall_score(labeluri_adevarate, predictii, average='weighted')}")
    print(f"F1_score (micro): {metrics.f1_score(labeluri_adevarate, predictii, average='micro')}")
    print(f"F1_score (macro): {metrics.f1_score(labeluri_adevarate, predictii, average='macro')}")
    print(f"F1_score (weighted): {metrics.f1_score(labeluri_adevarate, predictii, average='weighted')}")

    matrice_de_confuzie = metrics.confusion_matrix(labeluri_adevarate, predictii)
    np.set_printoptions(threshold=np.inf)
    print("Matrice de confuzie:")
    for row in matrice_de_confuzie:
        print(*row)
    print()
    print(
        metrics.classification_report(labeluri_adevarate, predictii, target_names=[f'Clasa {i}' for i in range(96)],
                                      zero_division=0))


def submit(predictii, nume_imagini):
    with open("submission.csv", 'w') as submisie:
        submisie.write("Image,Class\n")
        for nume_img, predictie in enumerate(nume_imagini, predictii):
            submisie.write(f"{nume_img},{predictie}\n")


def cnn(este_submit=False):
    antrenare_imagini, antrenare_labeluri_adevarate = load_data("train")

    if not este_submit:
        val_imagini, val_labeluri_adevarate = load_data("val")

    start = time.time()

    if not este_submit:
        training(antrenare_imagini, antrenare_labeluri_adevarate, val_imagini, val_labeluri_adevarate)
    cnn_model = load_best_model()

    terminat = time.time()

    # Nenecesar, dar orientativ pentru mine
    timp_executie = terminat - start
    print(f"Timp de executie: {timp_executie} secunde")

    cnn_model.eval()

    # Tratez diferit cazurile in care vreau sa generez predictii pt submission.csv
    # sau vreau doar sa verific pe validare
    if este_submit:
        nume_imagini, test_imagini = load_data('test')
        test_imagini = test_imagini.to(device)
        with torch.no_grad():  # no_grad dezactiveaza computatii extra care nu sunt necesare pentru evaluare
            evaluare = cnn_model(test_imagini)
            _, predictii = torch.max(evaluare.data, 1)

        submit(predictii.cpu(), nume_imagini)
    else:
        val_imagini = val_imagini.to(device)
        with torch.no_grad():  # no_grad dezactiveaza computatii extra care nu sunt necesare pentru evaluare
            evaluare = cnn_model(val_imagini)
            _, predictii = torch.max(evaluare.data, 1)

        statistici(val_labeluri_adevarate, predictii.cpu())


cnn()
