import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from utils.datasets import TestDataset, LibraryDataset
import torch.nn.functional as F
from mmpretrain import get_model
from model.panet import PANet


class Evaluator:
    def __init__(
        self,
        backbone,
        batch_size=1,
        test_root="data/test",
        library_root="data/library",
        embedding_size=128,
    ):
        self.backbone = backbone
        self.batch_size = batch_size
        self.test_root = test_root
        self.library_root = library_root
        self.embedding_size = embedding_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.model.to(self.device)
        self.test_loader = self.create_test_loader()
        self.feature_library_loader = self.create_feature_library_loader()

    def load_model(self):
        if self.backbone == "ConvNeXt":
            model = get_model("convnext-tiny_32xb128_in1k", pretrained=False)
        elif self.backbone == "ResNet50":
            model = get_model("resnet50_8xb16_cifar10", pretrained=False)
        elif self.backbone == "Res2Net":
            model = get_model("res2net50-w14-s8_3rdparty_8xb32_in1k", pretrained=False)
        elif self.backbone == "SENet":
            model = get_model("seresnet50_8xb32_in1k", pretrained=False)
        elif self.backbone == "Shufflenetv2":
            model = get_model("shufflenet-v2-1x_16xb64_in1k", pretrained=False)
        elif self.backbone == "MobileNetv2":
            model = get_model("mobilevit-small_3rdparty_in1k", pretrained=False)
        elif self.backbone == "EfficientNet":
            model = get_model("efficientnet-b0_3rdparty_8xb32_in1k", pretrained=False)
        elif self.backbone == "HRNet":
            model = get_model("hrnet-w18_3rdparty_8xb32_in1k", pretrained=False)
        elif self.backbone == "ViT":
            model = get_model(
                "vit-base-p32_in21k-pre_3rdparty_in1k-384px", pretrained=False
            )
        elif self.backbone == "Swin":
            model = get_model("swin-tiny_16xb64_in1k", pretrained=False)
        elif self.backbone == "PANet":
            model = PANet()
        else:
            raise NotImplementedError
        model_path = f"/weights/{self.backbone}_best.pth"
        model.load_state_dict(torch.load(model_path))
        return model.to(self.device)

    def create_test_loader(self):
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        test_dataset = TestDataset(self.test_root, transform=transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def create_feature_library_loader(self):
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        feature_library_dataset = LibraryDataset(self.library_root, transform=transform)
        return DataLoader(feature_library_dataset, batch_size=1, shuffle=False)

    def evaluate_recognition_accuracy(self):
        random_seed = 2000
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        feature_library = {}
        self.model.eval()

        with torch.no_grad():
            for images, label in self.feature_library_loader:
                images = images.to(self.device)
                class_id = label.to(self.device)
                embeddings = self.model(images)
                feature_library[class_id] = embeddings

        correct = 0
        total = len(self.test_loader.dataset)

        light0_count_yes, light1_count_yes, light2_count_yes, light3_count_yes = (
            0,
            0,
            0,
            0,
        )
        light0_count_no, light1_count_no, light2_count_no, light3_count_no = 0, 0, 0, 0

        cattle0_count_yes, cattle1_count_yes, cattle2_count_yes = 0, 0, 0
        cattle0_count_no, cattle1_count_no, cattle2_count_no = 0, 0, 0

        with torch.no_grad(), tqdm(self.test_loader, desc="Testing", unit="batch") as t:
            for image_path, images, labels in t:
                parts = image_path[0].split("_")
                light_number = parts[2][5:]
                cattle_number = parts[0][-1]

                images = images.to(self.device)
                embeddings = self.model(images)

                distances = []
                for class_id, class_embeddings in feature_library.items():
                    similarity = F.cosine_similarity(
                        embeddings, class_embeddings, dim=1
                    )
                    distance = 1 - similarity
                    distances.append(distance)
                distances = torch.stack(distances, dim=1)

                predicted_classes = torch.argmin(distances, dim=1)
                feature_labels = torch.tensor(list(feature_library.keys())).to(
                    self.device
                )
                predicted_classes = feature_labels[predicted_classes]
                true_classes = labels.to(self.device)

                if predicted_classes != true_classes:
                    if light_number == "0":
                        light0_count_no += 1
                    elif light_number == "1":
                        light1_count_no += 1
                    elif light_number == "2":
                        light2_count_no += 1
                    elif light_number == "3":
                        light3_count_no += 1

                    if cattle_number == "0":
                        cattle0_count_no += 1
                    elif cattle_number == "1":
                        cattle1_count_no += 1
                    elif cattle_number == "2":
                        cattle2_count_no += 1

                if predicted_classes == true_classes:
                    if light_number == "0":
                        light0_count_yes += 1
                    elif light_number == "1":
                        light1_count_yes += 1
                    elif light_number == "2":
                        light2_count_yes += 1
                    elif light_number == "3":
                        light3_count_yes += 1

                    if cattle_number == "0":
                        cattle0_count_yes += 1
                    elif cattle_number == "1":
                        cattle1_count_yes += 1
                    elif cattle_number == "2":
                        cattle2_count_yes += 1

                correct += (predicted_classes == true_classes).sum().item()
                t.set_postfix(accuracy=(correct / total) * 100)
                t.update()

        accuracy = correct / total
        print(f"Recognition Accuracy: {accuracy * 100}%")
        print("***************Light Recognition Correct******************")
        print("Dark true:", light0_count_yes)
        print("Normal true:", light1_count_yes)
        print("Exposure true:", light2_count_yes)
        print("Nonuniform true:", light3_count_yes)
        print("***************Light Recognition Incorrect******************")
        print("Dark false:", light0_count_no)
        print("Normal false:", light1_count_no)
        print("Exposure false:", light2_count_no)
        print("Nonuniform false:", light3_count_no)
        print("**************Cattle Face Orientation Recognition Correct****************")
        print("Front true", cattle0_count_yes)
        print("Left true", cattle1_count_yes)
        print("Right true", cattle2_count_yes)
        print("**************Cattle Face Orientation Recognition Incorrect****************")
        print("Front false", cattle0_count_no)
        print("Left false", cattle1_count_no)
        print("Right false", cattle2_count_no)


if __name__ == "__main__":
    evaluator = Evaluator(backbone="PANet")
    evaluator.evaluate_recognition_accuracy()
