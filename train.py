from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.datasets import TripletDataset
from utils.lr_utils import get_params_groups, create_lr_scheduler
import random
from mmpretrain import get_model
from model import panet


class Trainer:
    def __init__(
        self,
        backbone,
        num_epochs=300,
        batch_size=16,
        lr=5e-4,
        wd=5e-2,
        shape=224,
        train_root="data/train",
        val_root="data/test",
    ):
        self.backbone = backbone
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.shape = shape
        self.train_root = train_root
        self.val_root = val_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.instance_model()
        self.model.to(self.device)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        self.train_loader, self.val_loader = self.get_data_loaders()
        self.optimizer, self.lr_scheduler = self.get_optimizer_and_scheduler()
        self.best_loss = float("inf")
        self.train_loss = []
        self.val_loss_list = []

    def set_seed(self, seed=2000):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def instance_model(self):
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
            # model = PANet()
            model = panet(1000)
        else:
            raise NotImplementedError
        return model.to(self.device)

    def get_data_loaders(self):
        train_dataset = TripletDataset(self.train_root, shape=self.shape)
        val_dataset = TripletDataset(self.val_root, shape=self.shape)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def get_optimizer_and_scheduler(self):
        pg = get_params_groups(self.model, weight_decay=self.wd)
        optimizer = optim.AdamW(pg, lr=self.lr, weight_decay=self.wd)
        lr_scheduler = create_lr_scheduler(
            optimizer,
            len(self.train_loader),
            self.num_epochs,
            warmup=True,
            warmup_epochs=1,
        )
        return optimizer, lr_scheduler

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total = 0

        with tqdm(self.train_loader, desc="Training", unit="batch") as t:
            for batch in t:
                anchor_images, positive_images, negative_images = batch

                anchor_images, positive_images, negative_images = (
                    anchor_images.to(self.device),
                    positive_images.to(self.device),
                    negative_images.to(self.device),
                )

                self.optimizer.zero_grad()

                anchor_embeddings = self.model(anchor_images)
                positive_embeddings = self.model(positive_images)
                negative_embeddings = self.model(negative_images)

                loss = self.triplet_loss(
                    anchor_embeddings, positive_embeddings, negative_embeddings
                )
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                t.set_postfix(loss=loss.item())
                t.update()

                total += anchor_images.size(0)

        average_loss = total_loss / len(self.train_loader)
        return average_loss

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation", unit="batch") as t:
                for val_batch in t:
                    (
                        val_anchor_images,
                        val_positive_images,
                        val_negative_images,
                    ) = val_batch

                    val_anchor_images, val_positive_images, val_negative_images = (
                        val_anchor_images.to(self.device),
                        val_positive_images.to(self.device),
                        val_negative_images.to(self.device),
                    )

                    val_anchor_embeddings = self.model(val_anchor_images)
                    val_positive_embeddings = self.model(val_positive_images)
                    val_negative_embeddings = self.model(val_negative_images)

                    val_batch_loss = self.triplet_loss(
                        val_anchor_embeddings,
                        val_positive_embeddings,
                        val_negative_embeddings,
                    )
                    val_loss += val_batch_loss.item()

        val_loss /= len(self.val_loader)
        return val_loss

    def run(self):
        self.set_seed()

        for epoch in range(self.num_epochs):
            average_loss = self.train_epoch()
            self.train_loss.append(average_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {average_loss}")

            val_loss = self.evaluate()
            self.val_loss_list.append(val_loss)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Val Loss: {val_loss}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(
                    self.model.state_dict(), f"weights/{self.backbone}_best.pth"
                )

        torch.save(
            self.model.state_dict(),
            f"weights/{self.backbone}_last.pth",
        )

        plt.plot(range(1, self.num_epochs + 1), self.train_loss, label="Train Loss")
        plt.plot(
            range(1, self.num_epochs + 1), self.val_loss_list, label="Validation Loss"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    trainer = Trainer(backbone="PANet")
    trainer.run()
