{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "17xuIa0uJI9tmFesw0jkK1nBcnfbvHWrd",
      "authorship_tag": "ABX9TyNSSnXCEKabnEnY+WmVJpQV",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/nooralotibi/classifier/blob/main/predict.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bEaBFQaSu2P2",
        "outputId": "4e1c9f61-714d-4d89-b2ca-3653bef3e96c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Predicted Class: dew\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "from PIL import Image\n",
        "from torchvision import datasets, transforms\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torch.nn.functional as F\n",
        "import matplotlib.pyplot as plt\n",
        "from torch.utils.data import DataLoader, random_split\n",
        "import os\n",
        "\n",
        "\n",
        "transform = transforms.Compose([\n",
        "    transforms.Resize((268, 400)),\n",
        "    transforms.RandomRotation(10),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.RandomVerticalFlip(),\n",
        "    transforms.RandomCrop((268, 400)),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\n",
        "])\n",
        "\n",
        "\n",
        "dataset_root = \"/content/drive/MyDrive/dataset (1)\"\n",
        "\n",
        "\n",
        "dataset = datasets.ImageFolder(root=dataset_root, transform=transform)\n",
        "class_names = dataset.classes\n",
        "\n",
        "class CNNClassifier(nn.Module):\n",
        "    def __init__(self):\n",
        "        super(CNNClassifier, self).__init__()\n",
        "        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)\n",
        "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)\n",
        "        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)\n",
        "        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)\n",
        "        self.fc1_input_size = self.calculate_fc1_input_size()\n",
        "        self.fc1 = nn.Linear(self.fc1_input_size, 128)\n",
        "        self.fc2 = nn.Linear(128, 256)\n",
        "        self.fc3 = nn.Linear(256, 128)\n",
        "        self.fc4 = nn.Linear(128, 11)\n",
        "\n",
        "    def calculate_fc1_input_size(self):\n",
        "        with torch.no_grad():\n",
        "            x = torch.zeros(1, 3, 268, 400)\n",
        "            x = self.conv1(x)\n",
        "            x = F.relu(x)\n",
        "            x = self.pool(x)\n",
        "            x = self.conv2(x)\n",
        "            x = F.relu(x)\n",
        "            x = self.pool(x)\n",
        "            x = self.conv3(x)\n",
        "            x = F.relu(x)\n",
        "            x = self.pool(x)\n",
        "            return x.flatten().shape[0]\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.conv1(x)\n",
        "        x = F.relu(x)\n",
        "        x = self.pool(x)\n",
        "        x = self.conv2(x)\n",
        "        x = F.relu(x)\n",
        "        x = self.pool(x)\n",
        "        x = self.conv3(x)\n",
        "        x = F.relu(x)\n",
        "        x = self.pool(x)\n",
        "        x = x.view(x.size(0), -1)\n",
        "\n",
        "        x = self.fc1(x)\n",
        "        x = F.relu(x)\n",
        "        x = self.fc2(x)\n",
        "        x = F.relu(x)\n",
        "        x = self.fc3(x)\n",
        "        x = F.relu(x)\n",
        "        x = self.fc4(x)\n",
        "        return x\n",
        "\n",
        "\n",
        "model = CNNClassifier()\n",
        "\n",
        "#checkpoint_path = \"/content/drive/MyDrive/dataset (1)/checkpoint.pth\"\n",
        "#model.load_state_dict(torch.load(checkpoint_path))\n",
        "checkpoint_path = '/content/drive/MyDrive/dataset (1)/checkpoint.tar'\n",
        "\n",
        "model.eval()\n",
        "\n",
        "\n",
        "\n",
        "def predict(image):\n",
        "    model.eval()\n",
        "    transform = transforms.Compose([\n",
        "        transforms.Resize((268, 400)),\n",
        "        transforms.ToTensor(),\n",
        "        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\n",
        "    ])\n",
        "    image = transform(image).unsqueeze(0)\n",
        "    with torch.no_grad():\n",
        "        output = model(image)\n",
        "        _, predicted = torch.max(output, 1)\n",
        "        predicted_label = class_names[predicted.item()]\n",
        "    return predicted_label\n",
        "\n",
        "image = Image.open('/content/drive/MyDrive/dataset (1)/dew/2208.jpg')\n",
        "predicted_class = predict(image)\n",
        "print(\"Predicted Class:\", predicted_class)"
      ]
    }
  ]
}