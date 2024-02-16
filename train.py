import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from loader import get_loader
from model import CNNtoRNN
from tqdm.auto import tqdm
from datetime import datetime
import os

model_dir = None

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    load_model = False
    save_model = True

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    writer =  SummaryWriter(os.path.join(model_dir, "runs/flickr"))
    step = 0

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        checkpoint_path = "my_checkpoint.pth.tar"
        step = load_checkpoint(torch.load(checkpoint_path), model, optimizer)

    model.train()

    # for epoch in tqdm(range(num_epochs), desc="epoch"):
    for index, epoch in enumerate(range(num_epochs)):
        print(f"EPOCH: {index}")
        print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, os.path.join(model_dir, "checkpoint.pth.tar"))

        for idx, (imgs, captions) in enumerate(tqdm(train_loader, desc="training on examples")):
            # print(f"\n{idx}")
            imgs = imgs.to(device)
            captions = captions.to(device)
            # print(type(captions[:-1]))
            # print(imgs)
            # print(captions[:-1])
            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    save_checkpoint(checkpoint, os.path.join(model_dir, "resulting.pth.tar"))


if __name__ == '__main__':
    model_dir = os.path.join("models(unsuccessful)", f"{datetime.now().strftime('%d.%m_%H-%M')}")
    print(f"model_dir = {model_dir}")
    train()