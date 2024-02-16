from utils import get_result, load_checkpoint
from loader import get_loader
import torchvision.transforms as transforms
import torch
from model import CNNtoRNN
import torch.optim as optim
from torchsummary import summary

checkpoint_path = "models(unsuccessful)/13.02_16-12/resulting.pth.tar"

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

device = torch.device("cuda")

embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
load_checkpoint(torch.load(checkpoint_path), model, optimizer)

# summary(model=model, input_size=(3, 32))
model.eval()

images = [
    "1000268201_693b08cb0e",
    "1001773457_577c3a7d70",
]
images = [f"flickr8k/images/{im}.jpg" for im in images]
correct_captions = [
    "A child in a pink dress is climbing up a set of stairs in an entry way .",
    "black dog and a spotted dog are fighting",
]

for ind, (image, correct) in enumerate(zip(images, correct_captions)):
        predicted = get_result(model, image, dataset, device)
        print(ind)
        print(f"correct = {correct}")
        print(f"predicted = {predicted}")