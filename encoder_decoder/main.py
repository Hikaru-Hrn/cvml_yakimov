import matplotlib.pyplot as plt
import torch
from dec_enc import Decoder, Encoder, ImageDataset

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("decoder.pth"))
encoder.eval()
decoder.eval()

dataset = ImageDataset(10, 256, 4)
image, _ = dataset[0]
with torch.no_grad():
    latent = encoder(image.unsqueeze(0))
    result = decoder(latent)

    plt.subplot(131)
    plt.imshow(image.squeeze().cpu().numpy())
    plt.subplot(132)
    plt.imshow(result.squeeze().cpu().detach().numpy())
    plt.subplot(133)
    plt.imshow(image.squeeze() - result.squeeze())
    plt.show()
