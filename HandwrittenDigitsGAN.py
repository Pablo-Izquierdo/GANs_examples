import torch
from torch import nn

import math
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()
        # My discriminator is using is an MLP neural network defined in a sequential 
        # way using nn.Sequential().
        self.model = nn.Sequential(

            nn.Linear(784, 1024), # First hidden layer composed of 1024 neurons
            nn.ReLU(), # with ReLU activation
            nn.Dropout(0.3), # Use dropout to avoid overfitting.
            nn.Linear(1024, 512), # Second hidden layer is composed of 512 neurons
            nn.ReLU(), # with ReLU activation
            nn.Dropout(0.3), # Use dropout to avoid overfitting.
            nn.Linear(512, 256), # Third hidden layer is composed of 256 neurons
            nn.ReLU(), # with ReLU activation
            nn.Dropout(0.3), # Use dropout to avoid overfitting.
            nn.Linear(256, 1), # Output is composed of a single neuron
            nn.Sigmoid(), # with sigmoidal activation to represent a probability.

        )

    # Use .forward() to describe how the output of the model is calculated
    def forward(self, x): 
        x = x.view(x.size(0), 784)
        output = self.model(x)

        return output


class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 256), # First hidden layer composed of 256 neurons
            nn.ReLU(), # with ReLU activation
            nn.Linear(256, 512), # Second hidden layer composed of 512 neurons
            nn.ReLU(), # with ReLU activation
            nn.Linear(512, 1024), # Third hidden layer composed of 1024 neurons
            nn.ReLU(), # with ReLU activation
            nn.Linear(1024, 784), # Output is composed of 784 neurons
            nn.Tanh(),
        )


    def forward(self, x):

        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)

        return output



def main() :

    torch.manual_seed(111)

    # To reduce the training time, you can use a GPU
    # Check if gpu acceleration can be used
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define transform, a function to be used when loading the data
    transform = transforms.Compose(
    [transforms.ToTensor(), # converts the data to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))]) # converts the range of the tensor coefficients
    
    # Load the data of HandwrittenDigits
    train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform)

    # Creat a data loader
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    # Plot tsome samples of the train data
    real_samples, mnist_labels = next(iter(train_loader))
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])

    # Generate objects
    discriminator = Discriminator().to(device=device)
    generator = Generator().to(device=device)


    lr = 0.0001 # Learning rate -> I will use it to adapt the network weights
    num_epochs = 50 # Number of epochs -> defines how many repetitions of 
                    # training using the whole training set
    # The loss function i will use to train the models is the binary cross-entropy function
    # because this problems is consider as a binary classication task
    loss_function = nn.BCELoss()

    # I will use Adam algorithm to train the discriminatos and generatos models 
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)


    for epoch in range(num_epochs): # Training epochs

        # get the real samples of the current batch from the data loader and assign them to real_samples
        for n, (real_samples, _) in enumerate(train_loader): 

            # Data for training the discriminator
            # Gerenare labels with value 1 for real samples
            real_samples_labels = torch.ones((batch_size, 1)).to(device=device)

            # Create the generated samples by storing random data
            latent_space_samples = torch.randn((batch_size, 2)).to(device=device) # vector of 2 elements
            generated_samples = generator(latent_space_samples)

            # Gerenare labels with value 0 for fake samples
            generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)

            # Concatenate the real and generated samples and labels
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )


            ########## Training the discriminator##########
            # it’s necessary to clear the gradients at each training step to 
            # avoid accumulating them
            discriminator.zero_grad()

            # Calculate the output of the discriminator using the training data
            output_discriminator = discriminator(all_samples)

            # Calculate the loss function using the output from the model and labels
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)

            loss_discriminator.backward() # calculate the gradients to update the weights

            optimizer_discriminator.step() # update the discriminator weights


            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 2)).to(device=device)


            ######### Training the generator##########
            # it’s necessary to clear the gradients at each training step to 
            # avoid accumulating them
            generator.zero_grad()

            # Feed the generator with latent_space_samples
            generated_samples = generator(latent_space_samples)

            # Feed the generator’s output into the discriminator
            output_discriminator_generated = discriminator(generated_samples)

            # Calculate the loss function using the output of the classification system 
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )

            # Calculate the gradients and update the generator weights
            loss_generator.backward()
            optimizer_generator.step()


            # Show loss

            if epoch % 10 == 0 and n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")

    # generate handwritten digits
    latent_space_samples = torch.randn(batch_size, 100).to(device=device)
    generated_samples = generator(latent_space_samples)

    # Plot generated_samples, move the data back to the CPU is needed
    generated_samples = generated_samples.cpu().detach()
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])



main()