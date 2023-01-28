import torch
from torch import nn

import math
import matplotlib.pyplot as plt



class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()
        # My discriminator is using is an MLP neural network defined in a sequential 
        # way using nn.Sequential().
        self.model = nn.Sequential(

            nn.Linear(2, 256), # First hidden layer composed of 256 neurons
            nn.ReLU(), # with ReLU activation
            nn.Dropout(0.3), # Use dropout to avoid overfitting.
            nn.Linear(256, 128), # Second hidden layer is composed of 128 neurons
            nn.ReLU(), # with ReLU activation
            nn.Dropout(0.3), # Use dropout to avoid overfitting.
            nn.Linear(128, 64), # Third hidden layer is composed of 64 neurons
            nn.ReLU(), # with ReLU activation
            nn.Dropout(0.3), # Use dropout to avoid overfitting.
            nn.Linear(64, 1), # Output is composed of a single neuron
            nn.Sigmoid(), # with sigmoidal activation to represent a probability.

        )

    # Use .forward() to describe how the output of the model is calculated
    def forward(self, x): 

        output = self.model(x)

        return output


class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 16), # First hidden layer composed of 16 neurons
            nn.ReLU(), # with ReLU activation
            nn.Linear(16, 32), # Second hidden layer composed of 32 neurons
            nn.ReLU(), # with ReLU activation
            nn.Linear(32, 2),
            # the output will consist of a vector with two elements that can be 
            # any value ranging from negative infinity to infinity
        )


    def forward(self, x):

        output = self.model(x)

        return output

def main() :
    torch.manual_seed(111)

    train_data_length = 1024

    train_data = torch.zeros((train_data_length, 2))

    #Frist column are random values in the interval from 0 to 2π
    train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)

    #Second column of the tensor as the sine of the first column
    train_data[:, 1] = torch.sin(train_data[:, 0])

    #I will use ceros as train data labels
    train_labels = torch.zeros(train_data_length)

    #Create train_set as a list of tuples, with each row of train_data and train_labels
    #  represented in each tuple as expected by PyTorch’s data loader
    train_set = [
            (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]

    plt.plot(train_data[:, 0], train_data[:, 1], ".") #Print data 


    #Now I can create a Pytorch data loader with the tran_set
    # This loader will shuffle the data from train_set and return batches of 32 samples
    # to train the neural networks
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )


    # Generate discriminator object
    discriminator = Discriminator()

    # Generate generator object
    generator = Generator()

    # Define parameters
    lr = 0.001 # Learning rate -> I will use it to adapt the network weights
    num_epochs = 300 # Number of epochs -> defines how many repetitions of 
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
            real_samples_labels = torch.ones((batch_size, 1)) 

            # Create the generated samples by storing random data
            latent_space_samples = torch.randn((batch_size, 2)) # vector of 2 elements
            generated_samples = generator(latent_space_samples)

            # Gerenare labels with value 0 for fake samples
            generated_samples_labels = torch.zeros((batch_size, 1))

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

            latent_space_samples = torch.randn((batch_size, 2))


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


        latent_space_samples = torch.randn(100, 2)
        generated_samples = generator(latent_space_samples)
        generated_samples = generated_samples.detach()
        plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")

main()

