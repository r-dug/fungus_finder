# neural net for image classification of fungi
## Network Architecture
mobilenetV3Large

## Dataset
not listed in repo.
link to dataset info: https://github.com/visipedia/inat_comp/tree/master/2021
Currently, I simply split the dataset into training and validation data using tensorflows built in method for this. Upon writing this fact, I'm realizing... I may not want to do that, since examples are fairly limited for almost all species.
I use the built in tensorflow function for assigning child directory names to class names as well. I'ts actually a really tidy operation for how I am iterating through different taxanomic ranks to see how classification goes in them, because the immediate child directories of the selected dataset directory are the only ones which are used. I really should build all of this into code... For example, on writing, I am entering the number of classes manually into the code as a the variable. Specifically, this is done by adjusting the final dense softmax output layer.

## Development Environment
This has been a surprisingly important consideration in this project.
I started off on windows, but learned quickly that there was really no way to utilize my CUDA compatible GPU to train a model with tensorflow on Windows. On writing, there simply isn't support for that. I even tried WSL, through which I could use CUDA, but when the dataset grew large, creating a dataset from a directory was virtually intractible. Consequently, I needed to set up a linux environment on my machine for this project. So, I backed up important files on my SDD, and created a disc partition on which to boot Ubuntu. It was relatively painless to transition, and I generally don't use windows anymore. I was getting incredibly frustrated with OneDrive anyways. I actually even flashed my laptop with Ubuntu. Full convert pretty much.

## Status: very incomplete
Right now, I am simply trying to train a neural network that is:
1) Highly accurate in classifying pictures of mushrooms.
2) Small enough in terms of parameters to use in a mobile application

Once I have a suitable model, I'm to build a platform agnostic mobile application to wrap the model as a User Interface. I haven't really done much investigation into this, but am leaning towards using React Native. I have experience working with React, and there is javascript support for keras models. 
