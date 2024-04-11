# Distributed MRI image generation

This repository contains the code for the generation of MRI images of the brain in a distributed environment. We re-implement the model used in our previous work [1](https://ieeexplore.ieee.org/document/10175330), and also re-implement a WGAN-GP [2](https://proceedings.neurips.cc/paper_files/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html) to generate  MRI images of the brain in a larger distributed setting. Both models were implemented in TensorFlow and resorted to the *MultiWorkerMirroredStrategy* module.
