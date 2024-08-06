## About 
This student project (TX00 “green AI”, UTC) provides a review (2024), as well as some solutions to assess carbon emissions and reduce costs in modern AI projects, considering various (i) computer architectures, (ii) coding languages and (iii) Machine Learning (ML)/Deep Learning (DL) algorithms.

[![](https://tinyurl.com/greenai-pledge)](https://github.com/daviddao/green-ai)

## Tools & scripts
Without libraries, you can refer to Intel “Running Average Power Limit” ([RAPL](https://web.eece.maine.edu/~vweaver/projects/rapl/)) technology to track carbon emissions and compare different coding languages (here, we provide illustration scripts comparing Python vs C++ 10-layer artificial neural networks) ;

With libraries, you can use [CodeCarbon]( https://codecarbon.io/) to track carbon emissions when using [TensorFlow](https://www.tensorflow.org/) or [Scikit Learn]( https://scikit-learn.org/stable/). Following are some illustrations using scripts provided in the repository with the [MNIST]( https://docs.ultralytics.com/datasets/classify/mnist/) database:

- For ML, comparison of supervised (K-NN with 3 neighbors: emission = 9.0008262513e-06 kg CO2, accuracy = 0.945) vs unsupervised algorithms (K-means with 256 clusters: emissions = 1.3420924768e-06 kg CO2, accuracy: 0.901) ;

- For DL, comparison of ANN (10-layer perceptron: emissions =  1.4007240778e-05 kg CO2, accuracy = 0.979), CNN (7-layer: emissions = 3.7664445672e-05 kg CO2, accuracy = 0.991), RNN (GRU with 2 layers: emissions = 0.0001087401 kg CO2, accuracy = 0.987), and GAN (5-layer generator/ 5 layer discriminator : not tested) ;

- CAUTION: Transformers are still under development (in the branch 'Essais')

- Comparison of ML vs a mathematical model for image generation: K-means (emissions = 4.4933411717e-09 kg CO2, duration =  0.0261781215 sec.) vs Fourier (emissions = 1.72199680716e-09 kg CO2, duration = 0.0035228729 sec.)

## Report
A full description of the project, including literature review (until May 2024), scripts and guidelines for green AI are available in the project report (PDF, French).

## Authors
Rosalie JARDRI & Julie GUILLERMET (supervision : Pr. Marc SHAWKY)
