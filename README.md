# Neural Network (CUHK CSCI3230)

# Instruction
Run preprocessor:
```
./preprocessor.sh <raw-training.data> <dataset.dat> <answer.dat>
```
* raw-training.dat: the raw dataset
* dataset.dat: preprocessed data, the inputs of the neural network
* answer.dat: class label for the preprocessed data, also the inputs of the neural network

Run trainer:
```
./trainer.sh <dataset.dat> <answer.dat> <best.nn> <max_iteration> <small_sample_mode>
```
* dataset.dat: preprocessed data generated by preprocessor.sh
* answer.dat: class label for the preprocessed data, also generated by preprocessor.sh
* best.nn is the neural network your program trained as output
* max_iteration: maximum number of iterations for the learning process
* small_sample_mode: flag indicating the mode of training

Format of Best.nn:
```
<classification accuracy>
<number of hidden layer>
I <number of input> H <number of hidden neurons in next layer>
<bias for 1st neuron in next layer> <weight for 1st input> <weight for 2nd input> ... <weight for nth input>
<bias for 2nd neuron in next layer> <weight for 1st input> ...
.
.
.
<bias for nth neuron in next layer> ...
H <number of neurons in previous layer> H <number of hidden neurons in next layer>
<bias for 1st neuron in next layer> <weight for 1st neuron in previous layer> <2nd> ... <nth>
.
.
.
<bias for nth> ...
H <number of neurons in last -1 layer> O <number of output>
<bias for 1st output> <weight for 1st neuron in previous layer> <2nd> ... <nth>
.
.
.
<bias for nth output> ...
```
