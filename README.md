# VSDSquadron-Mini-Internship
---

Details 
## Task 1: 
---
### Step 1: Create the Program file 
Created a file named `sum1ton.c` in the home folder, typed a C code to compute the sum of numbers from 1 to n using a text editor, and saved it.
    ![c_code](https://github.com/user-attachments/assets/bfd3de06-efef-4f54-ac66-f96ebd0ed2ba)

### Step 2: Compile the C code using GCC compiler
The code was compiled using GCC Compiler and the output was printed on the terminal. 
  ```
  $ gcc sum1ton.c
  $ ./a.out
  ```
   ![c_output](https://github.com/user-attachments/assets/3320fbf7-f0c2-4508-b675-3b98b8653a20)

### Step 3: Compile the C code using the RISC-V compiler (-O1) 
The program was compiled with the `riscv64-unknown-elf-gcc` compiler using the `-O1` optimization level.
  ```
  $ riscv64-unknown-elf-gcc -O1 -mabi=lp64 -march=rv64i -o sum1ton.o sum1ton.c
  $ ls -ltr sum1ton.o
  ```
   ![sum1ton o](https://github.com/user-attachments/assets/dd69cb0b-974f-4fb6-9c33-a559b6942f88)

### Step 4: Inspect the Assembly Code for the Main Function (-O1)
Opened a new terminal and entered the below command. The disassembled assembly code for the main function was inspected to observe how the compiler optimized the program with the `-O1` optimization level.
   ```
   $ riscv64-unknown-elf-objdump -d sum1ton.o | less
   ```
   ![main](https://github.com/user-attachments/assets/4319d61e-678e-467c-a5ad-de6c4dce23ad)

### Step 5: Compile the C code using the RISC-V compiler (-Ofast) 
Gone back to the old terminal and again compiled the code  using the `riscv64-unknown-elf-gcc` compiler with the `-Ofast` optimization level to enable aggressive optimizations for performance.
  ```
  $ riscv64-unknown-elf-gcc -Ofast -mabi=lp64 -march=rv64i -o sum1ton.o sum1ton.c
  ```
  ![ofast](https://github.com/user-attachments/assets/4c9d90d4-0593-4c65-8364-8fc473597969)

### Step 6: Inspect the Assembly Code for the Main Function (-Ofast)
Again gone to the the new terminal and entered the below command.The disassembled assembly code for the main function was inspected again to analyze the effects of the `-Ofast` optimization level.
  ```
  $ riscv64-unknown-elf-objdump -d sum1ton.o | less
  ```
  ![main2](https://github.com/user-attachments/assets/5d08ed72-ee14-458f-8a51-3c0a881a4369)

---
## Task 2

### Branch Prediction Using a Neural Network

#### Introduction

Branch prediction is a critical component of modern CPUs to improve instruction pipeline efficiency. By predicting the outcome of a branch instruction (taken or not taken), processors can minimize delays. This project demonstrates a neural network-based approach for branch prediction, where a simple feedforward neural network is trained to predict branch behavior based on historical patterns.

---

#### Data Set

The table below represents the training data used for the neural network. Each row corresponds to a historical branch pattern (input) and the associated branch outcome (target).

| **Input (Branch History)** | **Target (Branch Outcome)** |
|-----------------------------|-----------------------------|
| `{1, 0, 1, 1, 0}`           | `1` (Taken)                |
| `{0, 1, 1, 0, 1}`           | `0` (Not Taken)            |
| `{1, 1, 0, 1, 0}`           | `1` (Taken)                |
| `{0, 0, 0, 1, 1}`           | `0` (Not Taken)            |

---

#### Specification

- **Neural Network Architecture**: 
  - Input layer: 5 neurons (representing 5 historical branch outcomes).
  - Hidden layer: 3 neurons (with activation function).
  - Output layer: 1 neuron (producing a value between 0 and 1).
  
- **Objective**: Predict whether a branch will be taken (output near 1) or not taken (output near 0).

---

#### Implementation

Below is the implementation of the neural network-based branch predictor in C:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 5
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1
#define EPOCHS 10000
#define LEARNING_RATE 0.1

// Activation function (Sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of Sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize weights and biases randomly
void initialize(double weights_in[HIDDEN_SIZE][INPUT_SIZE], double weights_out[OUTPUT_SIZE][HIDDEN_SIZE], 
                double bias_hidden[HIDDEN_SIZE], double bias_output[OUTPUT_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights_in[i][j] = (double)rand() / RAND_MAX;
        }
        bias_hidden[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_out[i][j] = (double)rand() / RAND_MAX;
        }
        bias_output[i] = (double)rand() / RAND_MAX;
    }
}

// Forward pass
void forward(double input[INPUT_SIZE], double weights_in[HIDDEN_SIZE][INPUT_SIZE], double bias_hidden[HIDDEN_SIZE],
             double hidden[HIDDEN_SIZE], double weights_out[OUTPUT_SIZE][HIDDEN_SIZE], 
             double bias_output[OUTPUT_SIZE], double output[OUTPUT_SIZE]) {
    // Hidden layer computation
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = bias_hidden[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * weights_in[i][j];
        }
        hidden[i] = sigmoid(hidden[i]);
    }

    // Output layer computation
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = bias_output[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hidden[j] * weights_out[i][j];
        }
        output[i] = sigmoid(output[i]);
    }
}

// Backward pass (Training)
void backward(double input[INPUT_SIZE], double weights_in[HIDDEN_SIZE][INPUT_SIZE], double bias_hidden[HIDDEN_SIZE],
              double hidden[HIDDEN_SIZE], double weights_out[OUTPUT_SIZE][HIDDEN_SIZE], double bias_output[OUTPUT_SIZE], 
              double output[OUTPUT_SIZE], double target[OUTPUT_SIZE]) {
    double output_error[OUTPUT_SIZE], hidden_error[HIDDEN_SIZE];

    // Calculate output error
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
    }

    // Calculate hidden layer error
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_error[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error[i] += output_error[j] * weights_out[j][i];
        }
        hidden_error[i] *= sigmoid_derivative(hidden[i]);
    }

    // Update weights and biases (Output layer)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_out[i][j] += LEARNING_RATE * output_error[i] * hidden[j];
        }
        bias_output[i] += LEARNING_RATE * output_error[i];
    }

    // Update weights and biases (Input to Hidden layer)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights_in[i][j] += LEARNING_RATE * hidden_error[i] * input[j];
        }
        bias_hidden[i] += LEARNING_RATE * hidden_error[i];
    }
}

// Main function
int main() {
    // Define network parameters
    double weights_in[HIDDEN_SIZE][INPUT_SIZE], weights_out[OUTPUT_SIZE][HIDDEN_SIZE];
    double bias_hidden[HIDDEN_SIZE], bias_output[OUTPUT_SIZE];
    double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];

    // Initialize network
    initialize(weights_in, weights_out, bias_hidden, bias_output);

    // Define training data
    double inputs[4][INPUT_SIZE] = {
        {1, 0, 1, 1, 0},  // Example historical branch patterns
        {0, 1, 1, 0, 1},
        {1, 1, 0, 1, 0},
        {0, 0, 0, 1, 1}
    };
    double targets[4][OUTPUT_SIZE] = {
        {1},  // Branch taken
        {0},  // Branch not taken
        {1},  // Branch taken
        {0}   // Branch not taken
    };

    // Train the network
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < 4; i++) {
            forward(inputs[i], weights_in, bias_hidden, hidden, weights_out, bias_output, output);
            backward(inputs[i], weights_in, bias_hidden, hidden, weights_out, bias_output, output, targets[i]);
        }
    }

    // Test the network
    double test_input[INPUT_SIZE] = {0, 1, 0, 1, 1};
    forward(test_input, weights_in, bias_hidden, hidden, weights_out, bias_output, output);
    printf("Prediction for input {0, 1, 0, 1, 1}: %.2f\n", output[0]);

    return 0;
}

---

#### Compile the C code using GCC compiler
![task2pic1](https://github.com/user-attachments/assets/1f26f056-573d-4877-a119-471febcb0ef4)
 ![c_code](https://github.com/user-attachments/assets/bfd3de06-efef-4f54-ac66-f96ebd0ed2ba)
#### Compile the C code using the RISC-V compiler (-O1)
![task2pic2](https://github.com/user-attachments/assets/b8e4daab-69de-487c-8b65-913ae3ffc714)
 ![c_code](https://github.com/user-attachments/assets/bfd3de06-efef-4f54-ac66-f96ebd0ed2ba)
#### Inspect the Assembly Code for the Main Function (-O1)
![task2pic3](https://github.com/user-attachments/assets/cebb4cb4-3df2-4a18-a0ae-8205af3e54e7)

#### Compile the C code using the RISC-V compiler (-Ofast) 
![task2pic4](https://github.com/user-attachments/assets/c109c6a8-090e-496d-addc-a9e07c576782)

#### Inspect the Assembly Code for the Main Function (-Ofast)
![task2pic5](https://github.com/user-attachments/assets/29c28935-0609-4f03-90e6-3bd93caf45e5)

#### Inspect the Assembly Code for the Main Function (-Ofast)
![task2pic6](https://github.com/user-attachments/assets/21c1d3ff-504c-45c6-9073-8e6f27330cbf)


  
