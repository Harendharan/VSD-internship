# VSDSquadron-Mini-Internship
---

Details 
## Task 1

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

#### implementation

Below is the implementation of the neural network-based branch predictor in C:

```c
#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 5
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1
#define EPOCHS 10000
#define LEARNING_RATE 0.1

// Custom approximation of the exponential function
double custom_exp(double x) {
    double result = 1.0;
    double term = 1.0;
    for (int i = 1; i < 20; i++) {  // Use 20 terms for better approximation
        term *= x / i;
        result += term;
    }
    return result;
}

// Activation function (Sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + custom_exp(-x));
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
    //double test_input[INPUT_SIZE] ={1, 0, 1, 1, 0};
    forward(test_input, weights_in, bias_hidden, hidden, weights_out, bias_output, output);
    printf("Prediction for input {0, 1, 0, 1, 1}: %.2f\n", output[0]);
    //printf("Prediction for input {1, 0, 1, 1, 0}: %.2f\n", output[0]);
    //printf("Prediction for input : %.2f\n", output[0]);
    printf("%s\n", output[0] >= 0.5 ? "Branch Taken" : "Branch Not Taken");
    
    return 0;
}
```

---

####  Compile the C code using GCC compiler
  ```
  $ gcc bpnn.c
  $ ./a.out
  ```
![task1pic1](https://github.com/user-attachments/assets/e29b6be4-fcc6-43b6-873a-242241e0b11a)

---

####  Compile the C code using the RISC-V compiler ( -O1)
  ```
  $ riscv64-unknown-elf-gcc -O1 -mabi=lp64 -march=rv64i -o bpnn.o bpnn.c
  $ ls -ltr bpnn.o
  ```
![task2pic2](https://github.com/user-attachments/assets/dd4e62ac-7fee-4034-aeb3-0a3d69c166f9)

---

####  Inspect the Assembly Code for the Main Function (-O1)
   ```
   $ riscv64-unknown-elf-objdump -d bpnn.o | less
   ```
![task2pic3](https://github.com/user-attachments/assets/3b9cdfc3-c2bb-4891-9cd7-66ae7b0a06c2)

---

####  Compile the C code using the RISC-V compiler (-Ofast) 
  ```
  $ riscv64-unknown-elf-gcc -Ofast -mabi=lp64 -march=rv64i -o bpnn.o bpnn.c
  ```
![task2pic4](https://github.com/user-attachments/assets/44512c5a-b8b8-4c19-8bf3-d862352252be)

---

####  Inspect the Assembly Code for the Main Function (-Ofast)
  ```
  $ riscv64-unknown-elf-objdump -d bpnn.o | less
  ```
![task2pic5](https://github.com/user-attachments/assets/fc8e6b90-40fe-4d98-bd46-318af0838088)

---

####  Observe the ouput given by RISC V Compiler
  ```
  $ spike pk bpnn.o
  ```
![task2pic6](https://github.com/user-attachments/assets/a3fdb3ac-4f99-4270-b537-e142749afac6)

---

####  Inspect the Stack pointer in the  Assembly Code of the Main Function (-Ofast)
  ```
  $ riscv64-unknown-elf-objdump -d bpnn.o | less
  ```
![task2pic7](https://github.com/user-attachments/assets/9d91196d-f9fa-4fbb-bb1f-9347ad47a02e)

---

####  Debug the C code compiled by RISC V Compiler using spike command by inspecting the stack pointer
  ```
  $ spike -d pk bpnn.o
  ```
![task2pic8](https://github.com/user-attachments/assets/20470995-d3b2-415b-88a7-8e810bb4af90)

---

# Task2

## RISC-V ISA

The RV32I instruction set architecture (ISA) in RISC-V is made up of several types of instructions, which can be classified based on their functionalities and encoding formats. Below is a summary and classification of the various instructions, their groupings by bits, and the combinations defined for each function in the recent RV32I specification (May 2024).

### Types of Instructions in RV32I:
- **R-Type Instructions**
- **I-Type Instructions**
- **S-Type Instructions**
- **B-Type Instructions**
- **U-Type Instructions**
- **J-Type Instructions**

---

## R-Type Instruction Format

| **Bit**  | 31-25      | 24-20     | 19-15     | 14-12     | 11-7      | 6-0       |
|----------|------------|-----------|-----------|-----------|-----------|-----------|
| **Field**| funct7     | rs2       | rs1       | funct3    | rd        | opcode    |
| **Description** | Function code (7 bits) | Source Register 2 (5 bits) | Source Register 1 (5 bits) | Function code (3 bits) | Destination Register (5 bits) | Operation code (7 bits) |

#### Example for R-Type (ADD):

| **Bit**  | 31-25    | 24-20    | 19-15    | 14-12    | 11-7     | 6-0      |
|----------|----------|----------|----------|----------|----------|----------|
| **Value**| 0000000  | 00000    | 00000    | 000      | 00000    | 0110011 |
| **Operation** | ADD      |          |          | funct3    | rd (destination) | opcode (ADD)  |

---

## I-Type Instruction Format

| **Bit**  | 31-20      | 19-15     | 14-12     | 11-7      | 6-0       |
|----------|------------|-----------|-----------|-----------|-----------|
| **Field**| imm        | rs1       | funct3    | rd        | opcode    |
| **Description** | Immediate value (12 bits) | Source Register 1 (5 bits) | Function code (3 bits) | Destination Register (5 bits) | Operation code (7 bits) |

#### Example for I-Type (ADDI):

| **Bit**  | 31-20    | 19-15    | 14-12    | 11-7     | 6-0      |
|----------|----------|----------|----------|----------|----------|
| **Value**| 000000000000 | 00000    | 000      | 00000    | 0010011 |
| **Operation** | ADDI     |          | funct3    | rd (destination) | opcode (ADDI)  |

---

## S-Type Instruction Format

| **Bit**  | 31-25      | 24-20     | 19-15     | 14-12     | 11-7      | 6-0       |
|----------|------------|-----------|-----------|-----------|-----------|-----------|
| **Field**| imm[11:5]  | rs2       | rs1       | funct3    | imm[4:0]  | opcode    |
| **Description** | Immediate bits [11:5] (7 bits) | Source Register 2 (5 bits) | Source Register 1 (5 bits) | Function code (3 bits) | Immediate bits [4:0] (5 bits) | Operation code (7 bits) |

#### Example for S-Type (SB):

| **Bit**  | 31-25    | 24-20    | 19-15    | 14-12    | 11-7     | 6-0      |
|----------|----------|----------|----------|----------|----------|----------|
| **Value**| 0000000  | 00000    | 00000    | 000      | 00000    | 0100011 |
| **Operation** | SB (Store Byte) |          | funct3    | opcode (SB)  |

---

## B-Type Instruction Format

| **Bit**  | 31-25      | 24-20     | 19-15     | 14-12     | 11-8      | 7-1       | 0       |
|----------|------------|-----------|-----------|-----------|-----------|-----------|---------|
| **Field**| imm[12]    | rs2       | rs1       | funct3    | imm[10:5] | imm[4:1]  | imm[11] |
| **Description** | Immediate bit [12] (1 bit) | Source Register 2 (5 bits) | Source Register 1 (5 bits) | Function code (3 bits) | Immediate bits [10:5] (6 bits) | Immediate bits [4:1] (4 bits) | Immediate bit [11] (1 bit) |

#### Example for B-Type (BEQ):

| **Bit**  | 31-25    | 24-20    | 19-15    | 14-12    | 11-8     | 7-1      | 0       |
|----------|----------|----------|----------|----------|----------|----------|---------|
| **Value**| 0000000  | 00000    | 00000    | 000      | 000000   | 0001     | 0       |
| **Operation** | BEQ (Branch if Equal) |          | funct3    | opcode (BEQ)  |

---

## U-Type Instruction Format

| **Bit**  | 31-12      | 11-7      | 6-0       |
|----------|------------|-----------|-----------|
| **Field**| imm        | rd        | opcode    |
| **Description** | Immediate value (20 bits) | Destination Register (5 bits) | Operation code (7 bits) |

#### Example for U-Type (LUI):

| **Bit**  | 31-12    | 11-7     | 6-0      |
|----------|----------|----------|----------|
| **Value**| 000000000000 | 00000    | 0110111 |
| **Operation** | LUI      |          | opcode (LUI)  |

---

## J-Type Instruction Format

| **Bit**  | 31-12      | 11-7      | 6-0       |
|----------|------------|-----------|-----------|
| **Field**| imm        | rd        | opcode    |
| **Description** | Immediate value (20 bits) | Destination Register (5 bits) | Operation code (7 bits) |

#### Example for J-Type (JAL):

| **Bit**  | 31-12    | 11-7     | 6-0      |
|----------|----------|----------|----------|
| **Value**| 000000000000 | 00000    | 1101111 |
| **Operation** | JAL (Jump and Link) |          | opcode (JAL)  |

---

## Summary of Instruction Formats:

| **Instruction Type** | **Bit Position (31-0)**       | **Example**              |
|----------------------|-------------------------------|--------------------------|
| **R-Type**           | funct7 | rs2 | rs1 | funct3 | rd | opcode | | `ADD`, `SUB`           |
| **I-Type**           | imm | rs1 | funct3 | rd | opcode | | `ADDI`, `XORI`         |
| **S-Type**           | imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode | | `SB`, `SH`, `SW`       |
| **B-Type**           | imm[12] | rs2 | rs1 | funct3 | imm[10:5] | imm[4:1] | imm[11] | | `BEQ`, `BNE`, `BLT`    |
| **U-Type**           | imm | rd | opcode | | `LUI`, `AUIPC`         |
| **J-Type**           | imm | rd | opcode | | `JAL`                  |

---

## Encoding Branch Prediction Using a Neural Network Application Instructions

---
