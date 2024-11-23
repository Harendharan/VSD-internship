# VSDSquadron-Mini-Internship
---

Details 
## Task 1: 
---
### Step 1: Create the Program file 
  : Created a file named `sum1ton.c` in the home folder, typed a C code to compute the sum of numbers from 1 to n using a text editor, and saved it.
    ![c_code](https://github.com/user-attachments/assets/bfd3de06-efef-4f54-ac66-f96ebd0ed2ba)

### Step 2: Compile the C code using GCC compiler
  : The code was compiled using GCC Compiler and the output was printed on the terminal. 
    ![c_output](https://github.com/user-attachments/assets/2b123eda-2d8c-4f79-8a75-c8b95762c5ca)

### Step 3: Compile the C code using the RISC-V compiler (-O1) 
  : The program was compiled with the riscv64-unknown-elf-gcc compiler using the -O1 optimization level.
    ![sum1ton o](https://github.com/user-attachments/assets/dd69cb0b-974f-4fb6-9c33-a559b6942f88)

### Step 4: Inspect the Assembly Code for the Main Function (-O1)
   : The disassembled assembly code for the main function was inspected to observe how the compiler optimized the program with the -O1 optimization level.
     ![main](https://github.com/user-attachments/assets/4319d61e-678e-467c-a5ad-de6c4dce23ad)

### Step 5: Compile the C code using the RISC-V compiler (-Ofast) 
  : The program was recompiled using the riscv64-unknown-elf-gcc compiler with the -Ofast optimization level to enable aggressive optimizations for performance.
  ![ofast](https://github.com/user-attachments/assets/4c9d90d4-0593-4c65-8364-8fc473597969)

### Step 6: Inspect the Assembly Code for the Main Function (-Ofast)
  : The disassembled assembly code for the main function was inspected again to analyze the effects of the -Ofast optimization level and compare it with -O1.
  ![main2](https://github.com/user-attachments/assets/5d08ed72-ee14-458f-8a51-3c0a881a4369)

  
