# Face Detection - Installation & Execution Guide

Follow these steps to set up and run the project:

---

## Table of Contents

- [1. Download and Extract the Project](#1-download-and-extract-the-project)
- [2. Open a Terminal in the Project Directory](#2-open-a-terminal-in-the-project-directory)
- [3. Launch the OpenCV Singularity Environment](#3-launch-the-opencv-singularity-environment)
- [4. Build the Project with CMake](#4-build-the-project-with-cmake)
    - [4.1 Generate the Makefile](#41-generate-the-makefile)
    - [4.2 Compile the Project](#42-compile-the-project)
- [5. Run the Program](#5-run-the-program)

---

## 1. Download and Extract the Project

- Download the ZIP file containing the project.
- Extract it to your desired location.

---

## 2. Open a Terminal in the Project Directory

- **Using GUI:**  
  Right-click inside the extracted folder and select **“Open in Terminal”**.

- **Using Command Line:**

```bash
  cd /path/to/your/project
```
  Make sure you're in the folder where you extracted the project before running any commands.

---
## 3. Launch the OpenCV Singularity Environment
- **On the virtual machine, activate the containerized environment:**

```bash
  start_opencv
```

**Note:** If this is your first time using the container, you might need to configure the Singularity container before running this command. Please refer to the setup documentation for the container configuration if necessary.

---
## 4. Build the Project with CMake

### 4.1 Generate the Makefile

```bash
cmake .
```
This step generates the **Makefile**, which contains the rules for building your project.

### 4.2 Compile the Project

```bash
make
```

This command compiles the project using the generated Makefile. If you encounter errors during this step, ensure you have the necessary dependencies and that the environment is correctly set up.

---

## 5️. Run the Program

- Execute the compiled program:

```bash
./Project_CV
```

**Note:** The executable will be located in the build folder if you used an out-of-source build, or directly in the project folder if you did an in-source build.

---
