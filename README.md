# Face Detection - Installation & Execution Guide

This guide explains how to set up, build, and execute your face detection project inside an OpenCV Singularity container.

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
---
## 3. Launch the OpenCV Singularity Environment
- **On the virtual machine, activate the containerized environment:**

```bash
  start_opencv
```
---
## 4. Build the Project with CMake

### 4.1 Generate the Makefile

```bash
cmake .
```

### 4.2 Compile the Project

```bash
make
```

- If you encounter errors, ensure all dependencies (e.g., OpenCV) are installed and the environment is active.

---

## 5️. Run the Program

```bash
./Project_CV
```

- The executable will be in the root or build/ directory, depending on your build configuration.

---
