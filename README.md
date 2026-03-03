
# OVERVIEW:
 This project is centered around building an Intrusion Detection System to detect network attacks on a Controller Area Network (CAN) bus. 
 I sought to build a supervised binary classification convolutional neural network (CNN) small and accurate enough to fit on a high performance MCU (STM32H723ZG).

The primary challenge I faced was managing systems integrations between my hardware, embedded software environment, and machine learning software stack. 
The challenge of swinging between high-performance and poor programmability to high programmability and low performance microcontrollers augmented the challenge of improving the Python data pipeline and TensorFlow Machine learning workflows for my model. I attempted to clean and train my 1D CNN on the ROAD, OTIDS, and SynCAN datasets. The ROAD dataset proved to be the best (most labeled data with indicated attack intervals) among these three, so it has more code for it related to preprocessing, training, testing, compressing, and deploying its model. There are cleaning_ scripts and model_training_ scripts for the machine learning models, and .c files taken from STM32 Cube IDE which are a mix of what I've wrote and what CubeMX generated when I built my project with the X-Cube-AI extension enabled. 

Video demonstration:
https://www.linkedin.com/posts/cyrus-p-057b6597_stm32-xcubeai-embeddedml-activity-7418911896982065153-KOPl?utm_source=share&utm_medium=member_desktop&rcm=ACoAABSjfAMBCD2FFO_jDz0WmHA_0T9tRgl2Evs

# PROJECT FLOW

* **Data Cleaning** (`cleaning_ROAD.ipynb`)
  * Load ROAD dataset extracted signal CSVs, combining attack and ambient samples
  * Read each CSV into a dataframe, fix and sort timestamps, forward-fill signal columns, replace remaining NaNs with 0
  * Resample to uniform 200Hz and construct attack interval labels using the Label column and injection interval metadata
  * Create 3-second overlapping sliding windows with 0.5-second stride
  * Perform stratified train/test split at the session (file) level to avoid ambient data dominance

* **Model Training** (`model_training_ROAD.ipynb`)
  * Preprocess dataset with standard mean/std scaling, create categorical output labels
  * Define 1D-CNN architecture: Input → Conv1D(32, 4, ReLU) → GlobalAveragePooling1D → Dense(2, softmax)
  * Fit model on training data with callbacks monitored on validation AUC
  * Attempt INT8 quantization via TFLite — classification accuracy collapses; FP32 TFLite saved instead

* **Sample Generation** (`ROAD_Model_Compression.ipynb`)
  * Generate `samples_fp32.h/.c` containing a small number of test samples alongside their per-feature mean and std deviation for use in on-chip inference demonstration

* **MCU Configuration** (STM32CubeMX)
  * Configure and enable peripherals: UART, I2C, timers
  * Upload `.tflite` model file via the X-CUBE-AI extension (Application Template option)
  * Set clock frequency (480MHz) and build system, then generate C code template as `.cproject`

* **Firmware Development** (STM32CubeIDE)
  * Import `samples_fp32.h/.c` into `main.c`
  * In `X-CUBE-AI/App/app_x-cube-ai.c`, define three functions:
    * `ai_run()` — overall inference loop
    * `acquire_and_process_data()` — reverses FP16 compression and applies mean/std standardization to input data
    * `post_process()` — interprets output probabilities and writes prediction result to I2C LCD display

* **Flashing** (STM32CubeProgrammer)
  * STM32CubeIDE was occasionally unreliable running its built executable directly
  * Workaround: manually load the compiled `.elf` file onto the chip via STM32CubeProgrammer — https://www.st.com/en/development-tools/stm32cubeprog.html

# CHALLENGES:
## General
* Understanding the X-CUBE-AI build process and generated code (ai_run, acquire_and_process_data, post_process)
* Understanding and locating datasheets for every Microcontroller chipset and evaluation board I used.
* Figuring out the entire STM32CubeIDE ecosystem and struggling to make builds and flash code to the board across a diverse ecosystem,
this involved testing out CLI tools like Make, OpenOCD, st-link, and arm-none-eabi-gcc. 

## Finding the right platform:  
* STM32H747 DISCO -> dual-core cortex-M7/M4 MCU, 2 MB Flash and 1 MB RAM, basically impossible to program
* STM32F411re -> single-core cortex-M4, 512 KB Flash, 128 KB SRAM, too weak to hold program but easy to program
* STM32H723zg -> single-core cortex-M7, 1 MB Flash, 564 KB RAM, easy to program, higher  memory than f411re. 

## Managing a software-hardware pipeline with multiple points of failure

* Large window (600 frames)
  * High AUC
  * Too large for MCU — only 2 samples fit in memory at once
* Small window
  * Fits on MCU
  * Accuracy collapses — AUC drops to ~0.5 (random guessing)
* Quantization
  * Would save memory
  * Destroys accuracy on a model this small due to high proportional quantization error

# NEXT STEPS / FURTHER RESEARCH

## Hardware & Deployment
* The STM32H723ZG tensor arena likely only used AXI SRAM (320KB), leaving SRAM1/2, SRAM3, and DTCM (~256KB total) unused. A custom linker script assembling a larger contiguous arena across banks could allow more samples in memory.
* Could use stronger or more flexible chips that support FP16 quantization, such as the TI Sitara AM62A or NXP i.MX RT1062/i.MX RT1176.
* Measure mean and worst-case inference latency, as well as power consumption on the H7

## Model & Training
* Fix class weights
* Quantization-aware training (QAT)
* Could continue trying to shorten window sizes and stride to fit more data into memory and increase model complexity.
* Small adjustments to filter count and kernel size 
* Report per-class precision, recall, and F1 on the attack class specifically. 

## Data & Generalization
* Verify window-level class imbalance 
* The ROAD dataset is dyno-only. Generalization to on-road traffic is unvalidated. Could also train on newer datasets like can-train-and-test (https://doi.org/10.1016/j.cose.2024.103777) or CAN-MIRGU, which include moving-vehicle captures.
* Cross-vehicle generalization

# SOURCES (not exhaustive):
* Verma et al., "A comprehensive guide to CAN IDS data and introduction of the ROAD dataset"
* Kukkala, Thiruloga, and Pasricha, "LATTE: LSTM Self-Attention based Anomaly Detection in Embedded Automotive Platforms"
* DigiKey, "TinyML: Getting Started with STM32 X-CUBE-AI" — https://www.digikey.com/en/maker/projects/tinyml-getting-started-with-stm32-x-cube-ai/f94e1c8bfc1e4b6291d0f672d780d2c0
* STMicroelectronics, ST Edge AI Developer Cloud — https://stedgeai-dc.st.com/home?ecmp=tt39873_gl_video_jul2024
* STMicroelectronics, STM32H723ZG Datasheet (DS13313) — https://www.st.com/resource/en/datasheet/stm32h723zg.pdf
* STMicroelectronics, STM32H723/733 Reference Manual (RM0468) — https://www.st.com/resource/en/reference_manual/dm00603761.pdf
* Arm, Cortex-M7 Processor Technical Reference Manual (DDI0489) — https://developer.arm.com/documentation/ddi0489/latest
* Arm, Cortex-M7 Generic User Guide (DUI0646, covers DWT cycle counter) — https://documentation-service.arm.com/static/61efd6602dd99944d051417b

