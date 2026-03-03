 This project is centered around building an Intrusion Detection System to detction System to detect network attacks on a Controller Area Network (CAN). 
 I sought to build a supervised binary classification neural network small and accurate enough to fit on a low-power MCU (STM32F411RE-NUCLEO) or a high performance MCU (i.e. STM32H747-DISCO).

OVERVIEW
The primary challenge I faced was managing systems integrations between my hardware, embedded software environment, and machine learning software stack. 
The challenge of swinging between high-performance and poor programmability to high programmability and low performance microcontroller evaluation boards augmented the challenge of improving the Python data pipeline and TensorFlow Machine learning workflows for my model. I attempted to clean and train my 1D CNN on the ROAD, OTIDS, and SynCAN datasets. The ROAD dataset proved to be the best among these three, so it has more code for it related to preprocessing, training, testing, compressing, and deploying its model. There are cleaning_ scripts and model_training_ scripts for the machine learning models, and .c files taken from STM32 Cube IDE which are a mix of what I've wrote and what CubeMX generated when I built my project with the X-Cube-AI extension enabled. 
Video demonstration:
https://www.linkedin.com/posts/cyrus-p-057b6597_stm32-xcubeai-embeddedml-activity-7418911896982065153-KOPl?utm_source=share&utm_medium=member_desktop&rcm=ACoAABSjfAMBCD2FFO_jDz0WmHA_0T9tRgl2Evs

Program Flow: 
1. In the *cleaning_ROAD.ipynb* file, it loads ROAD dataset data (only extracted signal columns) and combines attack/ambient samples
2. read csv into data frame, fixes and sorts timestamps, and forward fills signal columns, replacing NaNs with 0. 
3. makes uniform sampling rate of 200hz and creates attack intervals using Label column and attack injection interval metadata
4. creates 3s overlapping sliding windows with stride 0.5 seconds
5. perform train/test stratified split (avoid oversampling ambient data)
6. In model_training_ROAD.ipynb, 
CHALLENGES
Understanding the X-CUBE-AI build process and generated code (ai_run, acquire_and_process_data, post_process)
Understanding and locating datasheets for every Microcontroller chipset and evaluation board I used.

Finding the right platform: 
STM32H747 DISCO -> dual-core cortex-M7/M4 MCU, 2 MB Flash and 1 MB RAM, basically impossible to program
STM32F411re -> single-core cortex-M4, 512 KB Flash, 128 KB SRAM, too weak to hold program but easy to program
STM32H723zg -> single-core cortex-M7, 1 MB Flash, 564 KB RAM, easy to program, higher  memory than f411re. 

Managing a software-hardware pipeline with multiple points of failure.

Large window (600 frames) 
    -> good accuracy 
    -> too big for MCU (windows are too large to fit more than 2 samples)

Small window 
    -> fits on MCU 
    -> accuracy drops down to 0/AUC becomes 0.5 (random guessing)

Quantization 
    -> would save memory 
    -> destroys accuracy on tiny model

NEXT STEPS/FURTHER RESEARCH
Could use stronger chips or more flexible ones that support FP16 quantization from another manufacturer like the TI Sitara AM62A
or NXP's i.MX RT1062/i.MX RT1176.
Could also continue trying to shorten window sizes/stride to make more data fit into memory and increase model complexity. 
Could also try to train on new datasets like can-train-and-test: https://doi.org/10.1016/j.cose.2024.103777

SOURCES
https://www.digikey.com/en/maker/projects/tinyml-getting-started-with-stm32-x-cube-ai/f94e1c8bfc1e4b6291d0f672d780d2c0