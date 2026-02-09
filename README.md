 This project is centered around building an Intrusion Detection System to detction System to detect network attacks on a Controller Area Network (CAN). 
 I sought to build a supervised binary classification neural network small and accurate enough to fit on a low-power MCU (STM32F411RE-NUCLEO) or a high performance MCU (i.e. STM32H747-DISCO).

The primary challenge I faced was managing systems integrations between my hardware, embedded software environment, and machine learning software stack. 
The challenge of swinging between high-performance and poor programmability to high programmability and low performance augmented the challenge of improving the 
Python data pipeline and TensorFlow Machine learning workflows for my model. There are cleaning_ scripts and model_training_ scripts for the machine learning models, and .c files taken from STM32 Cube IDE which 
are a mix of what I've wrote and what CubeMX generated when I built my project with the X-Cube-AI extension enabled. 
Video demonstration: https://www.linkedin.com/posts/cyrus-p-057b6597_stm32-xcubeai-embeddedml-activity-7418911896982065153-KOPl?utm_source=share&utm_medium=member_desktop&rcm=ACoAABSjfAMBCD2FFO_jDz0WmHA_0T9tRgl2Evs