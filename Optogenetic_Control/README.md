# maxwell-optogenetics
Python library and examples for optogenetic experiments on MaxWell Biosystems MaxOne



## Contents

setup.sh : installs many of the requirements to run the code.

opto_hardware.py: low-level (hardware) optogenetic control library.

optogenetics_enviroment.py: top-level (user) optogenetic control library.

/Arduino/SerialCommunication_V8/SerialCommunication_V8.ino : Arduino code, use Arduino IDE to open and download to the Arduino

/Calibration : folder contains example csv calibration files for LED power output and Jupyter Notebook for generating and reading calibration files 

1_Test_Arduino.ipynb : helps test that Arduino code is working with the computer

2_Experiment_Demo.ipynb : demonstrates an example experiment workflow and capabilities of the optogenetic control library

20230404T172401-2023_04_04_example_opto_stim_log.csv : example of how a generic stimulation log file looks like.


### Example experiment files: 

Experiment_hckcr1_2023_04_02.ipynb:
Example of experiment with illumination with 530nm light, without inducing seizures or closed loop illumination

Experiment_2023_04_04_hc328_hckcr1-2.ipynb:
Example of open and closed loop illumination with 530nm light, closed loop after seizure induction




