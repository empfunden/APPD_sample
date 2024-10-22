Sample APPD and phase reduction implementation with the Hodgkin-Huxley model
==================================================================

Example of an implementation of the APPD method for simulating the Hodgkin-Huxley equations. 
The model used here is the classical 4-dimensional one, and includes a 5-dimensional cortical 
model as well. Code is based off of work by Ningyuan Wang.

Repository Structure
---------------------------
The main folder contains an implementation of the original APPD method for the 4D model. 
phase_reduction uses a shot method to calculate the limit cycle of both models and 
uses that as a metric for particle splits based on L2 distance to it. Numpy files and images
from previous runs are stored in png_output.
