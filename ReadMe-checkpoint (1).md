# Migrate3D

<p> Migrate3D is a python program that streamlines and automates cell migration analysis, returning meaningful metrics that provide insight into cell migration. Migrate3D requires a .csv file input that contains a cell’s movement through three-dimensional space. Typically, this includes cell id number, x, y, z coordinates, and the time points of measurements. The program will return a .xlsx file that contains stepwise calculations done between each time point, entire track-based summary statistics, and individual contacts.  </p>

<p> Migrate3D was created with ease of use in mind, to that end a graphical user interface (GUI) was implemented.</p>

## Packages Required:
<p> dearpygui, numpy, pandas, python, statistics</p>

## Running Migrate3D
<p> To run Migrate3D, please refer to this link:
   https://projectdestroyer.zendesk.com/hc/en-us/articles/360037257633-How-to-run-scripts-from-GitHub
</p>
<p>
This tutorial offers a cohesive walk through on how to install git, and run desired github files. 
</p>
<p>The python file you should run in your command line is dpg.py</p>
<p> When the GUI launches please be sure to include the .xlsx file extension when naming the save file </p>

## General Inputs:

### Interval:
The interval parameter is an integer variable that allows for the user to indicate the range of intervals stepwise calculations will be preformed on. The default is 15 and primarily impacts Euclidean distance calculations, and angle timepoint measurements and related calculations.
</p>

### Arrest Displacement:
<p>This parameter is a floating point variable that is used to determine if the program considers if a cell has moved between two timepoints. This parameter is compared to each cell’s instantaneous displacement, and if its above or equal to the user’s input it will consider the cell moving within that timeframe. It is important to note that even if the instantaneous displacement is below the user’s input calculations will still be performed, however if they pass this threshold, they will be considered filtered and reported again in their own column.</p>


### Contact Length:
<p>Contact length is a floating-point variable that represents the distance between two unique cells that would be considered a contact. The program assumes same units as the x, y, and z coordinates.</p>


### Moving:
<p>Moving is an integer parameter that denotes the amount of moving timepoints a cell must exceed to be considered for summary statistics.</p>

