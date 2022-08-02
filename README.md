# README

Created: August 1, 2022 2:49 PM
Last Edited Time: August 2, 2022 2:49 PM

# Migrate3D

Migrate3D is a python program that streamlines and automates cell migration analysis, returning meaningful metrics that provide insight into cell migration. Migrate3D requires a .csv file input that contains a cell’s movement through three-dimensional space. Typically, this includes cell id number, x, y, z coordinates, and the time points of measurements. The program will return a .xlsx file that contains stepwise calculations done between each time point, entire track-based summary statistics, and individual contacts.

Migrate3D was created with ease of use in mind, to that end a graphical user interface (GUI) was implemented.

## Python Packages Required:

dearpygui, numpy, pandas, openpyxl, statistics

## Running Migrate3D

1. To run Migrate3D, first download the latest 3.x version of Python: https://www.python.org/downloads/
2. Once Python is installed, create a folder where you would like to store Migrate3D.
3. Now, go to the command prompt or an equivalent application and set the working directory to the folder you have just created.
4. Now that you have set your working directory you will have to install the required packages. To do this, in the command prompt that you have just set to your working directory, type:

```powershell
py -3 -m pip install dearpygui numpy pandas openpyxl
```

1.  Download the code as a zip file from GitHub and move [dpg.py](http://dpg.py) into the folder created in step 2.
2. To run Migrate3D, navigate to the PowerShell or an equivalent change your working directory to where you stored dpg.py, then run:

```powershell
py dpg.py
```

- When the GUI launches please be sure to include the .xlsx file extension when naming the save file and also have the .csv file saved in the folder you created in step 2.

## General Inputs

### Interval:

The interval parameter is an integer variable that allows for the user to indicate the range of intervals stepwise calculations will be preformed on. The default is 15 and primarily impacts Euclidean distance calculations, and angle timepoint measurements and related calculations.

### Arrest Displacement:

This parameter is a floating point variable that is used to determine if the program considers if a cell has moved between two timepoints. This parameter is compared to each cell’s instantaneous displacement, and if its above or equal to the user’s input it will consider the cell moving within that timeframe. It is important to note that even if the instantaneous displacement is below the user’s input calculations will still be performed, however if they pass this threshold, they will be considered filtered and reported again in their own column.

### Contact Length:

Contact length is a floating-point variable that represents the distance between two unique cells that would be considered a contact. The program assumes same units as the x, y, and z coordinates.

### Moving:

Moving is an integer parameter that denotes the amount of moving timepoints a cell must exceed to be considered for summary statistics.

### Arrested:

A floating point variable that is used when determining contacts. An arrest coefficient is calculated for each cell, and if it is above the user specified input it will be considered an alive cell, and be analyzed as a moving cell.

### Time-lapse Interval:

The time between each observation, in minutes.

### Tau Value:

The number of Mean Squared Displacements you want to calculate. Will calculate from 1 to the number the user enters.

## Calculations

Step-wise calculations generated for each cell iteratively over each time point.

### Instantaneous Displacement:

The square root of the sum of the squared difference between x, y, and z coordinates over one timepoint difference. Used to show how much a cell has moved over a one timepoint difference.

$$
Instantaneous \ Displacement =\sqrt{(x_{t} - x_{t-1})^{2} + (y_{t} - y_{t-1})^{2} +(z_{t} - z_{t-1})^{2}}
$$

### Total Displacement:

The square root of the sum of the squared difference between x, y, and z coordinates compared to the timepoint 0. Gives a metric on how far a cell has moved from its origin.

$$
Total \ Displacement =\sqrt{(x_{t} - x_{t[0]})^{2} + (y_{t} - y_{t[0]})^{2} + (z_{t} - z_{t[0]})^{2}}
$$

### Path Length:

The total path length of a cell. It is the sum of all instantaneous displacements. Shows how far a cell has traveled. 

$$
Path \ Length = \sum_{}^{} Instantaneous \ Displacement
$$

### Instantaneous Velocity:

A cell’s velocity between one timepoint difference. 

$$
Instantaneous \ Velocity =  \frac{Instantaneous \ Displacement \ _{t}}{Time \ lag} 
$$

### Instantaneous Acceleration:

A cell’s acceleration between timepoints.

$$
Instantaneous \ Acceleration=\frac {Instantaneous \ Velocity \ _{t} \ - \ Instantaneous \ Velocity \ _{t-1}}{Time \ lag} 
$$

### Euclidean Distance:

The Euclidean distance over n timepoints. Finds the length of a line segment between a cell’s coordinates over n time points.

$$
Euclidean \ Distance =\sqrt{(x_{t} - x_{t-n})^{2} + (y_{t} - y_{t-n})^{2} + (z_{t} - z_{t-n})^{2}}
$$

### Angle Between Cell Vectors:

The magnitude of two vectors separated by n time points. The magnitude is found by subtracting a cell’s coordinates over n time points. From these magnitudes, a Frobenius norm is calculated. From this the Frobenius norm’s dot product is clipped between values -1.0 and 1.0, and the trigonometric inverse of the cosine is taken, returning a radian. From this, the angle is found by multiplying radians by 180 divided by pi.

## Summary Sheet

Summary statistics using the data acquired over a cell’s entire tracking period.

### Final Euclidean Distance:

The total displacement from a cell’s last recorded time point to a cell’s origin.

### Maximum Euclidean Distance:

The furthest recorded point a cell was recorded being away from the origin.

### Straightness:

The Final Euclidean divided by the Path Length. A metric of how confined a cell is.

$$
Straightness=\frac{Final Euclidean}{Path\ Length}
$$

### Time Corrected Straightness:

Straightness multiplied by the square root of the duration of a cell’s tracking duration.

$$
Time\ Corrected\ Straightness=Straightness\times\sqrt{Duration}
$$

### Displacement Ratio:

The Final Euclidean divided by the Max Euclidean. 

$$
Displacement \ Ratio = \frac{Final \ Euclidean \ Distance}{Maxiumum\ Euclidean\ Disatance}
$$

### Outreach Ratio:

The Max Euclidean divided by the Path Length.

$$
Outreach\ Ratio =\frac{Maximum\ Euclidean\ Distance}{Path\ Length}
$$

### Arrest Coefficient:

A metric used to determine how motile a cell is. Values will range from 0-1, the closer to 1, the more motile a cell is.

$$
Arrest\ Coefficient = \frac{Time\ spent\ arrested}{Duration}
$$

### Mean Squared Displacement (MSD):

The mean squared displacement over a certain number of time lags, denoted by Tau(τ). Used to show that over t time points a cell can be expected to move this much. This can be calculated over a user specified number of τ values.

$$
MSD(τ)=\lt(x(t + τ) - x(t))^2 \gt 
$$

### Filtered Metrics:

Metrics for cells that have been considered alive by the users Arrest Displacement input.

## Contacts

Contacts will iterate over over all the cells in the dataset comparing their x, y, and z coordinates. If two cells are closer in these dimensions than the user inputted contact length, it will be recorded as a contact.

Contacts will iterate over over all the cells in the dataset comparing their x, y, and z coordinates. If two cells are closer in these dimensions than the user inputted contact length, it will be recorded as a contact.
