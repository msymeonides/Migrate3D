# Migrate3D

<p> Migrate3D is a python program that streamlines and automates cell migration analysis, returning meaningful metrics that provide insight into cell migration. Migrate3D requires a .csv file input that contains a cell’s movement through three-dimensional space. Typically, this includes cell id number, x, y, z coordinates, and the time points of measurements. The program will return a .xlsx file that contains stepwise calculations done between each time point, entire track-based summary statistics, and individual contacts.  </p>

<p> Migrate3D was created with ease of use in mind, to that end a graphical user interface (GUI) was implemented.</p>

## Packages Required:
<p> dearpygui, numpy, pandas, openpyxl, python, statistics</p>

## Running Migrate3D
<p> 
1. To run Migrate3D, first download the latest version of Python: https://www.python.org/downloads/
</p>
<p>
2. Once Python is installed, create a folder where you would like to store Migrate3D. 
</p>
<p>
3. Now, go to the command prompt or an equivilent application and set the working directory to the folder you have just created this is done by typing: cd 'path to folder'. Note 'path to folder' would be the name of the folder you have just created. 
</p>
<p> 
4. Now that you have set your working directory you will have to install the required packages. To do this, in the command prompt that you have just set to your working directory, type:
</p>
<p>
py -3 -m pip install [packages listed above]. You will need to do this for each individual package.
</p>
<p>
5. You will now need to donload git, which is used to pull scripts from github and run them on your computer: https://gist.github.com/derhuerst/1b15ff4652a867391f03. You will need to close down your command prompt and repeat step 3.  </p>
<p>
6. Now that you have all of the required packages and git installed, you will need to clone the repository. To do this type in the command prompt: git clone https://github.com/msymeonides/Migrate3D.git
</p>
<p>
7. To run Migrate3D, now type: py dpg.py</p>
<p> 
8. When the GUI launches please be sure to include the .xlsx file extension when naming the save file and also have the .csv file saved in the folder you created in step 2.</p>

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

