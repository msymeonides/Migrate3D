# README

Last Edited: June 12, 2025 (Migrate3D version 2.0)

# Migrate3D

Migrate3D is a Python program that streamlines and automates biological object motion (e.g. cell migration) analysis, returning meaningful metrics that help the user evaluate biological questions. This program does not handle imaging data, only previously generated tracking data, so it is not meant to replace functions already performed very well by programs such as Imaris, Arivis Pro, CellProfiler, TrackMate etc. Migrate3D’s purpose is to take the tracks produced from any of these programs and quickly and easily process the data to generate various metrics of interest in a transparent and tunable fashion, all done through an intuitive graphical user interface (GUI).

These results can be used in downstream analyses to compare different conditions, categories of objects, etc. The calculated metrics are all adapted from existing reports in the literature where they were found to have biological significance. Migrate3D not only calculates simple metrics such as track velocity and arrest coefficient, but more complex ones such as straightness index (i.e. confinement ratio), mean squared displacement across selectable time lags, relative turn angles, etc., and includes adjustable constraints and filters to ensure that clean results are produced.

Migrate3D requires a .csv file input that contains data from object movements through two- or three-dimensional space. After execution, the program will return a set of .xlsx files, each with several worksheets, containing track-by-track summary metrics, mean squared displacement analysis, etc. If the user imports a Categories .csv file (simply listing the category for each object ID), two additional analyses will be performed: dimensionality reduction using principal component analysis (PCA), and decision tree analysis using XGBoost. There is no guarantee that these will be performed in a statistically-sound way, that is left up to the user to ensure, but the libraries used to perform these analyses are widely used.

A limitation of the program is that it does not currently handle cell divisions (or fusions) in any intelligent way, so the user needs to separate all such tracks at the split/merge point so that each track only represents one cell. (Note: a record of which daughters belong to which parent cell can easily be kept using a simple numbering system within the track’s Name field.)

Migrate3D was developed by Menelaos Symeonides, Emily Mynar, Matthew Kinahan, and Jonah Harris at the University of Vermont, funded by NIH R21-AI152816 and NIH R56-AI172486 (PI: Markus Thali). We welcome feedback and intend to continue developing and supporting the program as resources allow.

## Input Files

A Segments input file is required to run Migrate3D. Optionally, a Categories input file can be provided to perform additional analyses. In both cases, the program will "guess" which columns contain which data, but if this fails, you can select them through a drop-down box in the GUI. 

### Segments

The Segments input file should be a .csv with five columns (or four for 2D data): object ID, time, X, Y, and Z coordinates. Please ensure that column headers are in the first row of the .csv file input. Note that the Time column is expected to contain a "real" time value (e.g. number of seconds), not just the number of timepoints elapsed.

### Categories

The Categories input file should be a .csv with object ID and object category. Please ensure that column headers are in the first row of the .csv file input. If no Categories file is imported, the PCA and XGBoost analyses (and anything else done per-category) will not be performed. 

## Installing and Running Migrate3D

These installation instructions involve the use of the command line. If you are not familiar with using the command line, just copy each line and paste into your prompt/terminal and press Enter. Once the process is complete, you will be able to paste in the next line and press Enter, and so on. If "sudo" is used, you will need to enter your account password to proceed.

### On Windows (tested in Windows 10 and 11)

1. First, download and install the latest version of Miniconda3 for Windows using all the default options during installation: https://docs.conda.io/projects/miniconda/en/latest/index.html

2. From the Start menu, open the Anaconda Prompt that was just installed. Create a folder for Migrate3D and navigate to it:
```powershell
mkdir Migrate3D
cd Migrate3D
```

3. Download Migrate3D from GitHub, extract the ZIP file, and navigate into the subfolder that was just created:
```powershell
curl -LJO https://github.com/msymeonides/Migrate3D/archive/main/Migrate3D-main.zip
tar -xvzf Migrate3D-main.zip
cd Migrate3D-main
```

4. Set up a virtual environment (venv) and activate it:
```powershell
conda update conda
conda create --name Migrate3D
conda activate Migrate3D
```
Note: if you would like to exit the venv, i.e. return to the base Anaconda prompt, simply enter:
```powershell
conda deactivate
```

5. Install the required dependencies:
```powershell
conda install pip
pip install -r requirements.txt
```
Note that these packages are only installed within the venv you just created and will not affect your base python installation.

6. Finally, to run Migrate3D:
```powershell
python %USERPROFILE%\Migrate3D\Migrate3D-main\main.py
```
Remember to open the Anaconda Prompt and activate the Migrate3D venv next time you want to run Migrate3D:
```powershell
conda activate Migrate3D
python %USERPROFILE%\Migrate3D\Migrate3D-main\main.py
```
In the prompt, you will see a notification that the GUI is now available ("Dash is running on http://127.0.0.1:5555/). You can now go to this address in your web browser to access the Migrate3D GUI.

Note that the output result spreadsheets will be saved under C:\Users\your_username\Migrate3D\Migrate3D-main\.

### On macOS (tested in Catalina 10.15.7):

1. First, download and install the latest version of Miniconda3 for macOS (the pkg version will be easiest to install): https://docs.conda.io/projects/miniconda/en/latest/index.html

2. Open a Terminal. Create a folder for Migrate3D and navigate to it:
```powershell
mkdir Migrate3D
cd Migrate3D
```

3. Download Migrate3D from GitHub, extract the ZIP file, and navigate into the subfolder that was just created:
```powershell
curl -LJO https://github.com/msymeonides/Migrate3D/archive/main/Migrate3D-main.zip
tar -xvzf Migrate3D-main.zip
cd Migrate3D-main
```

4. Set up a virtual environment (venv) and activate it:
```powershell
conda create --name Migrate3D
conda activate Migrate3D
```
Note: if you would like to exit the venv, i.e. return to the base Anaconda prompt, simply enter:
```powershell
conda deactivate
```

5. Install the required dependencies:
```powershell
sudo xcode-select --install
conda install pip
pip3 install -r requirements.txt
```
Note that these packages are only installed within the venv you just created and will not affect your base python installation.

6. Finally, to run Migrate3D:
```powershell
python3 ~/Migrate3D/Migrate3D-main/main.py
```
Remember to first activate the Migrate3D venv next time you want to run Migrate3D before executing the main script:
```powershell
conda activate Migrate3D
python3 ~/Migrate3D/Migrate3D-main/main.py
```
Note that the output result spreadsheets will be saved under /Users/your_username/Migrate3D/Migrate3D-main/.

### On Linux (tested in Ubuntu 23.10):

1. Python 3 is already installed in Ubuntu. Begin by checking the installed version of python:
```powershell
python3 --version
```
On a fresh installation of Ubuntu 23.10, that should return "Python 3.11.6". Python versions 3.12.x and 3.13.x will also work, but earlier versions may not. If you have an earlier version of python, you may need to update it before proceeding.

2. If you have not previously configured a python virtual environment (venv) or installed python packages using pip, you will first need to get set up to do that:
Open a Terminal window and enter the following commands:
```powershell
sudo apt update
sudo apt upgrade
sudo apt-get install python3-pip python3-venv
```

3. Make a new directory (e.g. named "Migrate3D") and create a venv in it:
```powershell
mkdir ~/Migrate3D
python3 -m venv ~/Migrate3D
```

4. Download Migrate3D from GitHub and extract the ZIP file into the /home/Migrate3D directory you just created:
```powershell
wget https://github.com/msymeonides/Migrate3D/archive/main/Migrate3D-main.zip
unzip Migrate3D-main.zip -d ~/Migrate3D
```

5. You now need to activate the venv:
```powershell
source ~/Migrate3D/bin/activate
```
Note: if you would like to exit the venv, i.e. return to your normal Linux terminal, simply enter:
```powershell
deactivate
```

6. Install the required dependencies:
```powershell
pip install -r ~/Migrate3D/Migrate3D-main/requirements.txt
```
Note that these packages are only installed within the venv you just created and will not affect your python installation.

7. Finally, to run Migrate3D:
```powershell
python3 ~/Migrate3D/Migrate3D-main/main.py
```
Remember to first activate the Migrate3D venv next time you want to run Migrate3D before executing the main script:
```powershell
source ~/Migrate3D/bin/activate
python3 ~/Migrate3D/Migrate3D-main/main.py
```
Note that the output result spreadsheets will be saved under ~/Migrate3D/Migrate3D-main/.


## Tunable Parameters

Note that you can change the default values of these variables at the top of the main.py script, or you can change them in the GUI before running the analysis.

### Arrest Limit:

A floating point variable that is used to determine whether an object has "really" moved between two timepoints. This parameter is compared to each object’s instantaneous displacement between each pair of consecutive timepoints, and if that value is at least equal to the user’s Arrest Limit input, it will consider the object as moving within that timeframe. It is important to note that even if the instantaneous displacement is below the user’s Arrest Limit input, calculations will still be performed, however if they pass this threshold, they will survive the filter and be reported again in their own column (in verbose mode). Set this value by examining tracks of control objects which should not be moving and finding the maximum instantaneous displacement that they exhibit. It is not recommended to set this value to 0 as it is exceedingly unlikely that your imaging system is perfectly stable and all non-zero values of instantaneous displacement represent "biologically true" movements.

### Minimum Timepoints:

An integer parameter that denotes the amount of moving timepoints an object must exceed to be considered for certain summary statistics, namely those relating to Velocity and Acceleration that are denoted as "Filtered". Tracks which fail to meet this threshold will show no value for those summary statistics, but will still have their own row on the Summary Sheet with values everywhere but those Filtered Velocity/Acceleration columns. To turn this filter off, set this value to 0.

### Contact Length:

A floating-point variable that represents the largest distance between two objects at a given timepoint that would be considered a contact. The program assumes same units as that of the X/Y/Z coordinates given in the input dataset.

### Maximum Arrest Coefficient:

A floating point variable between 0 and 1 that uses each object's Arrest Coefficient to filter out "dead" objects during the contact detection process. To turn this filter off, set this value to 1.


## Autodetected Parameters

### Timelapse Interval:

The time between each observation (assumes the same units as the values in the Time column of your dataset). This is automatically detected from the input dataset, but can be manually overridden in the GUI if necessary.

### Maximum MSD Tau Value:

An integer variable that controls the number of Mean Squared Displacement intervals to calculate. It is recommended to set this value to a number equal to the total number of timepoints that the majority of the tracks in the dataset have. This is autodetected from the input dataset by calculating the number of timepoints in all object IDs and taking the mode of that set, but can be manually overridden in the GUI if necessary.

### Maximum Euclidean distance Tau value:

An integer variable that controls the range of intervals that Euclidean Distance and Turning Angle calculations will be performed on. This is automatically set to half the number of the MSD Tau Value, but can be manually overriden in the GUI if necessary.


## Formatting options

All of these formatting options are optional, but depending on the nature of your dataset, may be necessary to ensure accurate results.

### Multitracking:

If an object ID is represented by multiple segments at a given timepoint, they will be spatially averaged into one segment.

### Interpolation:

If an object ID is missing a timepoint, that timepoint will be inferred by simple linear interpolation and inserted. This will happen with any number of missing timepoints, but this will never add interpolated timepoints before the first or last available timepoints for that object ID, i.e. it will not extend the track in either direction but will fill in internal gaps.

### Verbose:

Includes the results of all intermediate step-wise calculations in the output .xlsx file. This will result in a larger file size, but allows the user to see how each metric was calculated for each object at each timepoint.

### Contacts:

Identifies contacts between objects at each timepoint, and returns a separate results .xlsx file containing each detected contact as well as a summary of contact history for each object (excluding objects that had no detected contacts).

### Attractors:

Identifies instances where an object is attracting other objects towards it (even if both objects are moving), and returns a separate results .xlsx file containing data on each detected attraction event. An additional set of tunable parameters for this function is available in the GUI. The default values for these parameters can be changed at the top of the main.py script.

### Generate Figures:

Generates interactive violin plots for each of the summary statistics, and an interactive plot containing all tracks. These figures will be saved as a single .html file which can be viewed in a browser.

### Subset Categories:

In case your dataset includes several categories of objects, but you are only interested in including certain ones for PCA and XGBoost, enter those here separated by spaces. The entire rest of the analysis will still be done on every object ID in the dataset.

### Output Filename:

Enter a name for your output file. The .xlsx extension will be added on, do not include it in this field. Note that any output file with the same name present in the program folder will be overwritten. Additionally, if a file with the same name is currently open in Excel, this will cause Migrate3D to crash.


## Calculations

Step-wise calculations generated for each object iteratively over each time point.

### Instantaneous Displacement:

The square root of the sum of the squared difference between X, Y, and Z coordinates over one timepoint difference. Used to show how much an object has moved over one time interval.

$$
Instantaneous \ Displacement =\sqrt{(x_{t} - x_{t-1})^{2} + (y_{t} - y_{t-1})^{2} +(z_{t} - z_{t-1})^{2}}
$$

### Total Displacement:

The square root of the sum of the squared difference between X, Y, and Z coordinates of each timepoint compared to timepoint 0 for that object. Measures how far an object has moved from its origin at any given timepoint.

$$
Total \ Displacement =\sqrt{(x_{t} - x_{t[0]})^{2} + (y_{t} - y_{t[0]})^{2} + (z_{t} - z_{t[0]})^{2}}
$$

### Instantaneous Velocity:

An object’s velocity when considering just one time interval. 

$$
Instantaneous \ Velocity =  \frac{Instantaneous \ Displacement \ _{t}}{Time \ lag} 
$$

### Instantaneous Acceleration:

An object’s acceleration when considering just two consecutive time intervals.

$$
Instantaneous \ Acceleration=\frac {Instantaneous \ Velocity \ _{t} \ - \ Instantaneous \ Velocity \ _{t-1}}{Time \ lag} 
$$

### Euclidean Distance:

The Euclidean distance over a number of timepoints. Finds the length of a line segment between a pair of timepoints for an object, for all values of time up to τ (which is set in an autodetected/tunable parameter).

$$
Euclidean \ Distance =\sqrt{(x_{t} - x_{t-n})^{2} + (y_{t} - y_{t-n})^{2} + (z_{t} - z_{t-n})^{2}}
$$

The results of this analysis are saved in a worksheet named "Euclidean medians", with each row being an object ID and each column being a number of timepoints.

### Turning Angle:

The angle, θ, between two consecutive vectors, a and b, with a given timepoint interval (up to τ, which is a tunable parameter), is calculated by first finding the dot product of the two vectors:

$$
a \cdot b = a_xb_x + a_yb_y + a_zb_z
$$

Then using Pythagoras's theorem to calculate the magnitudes of each vector, e.g.:

$$
|a| = \sqrt{{a_x}^2 + {a_y}^2 + {a_z}^2}
$$

And finally, to find the angle θ, taking the inverse cosine of the dot product divided by the two vector magnitudes:

$$
\theta = \cos^{-1} (\frac{a \cdot b}{ |a||b| })
$$

The results of this analysis are saved in a worksheet named "Turning Angles", with each row being an object ID and each column being a number of timepoints.


## Summary Sheet

Summary statistics using the data acquired over an object’s entire tracking period.

### Final Euclidean Distance:

The total displacement from a track’s final recorded time point to the track’s origin.

### Maximum Euclidean Distance:

The furthest from its origin that an object was observed during its entire history.

### Path Length:

The total path length of a track. It is the sum of all instantaneous displacements. Shows how far an object has traveled along its path. 

$$
Path \ Length = \sum_{}^{} Instantaneous \ Displacement
$$

### Straightness:

The Final Euclidean Distance divided by the Path Length. A metric of how confined an object is, or how straight its track was.

$$
Straightness=\frac{Final\ Euclidean}{Path\ Length}
$$

### Time Corrected Straightness:

Straightness multiplied by the square root of the duration of that object’s track. Needed when the tracking duration of the objects in the dataset varies, otherwise this will be identical to Straightness.

$$
Time\ Corrected\ Straightness=Straightness\times\sqrt{Duration}
$$

### Displacement Ratio:

The Final Euclidean Distance divided by the Maximum Euclidean Distance, giving values between 0 and 1. A high Displacement Ratio value (approaching 1) denotes an object which did not return back towards its origin at the end of its track. Conversely, a low value (approaching 0) denotes an object which ventured away from its origin at some point, but by the end of the track returned closer to its origin.

$$
Displacement \ Ratio = \frac{Final \ Euclidean \ Distance}{Maximum\ Euclidean\ Distance}
$$

### Outreach Ratio:

The Maximum Euclidean Distance divided by the Path Length, giving values between 0 and 1. A high Outreach Ratio value (approaching 1) denotes an object which essentially moves in a near-perfect straight line away from its origin. Conversely, a low value (approaching 0) denotes an object which takes a very tortuous or meandering path.

$$
Outreach\ Ratio =\frac{Maximum\ Euclidean\ Distance}{Path\ Length}
$$

### Arrest Coefficient:

A metric used to determine how motile an object is, with values between 0 and 1, where a value of 0 denotes an object that moved throughout its history, and a value of 1 denotes an object that did not move at all throughout its history. The instantaneous displacement threshold used to determine whether an object is moving at a given timepoint is a tunable variable (Arrest limit).

$$
Arrest\ Coefficient = \frac{Time\ spent\ arrested}{Duration}
$$

### Mean Squared Displacement (MSD):

The mean squared displacement over a certain number of time lags, or tau (τ) values. Used to show that over t time points an object can be expected to move a certain amount. The maximum number of τ values is a tunable variable.

$$
MSD(τ)=\ \lt(x(t + τ) - x(t))^2 \gt 
$$

In the output file, two MSD result sheets are provided: one ("Mean Squared Displacements") with the MSD at each τ value (columns) for each object (rows), and one (MSD Summary) with the average and standard deviation of the MSD (columns) at each τ value (rows) across the whole dataset.
 
If a Categories file is provided, two additional result sheets are given: one ("MSD Avg Per Category") with the mean MSD at each τ value (rows) for each object category (columns), and one ("MSD StDev Per Category") with the standard deviation of MSD at each τ value (rows) for each object category (columns). These can be used to plot MSD log-log plots and evaluate whether a category of object is moving with a certain pattern. 

### Convex Hull Volume

The volume of a convex hull contained within the track is calculated. Essentially represents how much volume an object covered during its tracking history. Similarly to Straightness, a Time-Corrected value is also provided by multiplying each Convex Hull Volume value by the square root of the duration of that object’s track (needed when the tracking duration of the objects in the dataset varies). In the case of 2D data, this column will be left blank.


## Machine Learning Analyses

### Principal Component Analysis (PCA):

PCA is performed on the summary statistics of each object, and the results are saved in a separate .xlsx file. A Kruskal-Wallis test is performed on the PCA results to determine whether each principal component is significantly different between the provided categories of objects.

### XGBoost:

XGBoost is a decision tree-based machine learning algorithm that can reveal which motion parameters are most important for describing the variation in a dataset. The results are saved in a separate .xlsx file, and include the feature importance scores for each summary statistic when looking at the entire dataset, as well as for all possible pairs of category-to-category comparisons.


## Contacts

Contacts will iterate over all the objects in the dataset comparing their X, Y, and Z coordinates at each timepoint. If two objects are closer in these dimensions than the "Contact Length" tunable variable, it will be recorded as a contact. This option reports all contacts, and specific to cells, also contacts that are not mitotic, contacts that are alive, and a summary of each object’s contacts.

### Contacts (minus dividing):

Contacts are analyzed for daughter cells resulting from mitosis (which do not represent true cell-cell contacts) and filtered out accordingly. A pair of daughter cells is detected when two cells in contact have Cell IDs that differ exactly by 1. This requires manual renumbering of known daughter cells to have Cell IDs that differ by 1, and there is a (remote) possibility of missing some true contacts where the Cell IDs happen to differ by 1 regardless of mitosis. Manual spot-checking of contacts is recommended.

### Contacts (minus dead):

Utilizes the "Maximum arrest coefficient" tunable variable to filter out contacts that involve "dead" or non-motile objects based on their Arrest Coefficient.

### Contacts Summary:

A summary of contact history for each individual object. For each object (excluding objects which had no contacts at all), the number of contacts, the total time spent in contact, and the median contact duration are reported. Note that, in the case of cells, this summary comes after filtering out of mitotic contacts and contacts involving "dead" cells.


## Attractors

Attractors are detected by iterating over all objects in the dataset and checking whether any other object is moving towards it. If an object is moving towards another object, it is considered an attraction event and recorded. The results are saved in a separate .xlsx file, and include the timepoint of the attraction event, the distance between the two objects at that timepoint, and the relative speed between the motion of the two objects. The tunable parameters for this function are:

- **Distance threshold**: The maximum distance between two objects at a given timepoint that would be considered an attraction. The program assumes same units as that of the X/Y/Z coordinates given in the input dataset.
- **Approach ratio**: An upper limit for the ratio of the distances between the objects at the end and at the start of a candidate attraction event for it to be recorded.
- **Minimum proximity**: The attracted object must get at least this close to the attractor for at least one timepoint for the attraction event to be recorded.
- **Time persistence**: The minimum number of consecutive timepoints that the attraction event must persist for it to be recorded.
- **Maximum time gap**: The number of consecutive timepoints of increasing distance allowed before the attraction chain is broken.
- **Attractor categories**: A space-separated list of categories of objects that may be considered as attractors. If this field is left blank, all object categories may be considered as attractors.
- **Attracted categories**: A space-separated list of categories of objects that may be considered as attracted. If this field is left blank, all object categories may be considered as attracted.
