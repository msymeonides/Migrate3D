# README

Last Edited: January 15, 2024 (Migrate3D version 1.5)

# Migrate3D

Migrate3D is a Python program that streamlines and automates cell migration analysis, returning meaningful metrics that provide insight into cell migration to help the user evaluate biological questions. This program does not handle imaging data, only previously generated tracking data, so it is not meant to replace functions already performed very well by programs such as Imaris, Vision4D, CellProfiler, TrackMate etc. Migrate3D’s purpose is to take the tracks produced from any of these programs and quickly and easily process the data to generate various metrics of interest in a transparent and tunable fashion.

These results can be used in downstream analyses to compare different conditions, different cell subpopulations, etc. The calculated metrics are all adapted from existing reports in the literature where they were found to have biological significance. Migrate3D not only calculates simple metrics such as track velocity and arrest coefficient, but more complex ones such as straightness index (i.e. confinement ratio), mean squared displacement across selectable time lags, relative turn angles, etc., and includes adjustable constraints and filters to ensure that clean results are produced.

Migrate3D requires a .csv file input that contains data from cell movements through two- or three-dimensional space. Each row should include a unique cell identifier (ID), time, and X/Y/Z coordinates. While complete/uninterrupted tracks are ideal, the program can interpolate missing data if needed, as long as the different segments of the track belong to the same unique cell ID. A key limitation of the program is that it does not currently handle cell divisions (or fusions) in any intelligent way, so the user needs to separate all such tracks at the split/merge point so that each track only represents one cell. (Note: a record of which daughters belong to which parent cell can easily be kept using a simple numbering system within the track’s Name field.)

Migrate3D has formatting functionality. If selected by the user, the program can account for multi-tracked timepoints, interpolate missing data points, and adjust for two-dimensional data. **Formatting functions do not alter original datafile** and the reformatted data will be part of the results output.

After execution, the program will return a .xlsx file with several worksheets, containing stepwise calculations done for each timepoint of each track, track-by-track summary metrics, and mean squared displacement analysis (both track-by-track and summary statistics). If the option is enabled, a separate .xlsx file will be returned with analysis of cell-cell contacts. Additionally, another .xlsx output will be generated if Principal Component Analysis (PCA) is performed. This will include detailed outputs, which can be further enhanced by providing categorization of the data as .csv file (simply listing the category for each cell ID). If a categorization file is given, the PCA results will be returned in their own .xlsx file. There is no guarantee that the PCA is performed in a statistically-sound way, that is left up to the user to ensure, but the library used to perform the PCA (sklearn) is widely used.

Migrate3D was created with ease of use in mind, to that end a graphical user interface (GUI) was implemented. This includes easy file open dialogs, user-adjustable parameters, and a progress bar. We welcome feedback and intend to support the program as resources allow.

Migrate3D was developed by Matthew Kinahan, Emily Mynar, and Menelaos Symeonides at the University of Vermont, funded by NIH R21-AI152816 and NIH R56-AI172486 (PI: Markus Thali).

## Input Files

Place your .csv input dataset in the Migrate3D folder you create during installation to make it easier to find in the GUI.

### Segments
Segments input files should be a .csv with cell ID, time, X, Y, and Z coordinates. Please ensure that column headers are in the first row of the .csv file input.

### Categories
Categories input files should be a .csv with cell ID and cell category (No categories file is necessary to run the program).
Please ensure that column headers are in the first row of the .csv file input. 

## Installing and Running Migrate3D

These installation instructions involve the use of the command line. If you are not familiar with using the command line, just copy each line and paste into your prompt/terminal and press Enter. Once the process is complete, you will be able to paste in the next line and press Enter, and so on. If "sudo" is used, you will need to enter your account password to proceed.

### On Windows (tested in Windows 11)

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
On a fresh installation of Ubuntu 23.10, that should return "Python 3.11.6".

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


## Tunable Variables

### Timelapse Interval: (Default value = 4) 

The time between each observation (assumes the same units as the values in the Time column of your dataset).

### Arrest Limit: (Default value = 3)

This parameter is a floating point variable that is used to determine whether a cell has "really" moved between two timepoints. This parameter is compared to each cell’s instantaneous displacement between each pair of consecutive timepoints, and if that value is at least equal to the user’s Arrest Limit input, it will consider the cell as moving within that timeframe. It is important to note that even if the instantaneous displacement is below the user’s Arrest Limit input, calculations will still be performed, however if they pass this threshold, they will survive the filter and be reported again in their own column. Set this value by examining tracks of dead cells (which should only "wobble" and never actively move) and finding the maximum instantaneous displacement that they exhibit. It is not recommended to set this value to 0 as it is exceedingly unlikely that your imaging system is perfectly stable and all non-zero values of instantaneous displacement represent "biologically true" movements.

### Minimum Timepoints Moving: (Default value = 4)

Minimum Timepoints Moving is an integer parameter that denotes the amount of moving timepoints a cell must exceed to be considered for certain summary statistics, namely those relating to Velocity and Acceleration that are denoted as "Filtered". Tracks which fail to meet this threshold will show no value for those summary statistics, but will still have their own row on the Summary Sheet with values everywhere but those Filtered Velocity/Acceleration columns. To turn this filter off, set this value to 0.

### Maximum Contact Length: (Default value = 12)

Maximum contact length is a floating-point variable that represents the largest distance between two cells at a given timepoint that would be considered a contact. The program assumes same units as that of the X/Y/Z coordinates given in the input dataset.

### Arrested/Dead: (Default value = 0.95)

A floating point variable between 0 and 1 that uses each cell's Arrest Coefficient to filter out "dead" cells during the contact detection process. If a cell's Arrest Coefficient is below this user specified input, it will be considered a live cell, and will be included in contacts between live cells. To turn this filter off, set this value to 1.

### Tau Value (MSD): (Default value = 50)

This Tau value is an integer variable that controls the number of Mean Squared Displacement intervals to calculate. It is recommended to set this value to a number equal to the total number of timepoints that the majority of the tracks in the dataset have.

### Tau Value (Euclidean & Angle): (Default value = 25)

This Tau value is an integer variable that controls the range of intervals that Euclidean Distance and Turning Angle calculations will be performed on. It is recommended to set this value to half the number of the MSD Tau Value.

### Output Filename: (Default = Migrate3D_Results)

Enter a name for your output file. The .xlsx extension will be added on, do not include it in this field. Note that any output file with the same name present in the program folder will be overwritten. Additionally, if a file with the same name is currently open in Excel, this will cause Migrate3D to crash.

### Column Header Names (Segments file):

Enter the name of each column in your input dataset to help Migrate3D find your Cell ID, Time, and X/Y/Z coordinate data correctly. Any additional columns will simply be ignored. You can just open your .csv, copy each header name, and paste it in the appropriate field. If your output always looks the same, you can edit the script to change the default entries. NOTE: Please ensure that column headers are in the first row of the .csv file input.

### Column Header Names (Categories file):

The last two column header name fields are only relevant if you are including a Categories file which associates each Cell ID with a cell category (which can be any string or value), which factors into the MSD Summary Sheet and the PCA calculations. If no Categories file is provided, these fields are ignored.

### PCA Category Limit:

In case your dataset includes several categories of cells, but you are only interested in including certain ones for PCA, enter those here separated by commas. The entire rest of the analysis will still be done on every cell ID in the dataset, only PCA will be subsetted. You may want to do this to exclude known dead cell tracks which would contaminate the PCA with nonsense values. E.g. if you have your cells categorized as 1 through 5, and dead cells are category 1, you can exclude them by entering the following in this field: 2,3,4,5

## Calculations

Step-wise calculations generated for each cell iteratively over each time point.

### Instantaneous Displacement:

The square root of the sum of the squared difference between X, Y, and Z coordinates over one timepoint difference. Used to show how much a cell has moved over one time interval.

$$
Instantaneous \ Displacement =\sqrt{(x_{t} - x_{t-1})^{2} + (y_{t} - y_{t-1})^{2} +(z_{t} - z_{t-1})^{2}}
$$

### Total Displacement:

The square root of the sum of the squared difference between X, Y, and Z coordinates of each timepoint compared to timepoint 0 for that cell. Measures how far a cell has moved from its origin at any given timepoint.

$$
Total \ Displacement =\sqrt{(x_{t} - x_{t[0]})^{2} + (y_{t} - y_{t[0]})^{2} + (z_{t} - z_{t[0]})^{2}}
$$

### Instantaneous Velocity:

A cell’s velocity when considering just one time interval. 

$$
Instantaneous \ Velocity =  \frac{Instantaneous \ Displacement \ _{t}}{Time \ lag} 
$$

### Instantaneous Acceleration:

A cell’s acceleration when considering just two consecutive time intervals.

$$
Instantaneous \ Acceleration=\frac {Instantaneous \ Velocity \ _{t} \ - \ Instantaneous \ Velocity \ _{t-1}}{Time \ lag} 
$$

### Euclidean Distance:

The Euclidean distance over n timepoints. Finds the length of a line segment between a cell’s coordinates over n time points.

$$
Euclidean \ Distance =\sqrt{(x_{t} - x_{t-n})^{2} + (y_{t} - y_{t-n})^{2} + (z_{t} - z_{t-n})^{2}}
$$

### Turning Angle:

The angle, θ, between two consecutive vectors, a and b, with a given timepoint interval, is calculated by first finding the dot product of the two vectors:

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

## Summary Sheet

Summary statistics using the data acquired over a cell’s entire tracking period.

### Final Euclidean Distance:

The total displacement from a cell’s final recorded time point to the cell’s origin.

### Maximum Euclidean Distance:

The furthest from its origin that a cell was recorded during its entire history.

### Path Length:

The total path length of a cell. It is the sum of all instantaneous displacements. Shows how far a cell has traveled along its path. 

$$
Path \ Length = \sum_{}^{} Instantaneous \ Displacement
$$

### Straightness:

The Final Euclidean Distance divided by the Path Length. A metric of how confined a cell is, or how straight its track was.

$$
Straightness=\frac{Final\ Euclidean}{Path\ Length}
$$

### Time Corrected Straightness:

Straightness multiplied by the square root of the duration of that cell’s track. Needed when the tracking duration of the cells in the dataset varies.

$$
Time\ Corrected\ Straightness=Straightness\times\sqrt{Duration}
$$

### Displacement Ratio:

The Final Euclidean Distance divided by the Maximum Euclidean Distance, giving values between 0 and 1. A high Displacement Ratio value (approaching 1) denotes a cell which did not return back towards its origin at the end of its track. Conversely, a low value (approaching 0) denotes a cell which ventured away from its origin at some point, but by the end of the track returned closer to its origin.

$$
Displacement \ Ratio = \frac{Final \ Euclidean \ Distance}{Maximum\ Euclidean\ Distance}
$$

### Outreach Ratio:

The Maximum Euclidean Distance divided by the Path Length, giving values between 0 and 1. A high Outreach Ratio value (approaching 1) denotes a cell which essentially moves in a near-perfect straight line away from its origin. Conversely, a low value (approaching 0) denotes a cell which takes a very tortuous or meandering path.

$$
Outreach\ Ratio =\frac{Maximum\ Euclidean\ Distance}{Path\ Length}
$$

### Arrest Coefficient:

A metric used to determine how motile a cell is, with values between 0 and 1, where a value of 0 denotes a cell that moved throughout its history, and a value of 1 denotes a cell that did not move at all throughout its history. The instantaneous displacement threshold used to determine whether a cell is moving at a given timepoint is a tunable variable.

$$
Arrest\ Coefficient = \frac{Time\ spent\ arrested}{Duration}
$$

### Mean Squared Displacement (MSD):

The mean squared displacement over a certain number of time lags, or tau (τ) values. Used to show that over t time points a cell can be expected to move a certain amount. The maximum number of τ values is a tunable variable.

$$
MSD(τ)=\ \lt(x(t + τ) - x(t))^2 \gt 
$$

MSD Summary Sheets are also provided with the average and standard deviation of the MSD at each τ value, either across the whole dataset or for each cell category (if provided). These can be used to plot MSD log-log plots and evaluate whether a category of cell is moving with a certain pattern. 

### Convex Hull Volume

The volume of a convex hull contained within the track is calculated. Essentially represents how much volume a cell covered during its tracking history. Similarly to Straightness, a Time-Corrected value is also provided by multiplying each Convex Hull Volume value by the square root of the duration of that cell’s track (needed when the tracking duration of the cells in the dataset varies).

## Contacts

Contacts will iterate over all the cells in the dataset comparing their X, Y, and Z coordinates at each timepoint. If two cells are closer in these dimensions than the Maximum Contact Length variable, it will be recorded as a contact. This option reports all contacts, contacts that are not mitotic, contacts that are alive, and a summary of each cell’s contacts.

### Contacts no Mitosis:

Contacts are analyzed for daughter cells resulting from mitosis (which do not represent true cell-cell contacts) and filtered out accordingly. A pair of daughter cells is detected when two cells in contact have Cell IDs that differ exactly by 1. This requires manual renumbering of known daughter cells to have Cell IDs that differ by 1, and there is a (remote) possibility of missing some true contacts where the Cell IDs happen to differ by 1 regardless of mitosis. Manual spot-checking of contacts is recommended.

### Contacts no Dead:

Utilizes the Arrested/Dead variable to filter out contacts that involve "dead" or non-motile cells based on their Arrest Coefficient.

### Contact Summary:

A summary of contact history for each individual cell. For each cell (excluding cells which had no contacts at all), the number of contacts, the total time spent in contact, and the median contact duration are reported. Note that this summary comes after filtering out of mitotic contacts and contacts involving "dead" cells.
