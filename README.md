# README

Last Edited: July 11, 2025 (Migrate3D version 2.1)

# Migrate3D

Migrate3D is a Python program that streamlines and automates biological object motion (e.g. cell migration) analysis, returning meaningful metrics that help the user evaluate biological questions. This program does not handle imaging data, only previously generated tracking data, so it is not meant to replace functions already performed very well by programs such as Imaris, Arivis Pro, CellProfiler, TrackMate etc. Migrate3D’s purpose is to take the tracks produced from any of these programs and quickly and easily process the data to generate various metrics of interest in a transparent and tunable fashion, all done through an intuitive graphical user interface (GUI). In addition to motion analysis, Migrate3D can also detect and quantify object-object interactions, such as contacts or attractions.

These results can be used in downstream analyses to compare different conditions, categories of objects, etc. The calculated metrics are all adapted from existing reports in the literature where they were found to have biological significance. Migrate3D not only calculates simple metrics such as track velocity and arrest coefficient, but more complex ones such as straightness index (i.e. confinement ratio), mean squared displacement, turning angles, etc., and includes adjustable constraints and filters to ensure that clean results are produced.

Migrate3D requires a .csv file input that contains data from object movements through two- or three-dimensional space. After execution, the program will return a set of .xlsx files, each with several worksheets, containing track-by-track summary metrics, mean squared displacement analysis, etc.

If the user imports a Categories .csv file (simply listing the category for each object ID), two additional analyses will be performed: dimensionality reduction using principal component analysis (PCA), and decision tree analysis using XGBoost. There is no guarantee that these will be performed in a statistically-sound way, that is left up to the user to ensure, but the libraries used to perform these analyses are widely used.

A limitation of the program is that it does not currently handle cell divisions (or fusions) in any intelligent way, so the user needs to separate all such tracks at the split/merge point so that each track only represents one cell. (Note: a record of which daughters belong to which parent cell can easily be kept using a simple numbering system within the track’s Name field.)

Migrate3D was developed by Menelaos Symeonides, Emily Mynar, Matthew Kinahan, and Jonah Harris at the University of Vermont, funded by NIH R21-AI152816 and NIH R56-AI172486 (PI: Markus Thali). We welcome feedback and intend to continue developing and supporting the program as resources allow.

## Input Files

A Segments input file is required to run Migrate3D. Optionally, a Categories input file can be provided to perform additional analyses. In both cases, the program will "guess" which columns contain which data, but if this fails, you can select them through a drop-down box in the GUI. These input files can be stored in any folder.

### Segments

The Segments input file should be a .csv with five columns (or four for 2D data): object ID, time, X, Y, and Z coordinates. Please ensure that column headers are in the first row of the .csv file input. Note that the Time column is expected to contain a "real" time value (e.g. number of seconds), not just the timepoint index.

If an object has non-consecutive timepoints assigned to it (i.e. if an object's track has gaps), the object will be dropped and not analyzed at all, unless the interpolation formatting option is used. The IDs of dropped objects will be recorded in the results output (along with any objects dropped due to the "Minimum Max. Euclidean" filter) in the sheet "Removed Objects". If interpolation is enabled, any missing timepoints will be linearly interpolated and the object will be used as normal.

### Categories

The Categories input file should be a .csv with object ID and object category. Please ensure that column headers are in the first row of the .csv file input. If no Categories file is imported, a default category ("0") will be assigned to every object, and the PCA and XGBoost analyses (and anything else done per-category) will not be performed. 

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
In the prompt, you will see a notification that the GUI is now available ("Dash is running on http://127.0.0.1:5555/"). You can now go to this address in your web browser to access the Migrate3D GUI.

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
In the prompt, you will see a notification that the GUI is now available ("Dash is running on http://127.0.0.1:5555/"). You can now go to this address in your web browser to access the Migrate3D GUI.

Note that the output result spreadsheets will be saved under /Users/your_username/Migrate3D/Migrate3D-main/.

### On Linux (tested in Linux Mint 22.1):

It is easiest to do everything in the terminal, so begin by opening a Terminal window.

1. Python 3 is already installed in Linux Mint. Begin by checking the installed version of python:
```powershell
python3 --version
```
On a fresh installation of Linux Mint 22.1, that should return "Python 3.12.3". Python versions 3.13.x and 3.11.x will also work, but earlier versions may not. If you have an earlier version of python, you may need to update it before proceeding.

2. If you have not previously configured a python virtual environment (venv) or installed python packages using pip, you will first need to get set up to do that (if you are already set up for that, skip to step 4).

First, update your system:
```powershell
sudo apt update
sudo apt upgrade
```

Then install some necessary packages (this is all one line, paste the whole thing in one go):
```powershell
sudo apt install build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
libxmlsec1-dev libffi-dev liblzma-dev python3.12-venv
```

Now we need to configure PyEnv to be able to set up a venv:
```powershell
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile
```
Now log out of your Linux account and log back in, then open a new Terminal window. You are now able to create python venvs.

4. Create a dedicated venv:
```powershell
pyenv virtualenv migrate3d
```

5. Download Migrate3D from GitHub and extract the ZIP file into the /home/Migrate3D directory you just created:
```powershell
wget https://github.com/msymeonides/Migrate3D/archive/main/Migrate3D-main.zip
unzip Migrate3D-main.zip -d ~/.pyenv/versions/migrate3d
```

6. You now need to activate the venv:
```powershell
pyenv activate migrate3d
```
Note: if you would like to exit the venv, i.e. return to your normal Linux terminal, simply enter:
```powershell
pyenv deactivate migrate3d
```

7. Install the required dependencies:
```powershell
cd ~/.pyenv/versions/migrate3d/Migrate3D-main
pip install -r requirements.txt
```
Note that these packages are only installed within the venv you just created and will not affect your python installation.

8. Finally, to run Migrate3D:
```powershell
cd ~/.pyenv/versions/migrate3d/Migrate3D-main
python3 main.py
```
Remember to first activate the Migrate3D venv next time you want to run Migrate3D before executing the main script:
```powershell
pyenv activate migrate3d
cd ~/.pyenv/versions/migrate3d/Migrate3D-main
python3 main.py
```
In the prompt, you will see a notification that the GUI is now available ("Dash is running on http://127.0.0.1:5555/"). You can now go to this address in your web browser to access the Migrate3D GUI.

Note that the output result spreadsheets will be saved under ~/.pyenv/versions/migrate3d/Migrate3D-main/. In the file explorer app, you will need to enable "Show hidden files" to see this folder.


## Tunable Parameters

Note that you can change the default values of these variables at the top of the main.py script, or you can set the values for the current run in the GUI.

### Arrest Limit:

A floating point variable that is used to determine whether an object has "really" moved between two timepoints. This parameter is compared to each object’s instantaneous displacement between each pair of consecutive timepoints, and if that value is at least equal to the user’s Arrest Limit input, it will consider the object as moving within that timeframe. It is important to note that even if the instantaneous displacement is below the user’s Arrest Limit input, calculations will still be performed on that timepoint, however if they pass this threshold, they will survive the filter and be reported again in their own column (in the Calculations output in verbose mode).

Set this value by examining tracks of control objects which should not be moving and finding the maximum instantaneous displacement that they exhibit. It is not recommended to set this value to 0 as it is exceedingly unlikely that your imaging system is perfectly stable and all non-zero values of instantaneous displacement represent "biologically true" movements. However, setting this value to 0 will disable this feature, and all Velocity/Acceleration metrics will be reported unfiltered (also, the Arrest Coefficient metric will be omitted entirely).

### Minimum Timepoints:

When a non-zero Arrest Limit has been set, this additional filter is an integer parameter that denotes the total number of 'moving' timepoints an object must exceed to be considered for certain summary features, namely those relating to Velocity and Acceleration. Tracks which fail to meet this threshold (i.e. number of timepoints for which displacement is above the Arrest Limit) will show no value for those summary features, but will still have their own row on the Summary Sheet with values everywhere but those Velocity/Acceleration columns.

This filter is most useful when the dataset contains objects which are not really moving and are outliers in that sense that could still be useful to have in order to provide context for the rest of the dataset. That said, your dataset may already have been filtered down to only objects you are interested in and they are by definition all moving, in which case you can turn this filter off by setting its value to 0.

### Contact Length:

This parameter only applies to the Contacts module (see 'Formatting Options' below). This is a floating-point variable that represents the largest distance between two objects at a given timepoint that will be considered a contact. The program assumes same units as that of the X/Y/Z coordinates given in the input dataset.

To set this value, manually find "true" contacts in your dataset, i.e. pairs of objects that you would consider to be in contact with each other at a given timepoint, measure the magnitude of the vector (that would be a 3D vector for a 3D dataset) between those pairs of centroids, and set the parameter to slightly above the largest of these values.

### Maximum Arrest Coefficient:

This parameter only applies to the Contacts module (see 'Formatting Options' below). This is a floating point variable between 0 and 1 that uses each object's measured Arrest Coefficient to filter out "dead" objects during the contact detection process. This filter is useful if you are only interested in contacts between objects that are both actively moving at some point during the dataset (e.g. live, motile cells).

To turn this filter off, set this value to 1. Note that if Arrest Limit has been set to 0, this parameter will have no effect as all objects will have an Arrest Coefficient of 0 and will survive the "minus dead" filter.

### Minimum Max. Euclidean:

This parameter can be used to filter out objects that do not venture far enough from their origin to be considered "moving" in a meaningful way. This is a floating point variable that represents the minimum value of Maximum Euclidean Distance (see Calculations section below) that an object must have to be included in the analysis. The IDs of dropped objects will be recorded in the results output (along with any objects dropped due having gaps in their tracks) in the sheet "Removed Objects".

## Autodetected Parameters

### Timelapse Interval:

The time between each observation (assumes the same units as the values in the Time column of your dataset). This is automatically detected from the input dataset, but can be manually overridden in the GUI if necessary.

### Maximum Tau Value:

An integer variable that controls the number of intervals to calculate for Mean Squared Displacement, Euclidean Distances, and Turning Angles. This is autodetected from the input dataset by calculating the maximum number of timepoints in all object IDs, but can be manually overridden in the GUI if necessary.


## Formatting options

All of these formatting options are optional, but depending on the nature of your dataset, may be necessary to ensure accurate results.

### Multitracking:

If an object ID is represented by multiple segments at a given timepoint, they will be spatially averaged into one segment.

### Interpolation:

If an object's track has any gaps (i.e. is internally missing one or more timepoints), the coordinates of that timepoint will be inferred by simple linear interpolation and inserted. This will happen with any number of missing timepoints, but this will never add interpolated timepoints before the first or after the last available timepoints for that object ID, i.e. it will not extend the track in either direction but will fill in internal gaps. If this option is not selected, any object with gaps will be dropped from the analysis entirely. The IDs of dropped objects will be recorded in the results output.

### Verbose:

Includes the results of all intermediate step-wise calculations in a separate .xlsx file (_Calculations.xlsx). Due to the size of this output file, enabling verbose mode may significantly increase the time needed for the final "Saving main output" step. This additional output allows the user to see how each metric was calculated for each object at each timepoint.

Also, this enables an additional output file which contains dataset processing information from the Machine Learning function. 

### Contacts:

Identifies contacts between objects at each timepoint, and returns a separate results .xlsx file containing each detected contact as well as a summary of contact history for each object (excluding objects that had no detected contacts), as well as per-category summary features.

### Filter out contacts between objects resulting from cell division?

See the Contacts (minus dividing) section below for more details. Enable this if you have identified daughter cells resulting from cell divisions and have manually set those objects' IDs to be consecutive.

### Attractors:

Identifies instances where an object is attracting other objects towards it (even if both objects are moving), and returns a separate results .xlsx file containing data on each detected attraction event. An additional set of tunable parameters for this function is available in the GUI. The default values for these parameters can be changed at the top of the main.py script.

### Generate Figures:

The following plotly figures are generated:

- **Summary Stats**: Per-category interactive violin plots for each of the summary features, plus the MSD log-log linear fit slope (error bars = 95% confidence interval) for each category.
- **Contacts**: Violin plots of the number of contacts, total time spent in contact, and median contact duration for each category, as well as bar graphs of the percent of cells in each category that have at least 1 or at least 3 contacts.
- **Tracks**: An interactive 2D (X/Y) or 3D (X/Y/Z) plot of all tracks (either raw or origin-zeroed), color-coded by category (if provided). Two versions of this are saved, one which allows filtering by category and one which allows filtering by object. Categories or objects can be toggled on/off by clicking them on the legend.
- **PCA**: A set of plots of only the first four PCs (even if additional PCs are recorded in the .xlsx PCA output) will be generated (1D violin, 2D scatter, and 3D scatter plots of all possible PC combinations).
- **MSD**: A log-log plot of the mean per-category MSD vs. τ, each with its linear fit line (dashed), and, for each category, a plot of all per-track MSD values vs. τ (gray traces), with the mean of all tracks overlaid (dark trace) plus the linear fit (dashed red line). The slope and 95% confidence interval of the linear fit for that category mean is also shown on each figure.

The color used for each category will be consistent across all of these figures. For all violin plots, an inner box plot is overlayed showing the median and upper and lower quartiles. All outputs are in .html format which can be viewed in a browser (note that for large 3D datasets, the tracks figure HTML file can take a while to load once opened, and may be poorly responsive) and all figures can be downloaded in PNG format.

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

Calculation of the angle, θ, between two consecutive vectors, a and b, where each vector spans tau (τ) timepoints. For each value of τ, a moving window of vectors is evaluated from the beginning to the end of the track. This is repeated for all values of τ up to the tunable Maximum Tau Value parameter (see above) or up to half of the maximum number of available timepoints, whichever is smaller.

First, we find the dot product of the two vectors:

$$
a \cdot b = a_xb_x + a_yb_y + a_zb_z
$$

Then we use Pythagoras's theorem to calculate the magnitude of each vector, e.g.:

$$
|a| = \sqrt{{a_x}^2 + {a_y}^2 + {a_z}^2}
$$

And finally, to find the angle θ, we take the inverse cosine of the dot product divided by the two vector magnitudes:

$$
\theta = \cos^{-1} (\frac{a \cdot b}{ |a||b| })
$$

The median angle calculated for each value of τ per object is stored in a worksheet named "Turning Angles", with each row being an object ID, each column being a value of τ, and the values within being the median angle detected for that value of τ.


## Summary Sheet

Summary features calculated using the data acquired over an object’s entire tracking period.

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

This is then multiplied by the square root of the duration of that object’s track. This is helpful when the tracking duration of the objects in the dataset varies.

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

### Velocity and Acceleration metrics:

The mean, median, and standard deviation of all measured values of Instantaneous Velocity and Instantaneous Acceleration for each object. Acceleration is also given as Absolute Acceleration, in which we convert all negative values of Acceleration to positive.

If the Arrest Limit parameter is enabled (i.e. not 0), the values used to calculate these metrics are first filtered down to observations where the displacement exceeds the Arrest Limit. Additionally, if an object fails the Minimum Timepoints filter (see above), it will not have any values for these metrics in the Summary Sheet, but will still have its own row with all other summary features.

### Arrest Coefficient:

A metric used to determine how motile an object is, with values between 0 and 1, where a value of 0 denotes an object that moved constantly throughout its history, and a value of 1 denotes an object that did not move at all throughout its history. The instantaneous displacement threshold used to determine whether an object is moving at a given timepoint is a tunable variable (Arrest limit). If Arrest Limit is set to 0, this metric will not be calculated and the column will be removed from the output file.

$$
Arrest\ Coefficient = \frac{Time\ spent\ arrested}{Duration}
$$

### Median Turning Angle:

The median of the median turning angles for each object across all values of τ. This can be taken as an alternative measure of straightness.

### Overall Euclidean Median:

The median of all Euclidean distances (see "Euclidean Distance" above) calculated for each object across all values of τ. This metric summarizes the typical displacement behavior of an object across multiple time lags. It can serve as a robust central measure that reduces the impact of atypical bursts of movement in an object's history, offering an alternative view to metrics like MSD or Convex Hull Volume that are subject to influence by such extreme movements. It is often highly correlated with those metrics, but during machine learning analysis that will be dealt with due to correlated feature aggregation. 

### Mean Squared Displacement (MSD):

The mean squared displacement over a certain number of time intervals, or tau (τ) values. Used to show that over t time points an object can be expected to move a certain amount. The maximum number of τ values is a tunable variable.

$$
MSD(τ)=\ \lt(x(t + τ) - x(t))^2 \gt 
$$

In the output file, the following result sheets are provided:

- **Mean Squared Displacements**: The MSD at each τ value (columns) for each object (rows).
- **MSD Summary**: The average and standard deviation of the MSD (columns) at each τ value (rows) across the whole dataset.
- **MSD Mean Per Category**: The mean MSD at each τ value (rows) for each object category (columns).
- **MSD StDev Per Category**: the standard deviation of MSD at each τ value (rows) for each object category (columns).
- **MSD Log-Log Fits**: The linear fit parameters for the log-log plot of mean per-category MSD vs. τ. The slope of the line is indicative of the type of motion exhibited by the object, with a slope of 1 indicating diffusion, and a slope of 2 indicating ballistic motion. The "Fit Max. Tau" is the upper limit of the τ values used to calculate the slope (always beginning at τ=1), above which the log-log relationship was judged to deviate from linearity.

### Convex Hull Volume

The volume of a convex hull contained within the track is calculated. Essentially represents how much volume an object covered during its tracking history. Similarly to Straightness, a time correction is applied by multiplying each Convex Hull Volume value by the square root of the duration of that object’s track (helpful when the tracking duration of the objects in the dataset varies). In the case of 2D data, this column will be removed from the output data.

### Maximum MSD

The highest value of MSD that each object reached during its tracking history at any value of τ.


## Machine Learning Analyses

The summary features calculated for each object are used to perform two machine learning analyses: Principal Component Analysis (PCA) and XGBoost (XGB). Before that happens, however, the dataset is processed as follows:

1. Any category filter specified by the user in the GUI will be applied, then any categories containing fewer objects than a set threshold (separate ones for PCA and XGB) will also be removed. These default thresholds (5 for PCA, 10 for XGB) can be adjusted at the top of machine_learning.py. 
2. Any non-moving objects (i.e. those with a Velocity Mean/Median of 0) are removed from the dataset.
3. The dataset is transformed (signed log10 + 1), then z-score scaling is performed. This processing step ensures that all features are on a similar scale and are normally distributed, reducing the impact of outliers.
4. Features with zero variance are removed, and highly-correlated features (i.e. those with a pairwise Pearson correlation coefficient greater than 0.95) are aggregated into a single feature by taking the mean of the (transformed and scaled) values for those features within each object and casting that to the new aggregated feature. The threshold for variance (default = 0.01) and the feature correlation threshold (default = 0.95) can be adjusted at the top of machine_learning.py.

If verbose mode is enabled, the result of each dataset processing step will be saved in a separate .xlsx file. This output also contains all pairwise Pearson correlation coefficients and which aggregated feature any highly-correlated features were aggregated into. A separate verbose output file is generated for PCA and for XGB as the data processing is done separately.

### Principal Component Analysis (PCA):

PCA is performed on the summary features of each object, and the results are saved in a separate .xlsx file. The minimum number of principal components (PCs) needed to explain at least 95% of the variance is determined, and that is the number of PCs that will end up shown in the output file. A Kruskal-Wallis test is performed on the PCA results to determine whether each PC is significantly different between the provided categories of objects. Additionally, p-values resulting from post-hoc comparisons (with Holm-Bonferroni correction) between each category for each PC are provided.

### XGBoost (XGB):

XGBoost is a decision tree-based machine learning algorithm that can reveal which motion parameters are most important for describing the variation in a dataset. This is performed using the entire dataset ('Full Dataset' sheets), as well as for all possible pairs of categories ('Comparison X' sheets). The output .XLSX file contains the following two sheets for each of these analyses:
* **Features**: This sheet lists how important each feature was for the model that was determined to be the best at describing the variance in the data. For category-to-category comparisons, the categories being compared are listed below the features table. 
  * **Categories**: Lists which categories were analyzed in this sheet.
  * **Method**: Lists the method used to train the model. This will be either "Train-Test Split" or "K-Fold CV" (for Stratified K-Fold Cross-Validation). Train-Test Split is the default method (using a 60%-40% training/testing split, respectively), but if the number of objects in one of the categories in a comparison is too low, the K-fold CV method will be used instead. Refer to XGBoost documentation for explanations on these methods. The thresholds for the minimum number of objects in a category to use Train-Test Split (default = 20) or to still use XGB but with K-Fold CV (default = 10), as well as the proportion of the dataset to use for testing when using the Train-Test Split method (default = 0.4), can be adjusted at the top of machine_learning.py.

* **Report**: This contains a confusion matrix which documents how well this model performed in classifying objects into their respective categories. Each row corresponds to a category included in that comparison (with the leftmost value in the row being the category's name), and the confusion matrix includes the following columns:
  * **Precision**: The proportion of true positive classifications out of all positive classifications made by the model for that category.
  * **Recall**: The proportion of true positive classifications out of all actual objects in that category.
  * **F1 Score**: The harmonic mean of Precision and Recall, a measure of the model's accuracy for that category.
  * **Support**: The number of objects in that category that were included in the analysis.
  * **Accuracy**: Below each confusion matrix, the overall accuracy of the model is reported, which is the proportion of all objects that were correctly classified into their respective categories.
  * **Method**: Either "Train-Test Split" or "K-Fold CV", as explained above.

Note that if only two categories are present in the dataset, only the "Full Dataset" analysis will be performed, as the only possible category-to-category comparison is identical to analyzing the full dataset.

## Contacts

Contacts will iterate over all the objects in the dataset comparing their X, Y, and Z coordinates at each timepoint. If two objects are closer in these dimensions than the "Contact Length" tunable variable, it will be recorded as a contact. This option is most relevant to analyses of cell migration, where contacts between migrating cells are of interest.

### Contacts (minus dividing):

Contacts are analyzed for daughter cells resulting from mitosis (which do not represent true cell-cell contacts) and filtered out accordingly. A pair of daughter cells is detected when two cells in contact have Object IDs that differ exactly by 1. This requires manual renumbering of known daughter cells to have Object IDs that differ by 1, and there is a (remote) possibility of missing some true contacts where the Object IDs happen to differ by 1 but were not daughter cells resulting from the same mitosis. Manual spot-checking of contacts is recommended.

This filter can be disabled in the GUI, in which case this results sheet will be ommitted.

### Contacts (minus dead):

Utilizes the "Maximum Arrest Coefficient" tunable parameter to filter out contacts that involve "dead" or non-motile cells based on their Arrest Coefficient. This filter is applied after the Contacts (minus dividing) filter, unless that filter is disabled, in which case it is applied to the total Contacts dataset.

### Contacts Summary:

A summary of contact history for each individual object. For each object (excluding objects which had no contacts at all), the number of contacts, the total time spent in contact, and the median contact duration are reported. Note that this summary comes after filtering out contacts involving dividing cells or "dead" (non-moving) cells. To summarize all possible contacts detected, do not enable the dividing filter and set Maximum Arrest Coefficient to 1.

### Contacts Per Category:

A per-category analysis of contacts, including the number of objects in each category that had at least one or at least three contacts, the median number of contacts per object, the median total time each object spent in contact, and the median duration of each contact event.


## Attractors

Attractors are detected by iterating over all objects in the dataset and checking whether any other object is moving towards it. If an object is moving towards another object, it is considered an attraction event and recorded. The results are saved in a separate .xlsx file, and include the timepoint of the attraction event, the distance between the two objects at that timepoint, and the relative speed between the motion of the two objects. The tunable parameters for this function are:

- **Distance threshold**: The maximum distance between two objects at a given timepoint that would be considered an attraction. The program assumes same units as that of the X/Y/Z coordinates given in the input dataset.
- **Approach ratio**: An upper limit for the ratio of the distances between the objects at the end and at the start of a candidate attraction event for it to be recorded.
- **Minimum proximity**: The attracted object must get at least this close to the attractor for at least one timepoint for the attraction event to be recorded.
- **Time persistence**: The minimum number of consecutive timepoints that the attraction event must persist for it to be recorded.
- **Maximum time gap**: The number of consecutive timepoints of increasing distance allowed before the attraction chain is broken.
- **Attractor categories**: A space-separated list of categories of objects that may be considered as attractors. If this field is left blank, all object categories may be considered as attractors.
- **Attracted categories**: A space-separated list of categories of objects that may be considered as attracted. If this field is left blank, all object categories may be considered as attracted.
