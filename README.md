# README

Last Edited: July 17, 2026 (Migrate3D version 3.1)

# Migrate3D

Migrate3D is a Python program that streamlines and automates biological object motion (e.g. cell migration) analysis, returning meaningful metrics that help the user evaluate biological questions. This program does not handle imaging data, only previously generated tracking data, so it is not meant to replace functions already performed very well by programs such as Imaris, Arivis Pro, CellProfiler, TrackMate etc. Migrate3D’s purpose is to take the tracks produced from any of these programs and quickly and easily process the data to generate various metrics of interest in a transparent and tunable fashion, all done through an intuitive graphical user interface (GUI). In addition to motion analysis, Migrate3D can also detect and quantify object-object interactions, such as contacts or attractions.

These results can be used in downstream analyses to compare different conditions, categories of objects, etc. The calculated metrics are all adapted from existing reports in the literature where they were found to have biological significance. Migrate3D not only calculates simple metrics such as track velocity and arrest coefficient, but more complex ones such as straightness index (i.e. confinement ratio), mean squared displacement, turning angles, etc., and includes adjustable constraints and filters to ensure that clean results are produced.

Migrate3D accepts two inputs in CSV format, a Segments file (required) and a Categories file (optional):
- Segments file: Data from object movements through two- or three-dimensional space (i.e. X, Y, and Z coordinates, Time, and Object ID).
- Categories file: Defines object categories (i.e. simply listing the category for each object ID).

After execution, the program will return a set of .xlsx files containing the results of the analysis, plus a set of .html files (and an accompanying .js file) containing interactive graphical plots of the data.

A limitation of the program is that it does not currently handle cell divisions (or fusions) in any intelligent way, so the user needs to separate all such tracks at the split/merge point so that each track only represents one cell. (Note: a record of which daughters belong to which parent cell can easily be kept using a simple numbering system within the Object ID field.)

An example dataset (and Categories file) of computer-generated tracks, generated using the "generate_synthetic_tracks.py" script available in this repository, is provided in the "Example Dataset" folder. Example results generated after a standard or verbose run of Migrate3D on this dataset are provided in the corresponding subfolders.

Migrate3D was developed by Menelaos Symeonides, Emily Mynar, Matthew Kinahan, and Jonah Harris at the University of Vermont, funded by NIH P20GM125498 (to MS) and NIH R21AI152816, NIH R56AI172486, and R01AI172486 (to Markus Thali). We welcome feedback and intend to continue developing and supporting the program as resources allow.

## Input Files

A Segments input file is required to run Migrate3D. Optionally, a Categories input file can be provided to perform additional analyses. In both cases, the program will "guess" which columns contain which data, but if this fails, you can select them through a drop-down box in the GUI. These input files can be stored in any folder.

### Segments

The Segments input file should be a .csv with at least five columns (or at least four for 2D data): object ID, time, X, Y, and Z coordinates. Please ensure that column headers are in the first row of the .csv file input. Note that the Time column is expected to contain a "real" time value (e.g. number of seconds), not just the timepoint index.

If an object has non-consecutive timepoints assigned to it (i.e. if an object's track has gaps), the object will be dropped and not analyzed at all, unless the interpolation option is used. The IDs of dropped objects will be recorded in the results output (along with any objects dropped due to the "Minimum Max. Euclidean" filter) in the sheet "Removed Objects". If interpolation is enabled, any missing timepoints will be linearly interpolated and the object will be used as normal.

### Categories

The Categories input file should be a .csv with object ID and object category. Please ensure that column headers are in the first row of the .csv file input. If no Categories file is imported, a default category ("0") will be assigned to every object, and the machine learning analyses (and anything else done per-category) will not be performed. 

## Installing and Running Migrate3D

These installation instructions involve the use of the command line. If you are not familiar with using the command line, just copy each line and paste into your prompt/terminal and press Enter. Once the process is complete, you will be able to paste in the next line and press Enter, and so on. If "sudo" is used, you will need to enter your account password to proceed.

### On Windows (tested in Windows 10 and 11)

1. First, download and install the latest version of Miniconda3 for Windows using all the default options during installation: https://www.anaconda.com/download/success

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

5. Install the required dependencies:
```powershell
conda install pip
pip install -r requirements.txt
```
Note that these packages are only installed within the conda env you just created and will not affect your system python installation or the base conda env.

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

The output result files will be saved under C:\Users\<your_username>\Migrate3D\Migrate3D-main\.

Note: if you would like to exit the Migrate3D env, i.e. return to the base conda env, simply enter:
```powershell
conda deactivate
```
### On macOS (tested in Tahoe 26.x):

1. Download and install the latest version of Miniconda3 for macOS: https://www.anaconda.com/download/success

2. Open a Terminal. Create a folder for Migrate3D and navigate to it:
```bash
cd ~
mkdir Migrate3D
cd Migrate3D
```

3. Download Migrate3D from GitHub, extract the ZIP file, and navigate into the subfolder that was just created:
```bash
curl -LJO https://github.com/msymeonides/Migrate3D/archive/main/Migrate3D-main.zip
unzip Migrate3D-main.zip
cd Migrate3D-main
```

4. Set up a conda virtual environment and activate it:
```bash
conda create --name Migrate3D python=3.13
conda activate Migrate3D
```

5. Install the required dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
conda install -c conda-forge llvm-openmp
```
Note that these packages are only installed within the conda env you just created and will not affect your system python installation or the base conda env.

6. Finally, to run Migrate3D:
```bash
python ~/Migrate3D/Migrate3D-main/main.py
```
Remember to first activate the Migrate3D venv next time you want to run Migrate3D before executing the main script:
```bash
conda activate Migrate3D
python ~/Migrate3D/Migrate3D-main/main.py
```
In the prompt, you will see a notification that the GUI is now available ("Dash is running on http://127.0.0.1:5555/"). You can now go to this address in your web browser to access the Migrate3D GUI.

The output result files will be saved under /Users/<your_username>/Migrate3D/Migrate3D-main/.

Note: if you would like to exit the Migrate3D env, i.e. return to the base conda env, simply enter:
```bash
conda deactivate
```

### On Linux (tested in Linux Mint 22.1 and Ubuntu 22.04):

It is easiest to do everything in the terminal, so begin by opening a Terminal window.

1. Update your system:
```bash
sudo apt update
sudo apt upgrade
```

2. Install miniconda3, and answer "yes" when prompted with "Proceed with initialization?":
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Close and reopen your Terminal.

3. Download Migrate3D from GitHub, extract the ZIP file, and navigate into the subfolder that was just created:
```bash
cd ~
curl -LJO https://github.com/msymeonides/Migrate3D/archive/main/Migrate3D-main.zip
unzip Migrate3D-main.zip
cd Migrate3D-main
```

4. Set up a conda virtual environment and activate it:
```bash
conda create --name Migrate3D python=3.13
conda activate Migrate3D
```

5. Install the required dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Note that these packages are only installed within the conda env you just created and will not affect your system python installation or the base conda env.

6. Finally, to run Migrate3D:
```bash
cd ~/Migrate3D-main
python main.py
```
Remember to first activate the Migrate3D venv next time you want to run Migrate3D before executing the main script:
```bash
conda activate Migrate3D
cd ~/Migrate3D-main
python main.py
```
In the prompt, you will see a notification that the GUI is now available ("Dash is running on http://127.0.0.1:5555/"). You can now go to this address in your web browser to access the Migrate3D GUI.

The output result files will be saved under /home/<your_username>/Migrate3D-main/.

Note: if you would like to exit the Migrate3D env, i.e. return to the base conda env, simply enter:
```bash
conda deactivate
```

See Migrate3D_extended_README.docx for a detailed description of all run parameters and options, calculations, summary features, and other analyses performed by Migrate3D. 