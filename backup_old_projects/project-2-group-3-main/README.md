# New Document[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/46o-wT5u)

# Project 2

### Nicole's Branch includes:
* Separating US and Japan data in the csv file (folder name: separating_data)
* Handling missing values (folder name: handle_missing_data)
* Binning (folder name: binning --> equi_depth_bin.py)
* Clustering (folder name: clustering) --> (3 types: k-means, db_scan and GMM) 
* Run the main_file.py (this file is connected to all other folders and will help you generate all the plots and csv files for the project)

<p></p>

How to run the above parts?
* Directly run the main_file.py (this should execute all the above mentioned part)

<p></p>

* Following are the Plots (plots are present in the plot folder) and new datasets (these are present in the current working directory) generated:
  * Dataset for separate data files: (us_data.csv and japan_data.csv)
  * Missing data issue solved: (us_solved_missing_data.csv, japan_solved_missing_data.csv)
  * Binning data and graph: (US_binned_data.csv, binning.png)
  * Clustering: (k_means_clustering.png, db_scan.png, GMM.png)


<p></p>

* Libraries
  * matplotlib
  * numpy
  * pandas
  * sklearn

    
### Structrue
- `scripts`
- `datasets`
- `scraping`
- `text_analysis`
- `codes.txt` - Some important feature names for reference

### Dhruv's (dhruvramani) Branch Includes ###
```
pip3 install gnews pandas sklearn nltk afinn vaderSentiment
```
+ Script to download news data, more relevant to our usecase (in `./scraping`)
+ Sentiment analysis (`./text_analysis`)
+ New data collected and downloaded from the script. Linking to <a href="https://drive.google.com/file/d/1hckgfcNo56fsqHkdq94KFOKojtIJP-ym/view?usp=sharing">additional data</a> in Google Drive.

### Dhiraj (dhiraj-branch) Branch Includes ###
- Code for LOF, Clustering and general data cleaning.
- Install Packages - `pip3 install pandas matplotlib scikit-learn`
- Steps to run
  - `cd scripts`
  - `python3 analysis-dhiraj.py`
  - Plots will be saved in the `./plot` folder and cleaned datasets to `./datasets`.
- Includes Final Project 2 report.


### Zhixuan's (Zhixuan-branch) Branch Includes ###
```
pip3 install matlotlib pandas 
```
+ Using the new_combined_clean.CSV in the datasets folder to draw the histogram and scatter plot
+ The code is in the plot folder, and the plots are also saved in the same folder.
+ Draw plots: python .\plot.py
  
<details>
  <summary><h3>Project 1</h3></summary>

 ### Dhruv's (dhruvramani) Branch Includes ### 
+  script to download data using <a href="https://fred.stlouisfed.org/">Federal Reserve Economic Data (FRED)</a>'s APIs to get the exchange rate of Japanese Yen wrt US Dollar. We also get common related economic indicators of both the countries.
+ Data downloaded from this script
+ Common economic indicators of various countries


Nicole's (nicole-mathias) Branch
+ Script for scraping historical events (year, event) from <a href="https://www.bbc.com/news">BBC NEWS<a/> to get data for US and Japan.

Zhixuan's (Zhixuan-branch) Branch Includes:
+ Downloaded stock market data from Yahoo Finance
+ Python version: 3.6
+ Packages : pyfredapi
+ ```
  pip3 install pyfredapi pandas
  python3 fred_data.py
  ```

### Nicole's (nicole-mathias) Branch ###
+ Script for scraping historical events (year, event) from <a href="https://www.bbc.com/news">BBC NEWS<a/> to get data for US and Japan.

<p></p>

***Instructions to run the script***
+ The python file contains two url links at the end (a) url for US (b) url for Japan, either of which can be select based on the need.
+ The function "bbc_scraper()" contains code to scrape the website and then saves the data into a csv file. The csv file needs to be renamed for each different country. eg. US_data.csv and Japan_data.csv.
+ Running the file: _python scraping.py_

***Packages used***
+ Python version: 3.8
+ Beautiful soup: 4

### Zhixuan's (Zhixuan-branch) Branch Includes: ###
+ Downloaded stock market data from Yahoo Finance

### Dhiraj's (dhiraj-branch) Branch ###
+ Dataset for US spendings and public debt held from source <a href="https://fiscaldata.treasury.gov/datasets/debt-to-the-penny/debt-to-the-penny"> [LINK] </a>
+ Project Report

</detail>
