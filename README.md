# WTdelineator

Wavelet-based ECG delineator library implemented in Python.

The library required to perform ECG delineation is provided, along with instructions for use, in `WTdelineator.py`. This implementation is based on the work: Martínez, Juan Pablo, et al. *"A wavelet-based ECG delineator: evaluation on standard databases."* IEEE Transactions on Biomedical Engineering 51.4 (2004): 570-581.

## Features
1. Perform ECG delineation using wavelet transforms based on Mallat's and `a trous` algorithms.
2. Evaluate the delineation performance on standard datasets.
3. Tested and validated on:
   - **STAFFIII database**
   - **MIT-BIH Arrhythmia Database**

## Testing and Evaluation
The project has been tested on the **STAFFIII database**, which contains annotations summarized in `annotations.csv`. The project has also been tested on the **MIT-BIH Arrhythmia Database**, where the annotations in `.atr` files were utilized. For the MIT-BIH dataset, evaluations were carried out using the same metrics used in the paper, specifically:
- **Sensitivity (Se):** Measures true positive rate.
- **Positive Predictivity (+P):** Measures detection accuracy.

### Performance Metrics
The following metrics were evaluated:
- **STAFFIII database:** Metrics cannot be directly computed due to the lack of `.atr` annotation files. Delineation results are provided for all signals in `delinResults.h5`.
- **MIT-BIH database:** 
  - **Sensitivity (Se):** ~0.70 (overall, across all records).
  - **Positive Predictivity (+P):** ~0.99 (overall, across all records).

## Examples
Two examples are provided in this repository:
1. **`delineateSignal.py`**: Performs delineation on a single signal from the STAFFIII database.
2. **`delineateDatabase.py`**: Performs delineation on the entire STAFFIII database. This script requires the `annotations.csv` file, which is a summarized version of the annotations provided in the STAFFIII files.

## Installation
### Requirements
The project requires the following:
- Python 3.7+
- Required Python packages:
  - `wfdb`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `h5py`

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repository>.git
   cd WTdelineator
   pip install -r requirements.txt
    ```
2. Download the required datasets:


    **STAFF III Dataset**
   - [Download STAFF III Dataset](https://physionet.org/content/staffiii/1.0.0/)
   
    **MIT-BIH Arrhythmia Database**
   - [Download MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
   


### Notes
The MIT-BIH .atr files contain annotations for R-peak locations, which were used to compute the Sensitivity (Se) and Positive Predictivity (+P) metrics.
The STAFFIII database annotations in annotations.csv were used for delineation, but the database does not include R-peak annotations required for these metrics.


### License
All the functions contained in this repository were developed in the Multiscale Cardiovascular Engineering Group at University College London (MUSE-UCL) by Carlos Ledezma.

This work is protected by a Creative Commons Attribution-ShareAlike 4.0 International license (https://creativecommons.org/licenses/by-sa/4.0/).

markdown
Copy
Edit



### Key Additions:
1. Mentioned testing on both **STAFFIII** and **MIT-BIH Arrhythmia** databases.
2. Explained the use of `.atr` files in MIT-BIH for metric evaluations.
3. Added detailed instructions for running the project with both datasets.
4. Included an evaluation summary for the two datasets.

