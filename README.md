\# Jambi Wind Correction MLOps



Proyek Machine Learning Operations (MLOps) untuk melakukan post-processing dan koreksi prediksi kecepatan angin jangka pendek ECMWF IFS di Kota Jambi menggunakan machine learning.



\## Project Overview



ECMWF IFS digunakan sebagai sumber raw forecast, sedangkan ECMWF IFS Analysis digunakan sebagai reference.



Pendekatan machine learning digunakan untuk memprediksi residual:



```text

residual = reference\_wind\_speed - forecast\_wind\_speed

```



Prediksi kecepatan angin yang telah dikoreksi kemudian dihitung sebagai:



```text

corrected\_wind = forecast\_wind + predicted\_residual

```



\## Forecast Horizons



Model lead yang digunakan:



\- +6 jam

\- +12 jam

\- +18 jam

\- +24 jam



\## Project Structure



```text

jambi-wind-correction-mlops/

├── data/

│   ├── raw/

│   ├── processed/

│   └── reports/

├── src/

│   ├── data/

│   │   ├── test\_single\_run.py

│   │   ├── test\_reference.py

│   │   └── build\_poc\_dataset.py

│   └── models/

│       └── sanity\_residual\_correction.py

├── .gitignore

├── requirements.txt

└── README.md

```



\## Setup



Create virtual environment:



```powershell

python -m venv .venv

```



Activate the environment:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

pip install -r requirements.txt

```



\## Run Proof-of-Concept



Validate ECMWF IFS Single Runs:



```powershell

python .\\src\\data\\test\_single\_run.py

```



Validate ECMWF IFS Analysis reference:



```powershell

python .\\src\\data\\test\_reference.py

```



Build the forecast-reference dataset:



```powershell

python .\\src\\data\\build\_poc\_dataset.py

```



Run the temporal residual-correction sanity test:



```powershell

python .\\src\\models\\sanity\_residual\_correction.py

```



\## Current Status



LK-01 / project initiation and proof-of-concept:



\- ECMWF IFS Single Runs validated

\- ECMWF IFS Analysis reference validated

\- Forecast-reference pairing validated

\- Data quality validation implemented

\- Raw ECMWF baseline evaluated

\- Temporal residual-correction sanity test completed



Further MLOps components such as data versioning, experiment tracking, orchestration, model serving, monitoring, and continuous training will be implemented progressively in subsequent project stages.

