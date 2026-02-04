# cargill_datathon_freight_calculator_chatbot

## Project Summary

This repository provides a freight calculator chatbot for the Cargill Datathon. It combines machine learning, port congestion simulation, and weather prediction to estimate shipping delays and costs for bulk cargo vessels. The project is modular, with components for data generation, ML modeling, weather simulation, and business logic.

---

## Project Structure

```
cargill_datathon_freight_calculator_chatbot/
│
├── data/                      # Raw and processed datasets (BDI, training data, etc.)
├── outputs/                   # Generated results, reports, and scenario CSVs
├── src/
│   ├── ml_port_congestion/    # Port congestion ML model, simulation notebook, models/
│   ├── ml_weather_predictor/  # Weather prediction logic and models
│   ├── freight_calculator.py  # Main business logic for freight calculation
│   ├── congestion_script.py   # Script for running congestion scenarios
│   └── ...                    # Other supporting scripts
├── environment.yml            # Conda environment specification
├── README.md                  # Project documentation (this file)
└── ...
```

---

## Getting Started

### 1. Clone the Repository

```sh
git clone <repo_url>
cd cargill_datathon_freight_calculator_chatbot
```

### 2. Set Up the Conda Environment

Create the environment from the provided `environment.yml`:

```sh
conda env create -f environment.yml
conda activate <env_name>
```
Replace `<env_name>` with the name specified in `environment.yml`.

To update the environment after changes:

```sh
conda env update -f environment.yml --prune
```

### 3. Install Additional Python Packages

If you need a new package (e.g., pandas):

```sh
conda install pandas
# or
pip install pandas
```
After installing, update `environment.yml`:

```sh
conda env export > environment.yml
```

---

## Usage

- **Data Preparation:** Place raw data files (e.g., `bdi_raw.csv`) in the `data` folder. Run preprocessing scripts(ml_port_congestion/bdi_cleaning.py) as needed.
- **Model Training(OPTIONAL):** Use the Jupyter notebook in `src/ml_port_congestion/ml_port_congestion.ipynb` to generate synthetic data, train models, and run simulations. Otherwise, directly running calculator also works as model file already exist.
- **Freight Calculation:** Run `src/freight_calculator.py` for business logic and cost estimation.
- **Scenario Analysis:** Use `src/congestion_script.py` to simulate port congestion scenarios.

---

## Contributing & Extending

- Update `environment.yml` after installing new dependencies.
- Follow the modular structure for adding new models or business logic.
- Document major changes in the README.


---

**For more details on Conda environments, see the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).**