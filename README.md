# cargill_datathon_freight_calculator_chatbot

## Managing Python Libraries and Conda Environment

### 1. Updating Requirements / Libraries in Your Conda Environment

To install a new library (e.g., pandas) in your active conda environment:

```
conda install pandas
```

Or, using pip:

```
pip install pandas
```

After installing new packages, update your `environment.yml` so others can reproduce your environment:

```
conda env export > environment.yml
```

Commit and push the updated `environment.yml` to your repository.

### 2. Creating and Running Your Own Environment Using `environment.yml`

To create a new conda environment from the provided `environment.yml` file:

```
conda env create -f environment.yml
```

To activate the environment:

```
conda activate <env_name>
```
Replace `<env_name>` with the name specified in the `environment.yml` file (usually near the top under `name:`).

To update an existing environment after changes to `environment.yml`:

```
conda env update -f environment.yml --prune
```

---
For more details, see the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).