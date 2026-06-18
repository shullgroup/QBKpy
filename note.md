# Suggestions

- Copies of the are testing version of those in the root.
- We can use readme as docs to explain how to use the codes. If we want to have detailed explainations for each test method, I suggest to have a 'docs' folder and instructions for each method. And then, the README.md file can be used for general information.


## Structure for current stage

    QKBPy/
    ├── examples/
    │   ├── data/               # 
    │   ├── figure/             # 
    │   └── python/             #  
    ├── DMA.py                  # 
    ├── DSC.py                  # 
    ├── QCM.py                  # 
    ├── SWE.py                  # 
    ├── TGA.py                  # 
    ├── fracture.py             # 
    ├── utils.py                # General helpers (to keep the structure simple)
    ├── tests/                  # Codes under test
    ├── requirements.txt        # packages required for running the codes
    ├── README.md               # Project overview, and instructions
    ├── LICENSE                 # 
    └── .gitignore              # 

For current daily use, everyone can keep syncing this repo and add the local directory to the code to import the analysis code.

## Structure for future Conda/PIP integration 

    QKBPy/
    ├── examples/
    │   ├── data/               # 
    │   ├── figure/             # 
    │   └── python/             #  
    ├── src/                    # Core analysis
    │   ├── __init__.py
    │   ├── data_loader.py      # 
    │   ├── (physics_models).py   # 
    │   └── utils.py            # General helpers 
    ├── tests/                  # Codes under test
    ├── requirements.txt        # packages required for running the codes
    ├── environment.yml         # Conda environment file for reproducibility
    ├── README.md               # Project overview, and instructions
    └── .gitignore              # 
