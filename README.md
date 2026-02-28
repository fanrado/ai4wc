# ai4wc

ai4wc is a python package gathering deep learning models for LArTPC data, developed in teh context of Wire-Cell (WC).

## Installation
### Create a python virtual environment
```bash
python3 -m venv env_ai4wc
source env_ai4wc/bin/activate
```

### Install ai4wc : edition mode
```bash
cd path/to/ai4wc
python3 -m pip install -e .
```
### Install ai4wc : standard mode
```bash
cd path/to/ai4wc
python3 -m pip install .
```

## Usage
```python
import ai4wc
```

## Contributing
Feel free to fork the repository and submit pull requests.

--------------------------------------------------------

# To do list
- [ ]  ai4wc source code for vision transformer model
    - [x] data preparation
    - [x] model definition
    - [x] training script : supervised learning using neutrino events
    - [x] evaluation script
    - [ ] utility functions for optimization and visualization
- [ ] tests
    - [ ] extraction of an attention map from a vision transformer model
- [ ] documentation
    - [ ] docstrings
