# Heatmap
Generating a heatmap using openCV

# Getting Started

We use a virtual environment to do all our development in, this ensures that we don't import modules that accidentally conflict with other project work. 

1. `python3 -m venv venv` - this will create your virtual environment inside a `venv` folder
2. `source venv/bin/activate` - this will activate your virtual env
3. `python3 -m pip install -r requirements.txt`

## Adding new modules

New modules should be added to the virtual env with the following command: `pip install <module-name>`
After the module has been added you should update the `requirements.txt` file using: `pip freeze > requirements.txt`


If VSC is not linting or suggestions aren't working correctly then: 

1. press cmd + shift + p to bring up command palette and search for python: select interpreter
2. Pick then one which is associated with your virtual environment
