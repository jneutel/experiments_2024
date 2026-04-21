# experiments_2024
This repository was created to replicate results found in "Targeted cooling setpoint adjustments in “dominant zones” for demand response in commercial buildings" J. Neutel, JA. de Chalendar, SM. Benson.

# Installation and use 
This repo can be installed by cloning the repo onto your machine, and then in the same folder as the ```pyproject.toml``` file running the terminal command ```pip install -e .```. This command will also automatically install the required packages for this repo, and thus we recommend running it in a new virtual environment of some kind (for example, using ```conda```). 

You will also need to update the paths in ```experiments_2024/src/experiments_2024/paths.py```

Data for this work can be found here: ```https://drive.google.com/drive/u/0/folders/1u80Pmupr2ECjsFWx4UGN7ByNN7XN2fA0```. You will need to download this data and place it where your ```DATASETS_PATH``` is pointing to. 

After completing these steps, results can be replicated by running provided notebooks. 

# Citation
To come after the review process. 

# Abstract
In this work, we report the first large-scale field demonstration of zone-selective cooling setpoint adjustments for commercial building demand response. The motivation of this approach is to achieve energy flexibility benefits from setpoint adjustments while reducing risk or perceived risk of thermal discomfort. We conduct four experimental trials across six commercial buildings (~55,000 m2 of floor space, ~1,000 zones). These include one trial where setpoints are increased in all non-critical zones, followed by three trials in which cooling setpoints are increased in progressively smaller sets of increasingly higher-demanding (“more dominant”) zones. The experiments allow us to characterize the tradeoff between energy savings and the fraction of zones targeted, and to examine interzonal dynamics when only a subset of zones is adjusted. We empirically find a 60/20 rule for commercial buildings: approximately 60% of savings can be achieved with setpoint adjustments to the top 20% highest-demanding (most dominant) zones. We also measure little to no temperature increase in zones whose cooling setpoints are not directly adjusted. By limiting risk of thermal discomfort to a smaller number of rooms, selective setpoint adjustments may be more palatable to building owners, managers, operators, and occupants.



