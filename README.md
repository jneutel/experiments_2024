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
We report the first large-scale field demonstration of zone-selective cooling setpoint adjustments for commercial building demand response. The motivation of this approach is to achieve energy flexibility benefits from setpoint adjustments while reducing risk and perceived risk of thermal discomfort. We conduct four experimental trials across six commercial buildings (~55,000 m2, ~1,000 zones). Critical zones are always excluded. In one trial, we raise cooling setpoints in all non-critical zones. In subsequent trials, we use previously collected data to identify and modulate progressively smaller sets of higher-demanding (“dominant”) zones. The experiments allow us to characterize the tradeoff between energy savings and the fraction of zones targeted. We identify an empirical 60/20 rule for commercial buildings: approximately 60% of savings can be achieved with setpoint adjustments to the top 20% highest-demanding (most dominant) zones. Savings are reduced by new zones emerging as dominant, but are still significant because new zones are less dominant than the prior. We also measure little to no temperature increase in zones whose cooling setpoints are not directly adjusted. By limiting risk of thermal discomfort to a smaller number of rooms, selective setpoint adjustments may be more palatable to building owners, managers, operators, and occupants.




