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
We report the first large-scale field demonstration of zone-selective cooling setpoint adjustments for commercial building demand-response. The motivation of this approach is to achieve substantial energy flexibility benefits while reducing actual or perceived risk of temperature-related discomfort and operating within existing hardware and controls. We conduct four experimental trials in each of six operational commercial buildings (~55,000 m2, ~1,000 zones), targeting progressively smaller sets of higher-demanding (“more dominant”) zones. Experiments allow us to characterize the tradeoff between energy savings and the fraction of zones targeted. We identify an empirical 60/20 rule across the studied buildings: approximately 60% of savings were achieved with 20% of zones. Savings were reduced by new zones emerging as dominant but were still significant because new zones were less dominant than the prior set. We also measured little to no temperature increase in zones whose cooling setpoints were not adjusted. By limiting risk of temperature-related discomfort to fewer occupants, selective setpoint adjustments may be more attractive to building owners, managers, and operators.





