# 1498-ML-Project: 
## Using Machine Learning to Classify the Cycling Accessibility of Roads

This repo contains a proof-of-concept machine learning classifier, created by Saad Akbar and Keagan Rankin at the University of Toronto, that classifies roads as high-stress or low-stress for cyclists based on a set of features. The classifier attempts to reduce the volume of work required in the popular LTS classification method ([Furth et al. in 2016](https://journals.sagepub.com/doi/pdf/10.3141/2587-06), [Imani et al. 2019](https://journals.sagepub.com/doi/10.3141/2587-06), [Lin et al. 2021](https://findingspress.org/article/19069-the-impact-of-covid-19-cycling-infrastructure-on-low-stress-cycling-accessibility-a-case-study-in-the-city-of-toronto)) the model was trained using open Toronto data and labels [from](https://github.com/lin-bo/Toronto_LTS_network) the cited literature.

## Usage
The final models are saved in all_roads_model.py and all_modes_model.py in the projsubmission folder. These models vary based on input features (see the files 'allmodes_train' and 'centrelinebike_train_spatial' in the train data folder for feature format). Models can be saved as a binary and used for prediciton or improved training on data/labels from other cities.

see _____.txt for dependencies

## Contributions
Contact authors for comments or suggestions.

## References
Furth, P. G., Mekuria, M. C., & Nixon, H. (2016). Network connectivity for low-stress bicycling. Transportation Research Record, 2587(1), 41-49. DOI: https://doi.org/10.3141/2587-06.

Imani, A. F., Miller, E. J., & Saxe, S. (2019). Cycle accessibility and level of traffic stress: A case study of Toronto. Journal of Transport Geography, 80, 102496. DOI: https://doi.org/10.1016/j.jtrangeo.2019.102496.

Lin, B., Chan, T. C., & Saxe, S. (2021). The Impact of COVID-19 Cycling Infrastructure on Low-Stress Cycling Accessibility: A Case Study in the City of Toronto. Findings, 19069. DOI: https://doi.org/10.32866/001c.19069.
