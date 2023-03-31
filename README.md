# Metabolomics_project
This project is to identify the potential labeled metabolites in human biofluid. 

There are 3 files in total:
1. First run python3 data_preprocessing_pipeline.py is to preprocess the training (plant) data and testing (human) data. Both the original data and the preprocessed data are under 'data' folder.
2. Then run python3 rf_classifier.py is to train the classifier and get the predicted probability ranking results for each Labeled and Unlabeled human candidate metabolite. The trained classifier is stored as 'classifier.pkl', and the ranking results are under 'results' folder.
3. At last, run python3 results_plot.py is to plot the ranking of each human candidate metabolite, where darker dots represent higher ranking. We want to focus on the darker dots under the red diagonal line. The ranking results and plots of Labeled / Unlabeled are under '/results/ranking/' folder.
