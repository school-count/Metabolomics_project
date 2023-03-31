import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

file_name_set = set()
# get unique file names
for results_file_name in glob.glob('./results/*.csv'):
	file_name = os.path.basename(results_file_name)
	file_name_set.add(file_name[:-6])

# create ranking folder
if not os.path.exists('./results/ranking/'):
	os.makedirs('./results/ranking/')

results_folder = './results/ranking/'
for file_name in file_name_set:
	# file paths for data results
	test_l_file_path = './results/' + file_name + '_L.csv'
	test_u_file_path = './results/' + file_name + '_U.csv'
	ranking_file_path = results_folder + file_name + '_ranking.csv'

	# get dfs from all result files
	test_l = pd.read_csv(test_l_file_path)
	test_u = pd.read_csv(test_u_file_path)

	# group by 'cand'
	test_l = test_l.groupby(['cand']).mean()
	test_u = test_u.groupby(['cand']).mean()

	test_l = test_l.rename(columns={"prob": "prob_mean_L"})
	test_u = test_u.rename(columns={"prob": "prob_mean_U"})

	x = test_l['prob_mean_L']
	y = test_u['prob_mean_U']
	plot_df = pd.concat([x, y], axis = 1)
	# calculate the ranking of prob of Labeled candidates / Unlabeled candidates
	plot_df['ranking'] = (plot_df['prob_mean_L'] + 0.01) / (plot_df['prob_mean_U'] + 0.01)
	plot_df = plot_df.drop(plot_df[plot_df['prob_mean_L'] < 0.05].index)
	plot_df = plot_df.sort_values(by=['ranking'], ascending=False)
	plot_df.to_csv(ranking_file_path)
	plot_df = plot_df.dropna()

	# draw dot plot
	sns.scatterplot(data=plot_df, x="prob_mean_L", y="prob_mean_U", hue='ranking', s=80)

	X_plot = np.linspace(0, 0.7)
	Y_plot = X_plot
	plt.plot(X_plot, Y_plot, color='r')
	plt.savefig(results_folder + file_name + '.png')
	plt.clf()