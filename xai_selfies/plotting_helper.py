import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np


def plot_2D(LEGEND, placeLegend, Xaxes, Yaxes, Xlable, Ylable, SAVEDIR, color, include_line=True, line_style='-'):
    plt.rcParams['font.family'] = 'arial'
    plt.plot(Xaxes, Yaxes, '.', color=color, markersize=10, linestyle=line_style, alpha=0.5)
    plt.xlabel(Xlable, fontsize=18)
    plt.ylabel(Ylable, fontsize=18)
    if include_line:
        plt.plot((np.min(Xaxes),np.max(Xaxes)), (np.min(Yaxes), np.max(Yaxes)), color='gray')
    plt.tick_params(labelsize=16)
    plt.legend(LEGEND, fontsize=16, loc=placeLegend)
    plt.savefig(SAVEDIR, dpi=300, bbox_inches='tight')
    plt.show()

def plot_histogram(dataset1, dataset2, colum_of_interest, x_Label, y_Label, dataset1_name,dataset2_name,  SAVEDIR):
    '''
    Plots a histogram on a property on interest in bins on 10.

    Keyword arguments:
    -- dataset1: first dataset
    -- dataset2: second dataset
    -- colum_of_interest: Colum with the data on which the histogram should be plotted (needs to be present in both datasets).
    -- x_Label: Label which should be displayed on the y-axis.
    -- y_Label: Label which should be displayed on the y-axis.
    -- dataset1_name: name of the first dataset
    -- dataset2_name: name of the second dataset
    -- SAVEDIR: Directory where the plot should be stored.
    '''
    dataset1['data'] = 'dataset1'
    dataset2['data'] = 'dataset2'
    data = pd.concat([dataset1, dataset2], ignore_index=True)

    bins = np.arange(math.floor(min(data[colum_of_interest])), math.ceil(max(data[colum_of_interest])) + 1, 1)

    dataset1_counts, _ = np.histogram(dataset1[colum_of_interest], bins=bins)
    dataset2_counts, _ = np.histogram(dataset2[colum_of_interest], bins=bins)

    plt.rcParams['font.family'] = 'arial'
    bar_width = 0.4  # Width of the bars
    plt.bar(bins[:-1] - bar_width / 2, dataset1_counts, width=bar_width, label=dataset1_name, align='center', color='#A43341')
    plt.bar(bins[:-1] + bar_width / 2, dataset2_counts, width=bar_width, label=dataset2_name, align='center', color='#A43341')

    plt.xlabel(x_Label, fontsize=18)
    plt.ylabel(y_Label, fontsize=18)
    plt.legend(fontsize=16)

    bin_labels = [f'{int(b)}-{int(b + 1)}' for b in bins[:-1]]
    plt.xticks(bins[:-1], labels=bin_labels, rotation=90)  #
    plt.tick_params(labelsize=16)
    plt.xticks(bins[:-1], labels=bin_labels)
    plt.savefig(SAVEDIR, dpi=300, bbox_inches='tight')
    plt.show()

def replace_nan_with_zero(vector_str):
    #vector_str = vector_str.replace('nan', 'np.nan')
    #vector = eval(vector_str, {"__builtins__": None}, {"np": np})
    return [0 if pd.isna(x) else x for x in vector_str]

def get_unmatched_attributions(data, attributions, indices):
    sum_weights = []

    # Loop through both columns and accumulate the sum of the weights for each index
    for idx, weight in zip(data[indices], data[attributions]):
        total_weight = sum(weight[i] for i in idx)  
        sum_weights.append(total_weight)

    return sum_weights

def get_r2_and_summed_data_attributions(data, pred_column_1, pred_column_2, attribution_column_1, attribution_column_2, attribution_type, unmatched_atom_index_column_1, unmatched_atom_index_column_2):
    data['delta_prediction'] = data[pred_column_1] - data[pred_column_2]

    #convert strings to real lists
    data[attribution_column_1 + "_fix"] = data[attribution_column_1].apply(replace_nan_with_zero)
    data[attribution_column_2 + "_fix"] = data[attribution_column_2].apply(replace_nan_with_zero)

    data["sum_" + attribution_column_1] = data[attribution_column_1 + "_fix"].apply(lambda x: sum(x))
    data["sum_" + attribution_column_2] = data[attribution_column_2 + "_fix"].apply(lambda x: sum(x))
    data['delta_sum_' + attribution_type] = data["sum_" + attribution_column_1] - data["sum_" + attribution_column_2]

    r2_whole_mol = np.corrcoef(data['delta_prediction'], data['delta_sum_' + attribution_type])[0,1]**2

    data['sum_unmatched_contributions_1_' + attribution_type] = get_unmatched_attributions(data, attribution_column_1 + "_fix", unmatched_atom_index_column_1)
    data['sum_unmatched_contributions_2_' + attribution_type] = get_unmatched_attributions(data, attribution_column_2 + "_fix", unmatched_atom_index_column_2)

    data['delta_sum_fragment_contributions_' + attribution_type] = data['sum_unmatched_contributions_1_' + attribution_type] - data['sum_unmatched_contributions_2_' + attribution_type]

    r2_fragment = np.corrcoef(data['delta_prediction'], data['delta_sum_fragment_contributions_' + attribution_type])[0,1]**2

    return r2_whole_mol, r2_fragment, data

def get_r2_and_summed_data_attributions_const(data, pred_column_1, pred_column_2, attribution_column_1, attribution_column_2, attribution_type, unmatched_atom_index_column_1, unmatched_atom_index_column_2):
    data['delta_prediction'] = data[pred_column_1] - data[pred_column_2]

    data['sum_constant_contributions_1_' + attribution_type] = get_unmatched_attributions(data, attribution_column_1 + "_fix", unmatched_atom_index_column_1)
    data['sum_constant_contributions_2_' + attribution_type] = get_unmatched_attributions(data, attribution_column_2 + "_fix", unmatched_atom_index_column_2)

    data['delta_sum_const_contributions_' + attribution_type] = data['sum_constant_contributions_1_' + attribution_type] - data['sum_constant_contributions_2_' + attribution_type]

    r2_const = np.corrcoef(data['delta_prediction'], data['delta_sum_const_contributions_' + attribution_type])[0,1]**2

    return r2_const, data