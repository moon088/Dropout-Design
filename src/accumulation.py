from __future__ import print_function

import argparse
import csv
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy as sp

def read(file):
    with open(file, 'r') as f:
        print(''.join(['read ', str(f)]))
        reader = csv.reader(f)
        for r in reader:
            yield r

def remove(path):
    for f in glob.glob(path):
        print(''.join(['removed ', str(f)]))
        os.remove(f)

def write(file, data):
    with open(file, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        for v in data:
            writer.writerow(v)
    print(''.join(['saved ', str(file)]))

def summarize(dt):
    return np.average(dt), np.var(dt), np.std(dt), sp.median(dt)

def accumulate_test():
    src_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_path)
    models_path = os.path.abspath(src_path + '/models')
    
    info_path = models_path + '/info.csv'
    info = [row for row in read(info_path)]
    print(info)

    # collect measured data
    collection = {}
    exp_info = set(tuple(e[:-1]) for e in info)
    for i in exp_info:
        p = '_'.join(['evaluation_test_total']+list(i))
        losses = []
        accs = []
        for csv in glob.glob(models_path + '/' + p + '_*.csv'):
            print(csv)
            dt = [ [ float(e) for e in row] for row in read(csv)][0]
            loss = dt[0]
            acc = dt[1]
            losses.append(loss)
            accs.append(acc)
            # print(dt)
        exp_count_info = i[:-1]
        exp_count = int(i[-1])

        if exp_count_info in collection:
            collection[exp_count_info][exp_count] = {'loss':losses, 'acc':accs}
        else:
            collection[exp_count_info] = {exp_count: {'loss':losses, 'acc':accs}}
    print(collection)

    # summarize measured data
    summaries = {}
    for c_info, c_dt in collection.items():
        epoch_losses = None
        epoch_accs = None
        for _, dt in c_dt.items():
            loss_dt = dt['loss']
            acc_dt = dt['acc']
            if epoch_losses is None:
                epoch_losses = [[] for i in range(len(loss_dt))]
            if epoch_accs is None:
                epoch_accs = [[] for i in range(len(acc_dt))]
            for i in range(len(loss_dt)):
                epoch_losses[i].append(loss_dt[i])
                epoch_accs[i].append(acc_dt[i])
        summaries[c_info] = [ list(summarize(l)) + list(summarize(a)) for l, a in zip(epoch_losses, epoch_accs)]
        
    # save summaries
    for stat_info, stat_data in summaries.items():
        stat_path = models_path + '/statistics_test_' + '_'.join(stat_info) + '.csv'
        remove(stat_path)
        write(stat_path, ["loss_avg, loss_var, loss_stddev, loss_med, acc_avg, acc_var, acc_stddev, acc_med"])
        write(stat_path, stat_data)

    # create plot data
    statistics_plot = {}
    for stat_info, stat_data in summaries.items():
        net, nlayers, nnodes, nalives, dropout, total_epochs, total_exps, dataset_name = stat_info
        plot_title = (net, nlayers, nnodes, nalives, total_epochs, total_exps, dataset_name)

        loss_avg = []
        loss_var = []
        loss_std = []
        loss_med = []
        acc_avg = []
        acc_var = []
        acc_std = []
        acc_med = []
        for i in range(len(stat_data)):
            sdt = stat_data[i]
            loss_avg.append(sdt[0])
            loss_var.append(sdt[1])
            loss_std.append(sdt[2])
            loss_med.append(sdt[3])
            acc_avg.append(sdt[4])
            acc_var.append(sdt[5])
            acc_std.append(sdt[6])
            acc_med.append(sdt[7])
        if plot_title in statistics_plot:
            statistics_plot[plot_title]['loss_avg'][dropout] = loss_avg
            statistics_plot[plot_title]['loss_var'][dropout] = loss_var
            statistics_plot[plot_title]['loss_std'][dropout] = loss_std
            statistics_plot[plot_title]['loss_med'][dropout] = loss_med
            statistics_plot[plot_title]['acc_avg'][dropout] = acc_avg
            statistics_plot[plot_title]['acc_var'][dropout] = acc_var
            statistics_plot[plot_title]['acc_std'][dropout] = acc_std
            statistics_plot[plot_title]['acc_med'][dropout] = acc_med
        else:
            statistics_plot[plot_title] = {
                'loss_avg': {dropout: loss_avg},
                'loss_var': {dropout: loss_var},
                'loss_std': {dropout: loss_std},
                'loss_med': {dropout: loss_med},
                'acc_avg': {dropout: acc_avg},
                'acc_var': {dropout: acc_var},
                'acc_std': {dropout: acc_std},
                'acc_med': {dropout: acc_med}}
    print(statistics_plot)
    
    # plot
    for stat_info, stat_plot_dataset in statistics_plot.items():
        for stat_item, plots in stat_plot_dataset.items():
            fig = plt.figure()
            ax = fig.add_subplot()
            colors = {'design': 'red', 'non': 'green', 'random': 'blue'}
            markers = {'design': 'o', 'non': '^', 'random': 'x'}
            labels = {'design': 'dropout design', 'non': 'no regularization', 'random': 'dropout'}
            maxdt = None
            for dropout_mode, dt in plots.items():
                x = [i+1 for i in range(len(dt))]
                ax.scatter(x, dt, c=colors[dropout_mode], marker=markers[dropout_mode],  label=labels[dropout_mode])
                maxdt = max(maxdt, max(dt)) if maxdt is not None else max(dt)
            maxdt = 10**(np.floor(np.log10(maxdt)) + 1) # adjust y's upper bound
            ax.set_xlim(0, len(x))
            ax.set_ylim(0, maxdt)
            ax.tick_params(labelsize=16) # font size
            ax.legend()
            fig.savefig(models_path + '/test_' + '_'.join(list(stat_info) + [stat_item, '.png']))

def accumulate_training():
    src_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_path)
    models_path = os.path.abspath(src_path + '/models')
    
    info_path = models_path + '/info.csv'
    info = [row for row in read(info_path)]
    print(info)

    # collect measured data
    collection = {}
    exp_info = set(tuple(e[:-1]) for e in info)
    for i in exp_info:
        p = '_'.join(['evaluation_training_total']+list(i))
        losses = []
        accs = []
        for csv in glob.glob(models_path + '/' + p + '_*.csv'):
            print(csv)
            dt = [ [ float(e) for e in row] for row in read(csv)][0]
            loss = dt[0]
            acc = dt[1]
            losses.append(loss)
            accs.append(acc)
            # print(dt)
        exp_count_info = i[:-1]
        exp_count = int(i[-1])

        if exp_count_info in collection:
            collection[exp_count_info][exp_count] = {'loss':losses, 'acc':accs}
        else:
            collection[exp_count_info] = {exp_count: {'loss':losses, 'acc':accs}}
    print(collection)

    # summarize measured data
    summaries = {}
    for c_info, c_dt in collection.items():
        epoch_losses = None
        epoch_accs = None
        for _, dt in c_dt.items():
            loss_dt = dt['loss']
            acc_dt = dt['acc']
            if epoch_losses is None:
                epoch_losses = [[] for i in range(len(loss_dt))]
            if epoch_accs is None:
                epoch_accs = [[] for i in range(len(acc_dt))]
            for i in range(len(loss_dt)):
                epoch_losses[i].append(loss_dt[i])
                epoch_accs[i].append(acc_dt[i])
        summaries[c_info] = [ list(summarize(l)) + list(summarize(a)) for l, a in zip(epoch_losses, epoch_accs)]
        
    # save summaries
    for stat_info, stat_data in summaries.items():
        stat_path = models_path + '/statistics_training_' + '_'.join(stat_info) + '.csv'
        remove(stat_path)
        write(stat_path, ["loss_avg, loss_var, loss_stddev, loss_med, acc_avg, acc_var, acc_stddev, acc_med"])
        write(stat_path, stat_data)

    # create plot data
    statistics_plot = {}
    for stat_info, stat_data in summaries.items():
        net, nlayers, nnodes, nalives, dropout, total_epochs, total_exps, dataset_name = stat_info
        plot_title = (net, nlayers, nnodes, nalives, total_epochs, total_exps, dataset_name)

        loss_avg = []
        loss_var = []
        loss_std = []
        loss_med = []
        acc_avg = []
        acc_var = []
        acc_std = []
        acc_med = []
        for i in range(len(stat_data)):
            sdt = stat_data[i]
            loss_avg.append(sdt[0])
            loss_var.append(sdt[1])
            loss_std.append(sdt[2])
            loss_med.append(sdt[3])
            acc_avg.append(sdt[4])
            acc_var.append(sdt[5])
            acc_std.append(sdt[6])
            acc_med.append(sdt[7])
        if plot_title in statistics_plot:
            statistics_plot[plot_title]['loss_avg'][dropout] = loss_avg
            statistics_plot[plot_title]['loss_var'][dropout] = loss_var
            statistics_plot[plot_title]['loss_std'][dropout] = loss_std
            statistics_plot[plot_title]['loss_med'][dropout] = loss_med
            statistics_plot[plot_title]['acc_avg'][dropout] = acc_avg
            statistics_plot[plot_title]['acc_var'][dropout] = acc_var
            statistics_plot[plot_title]['acc_std'][dropout] = acc_std
            statistics_plot[plot_title]['acc_med'][dropout] = acc_med
        else:
            statistics_plot[plot_title] = {
                'loss_avg': {dropout: loss_avg},
                'loss_var': {dropout: loss_var},
                'loss_std': {dropout: loss_std},
                'loss_med': {dropout: loss_med},
                'acc_avg': {dropout: acc_avg},
                'acc_var': {dropout: acc_var},
                'acc_std': {dropout: acc_std},
                'acc_med': {dropout: acc_med}}
    print(statistics_plot)
    
    # plot
    for stat_info, stat_plot_dataset in statistics_plot.items():
        for stat_item, plots in stat_plot_dataset.items():
            fig = plt.figure()
            ax = fig.add_subplot()
            colors = {'design': 'red', 'non': 'green', 'random': 'blue'}
            markers = {'design': 'o', 'non': '^', 'random': 'x'}
            labels = {'design': 'dropout design', 'non': 'no regularization', 'random': 'dropout'}
            maxdt = None
            for dropout_mode, dt in plots.items():
                x = [i+1 for i in range(len(dt))]
                ax.scatter(x, dt, c=colors[dropout_mode], marker=markers[dropout_mode],  label=labels[dropout_mode])
                maxdt = max(maxdt, max(dt)) if maxdt is not None else max(dt)
            maxdt = 10**(np.floor(np.log10(maxdt)) + 1) # adjust y's upper bound
            ax.set_xlim(0, len(x))
            ax.set_ylim(0, maxdt)
            ax.tick_params(labelsize=16) # font size
            ax.legend()
            fig.savefig(models_path + '/training_' + '_'.join(list(stat_info) + [stat_item, '.png']))

if __name__ == '__main__':
    accumulate_test()
    accumulate_training()