import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch


def plot_episodes(all_total_mean,all_total_std, all_total_max, y_labels, title, models_dir, plot_file):
    '''
    Plot episodes of one experiment. showing max and mean reards( with std boundaries)
    :param all_total_mean:
    :param all_total_std:
    :param all_total_max:
    :param y_labels:
    :param title:
    :param models_dir:
    :param plot_file:
    :return:
    '''
    fig = plt.figure(figsize=(12,8))
    plot_file=os.path.join(models_dir, plot_file+'.png')
    plt.plot(range(len(all_total_mean)), all_total_mean, color='blue', label='mean')
    plt.plot(range(len(all_total_max)), all_total_max, color='green', label='max')
    plt.fill_between(range(len(all_total_mean)), all_total_mean - all_total_std / 2, all_total_mean + all_total_std / 2, color='red', alpha=0.2)
    plt.xlabel('Episodes')
    plt.ylabel(y_labels)
    plt.legend()
    plt.title(title)
    plt.savefig(plot_file, dpi=fig.dpi)
    plt.show()



def plot_all_experiments(list_all_total_mean, labels, y_labels, title, models_dir, plot_file):
    '''
        Plot a few experiments  Create line plot of a few experiment ont he same plot
        :param all_total_mean:
        :param all_total_std:
        :param all_total_max:
        :param y_labels:
        :param title:
        :param models_dir:
        :param plot_file:
        :return:
    '''

    plot_file = os.path.join(models_dir, plot_file + '.png')

    fig = plt.figure(figsize=(14,10))
    for ind, _ in enumerate(list_all_total_mean):
        plt.plot(range(len(list_all_total_mean[ind])), list_all_total_mean[ind], label=labels[ind])

    plt.xlabel('Episodes')
    plt.ylabel(y_labels)
    plt.legend()
    plt.title(title)
    plt.savefig(plot_file, dpi=fig.dpi)
    plt.show()




def save_to_pickle(entity, file):
    # Save to pickle file
    with open(file, 'wb') as file:
        pickle.dump(entity, file)
    file.close()

def load_from_pickle(file):
    # Load object from  pickle file
    with open(file, 'rb') as file:
        entity = pickle.loads(file.read())
    file.close()
    return entity



def save_model(model, optimizer, models_dir, file_name, model2=None):
    #Save to checkpoint file

    filename=os.path.join(models_dir, file_name + '.pth')

    if model2 is None:
        model2_state_dict=None
    else:
        model2_state_dict=model2.state_dict()

    torch.save({
        'model_state_dict': model.state_dict(),
        'model2_state_dict': model2_state_dict,
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)



def load_model(models_dir,file_name):
    # return state dict from a checkpoint file

    filename = os.path.join(models_dir,file_name)
    checkpoint = torch.load(filename)
    model_state_dict=checkpoint['model_state_dict']
    optimizer_state_dict=checkpoint['optimizer_state_dict']
    model2_state_dict = checkpoint['model2_state_dict']

    return model_state_dict, optimizer_state_dict, model2_state_dict