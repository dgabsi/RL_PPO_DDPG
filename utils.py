import pickle
import os
import torch

def save_to_pickle(entity, file):
    with open(file, 'wb') as file:
        pickle.dump(entity, file)
    file.close()

def load_from_pickle(file):
    with open(file, 'rb') as file:
        entity = pickle.loads(file.read())
    file.close()
    return entity



def save_model(model, optimizer, models_dir, file_name, model2=None):

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

    filename = os.path.join(models_dir,file_name)
    checkpoint = torch.load(filename)
    model_state_dict=checkpoint['model_state_dict']
    optimizer_state_dict=checkpoint['optimizer_state_dict']
    model2_state_dict = checkpoint['model2_state_dict']

    return model_state_dict, optimizer_state_dict, model2_state_dict