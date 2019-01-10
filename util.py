import requests
import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np

def download_training_data(url):
    print('downloading training data...')
    if not os.path.isdir("data"):
        os.makedirs('data')
    r = requests.get(url)
    with open("file.zip", "wb") as code:
        code.write(r.content)
    with zipfile.ZipFile("file.zip", "r") as zip_ref:
        zip_ref.extractall("data")
        
def historyplot(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return plt.show()

def df_padd1(final_length, input_df):  # pad data from  previous rows
    """
    final_length: max length
    input_df: input data
    """
    new_df=[]
    for i in input_df:
        if final_length>=len(i):
            #add =final_length-input_df # padd
            new_df1=np.concatenate([i,i,i,i])
            new_df1=new_df1[0:final_length]
        if final_length<len(i): 
            new_df1=i[0:final_length]
        new_df.append(new_df1) 
    return  new_df

def df_padd2(final_length, input_df): #  just pad the last row.
    """
    final_length: max length
    input_df: input data
    """
    new_df=[]
    for i in input_df:
        if final_length>=len(i):
            lastrow=np.repeat(i[-1], final_length).reshape(4, final_length).T
            new_df1=np.concatenate([i,lastrow])
            new_df1=new_df1[0:final_length]
        if final_length<len(i): 
            new_df1=i[0:final_length]        
        new_df.append(new_df1)  
    return new_df

def train_dev_test(final_df,groups,targets): #split training data based on groups
    train_X=np.array([final_df[i] for i in range(len(groups)) if (groups[i]==2)])
    dev_X  =np.array([final_df[i] for i in range(len(groups)) if (groups[i]==1)])
    test_X =np.array([final_df[i] for i in range(len(groups)) if (groups[i]==3)])

    train_Y= [targets[i] for i in range(len(groups)) if (groups[i]==2)]
    dev_Y  = [targets[i] for i in range(len(groups)) if (groups[i]==1)]
    test_Y = [targets[i] for i in range(len(groups)) if (groups[i]==3)]
    
    train_Y =np.array(list(map(lambda x: 1 if train_Y[x]==1 else 0, range(len(train_Y)))))
    dev_Y   =np.array(list(map(lambda x: 1 if dev_Y[x]  ==1 else 0, range(len(dev_Y  )))))
    test_Y  =np.array(list(map(lambda x: 1 if test_Y[x] ==1 else 0, range(len(test_Y )))))
    
    return train_X, dev_X, test_X, train_Y, dev_Y, test_Y
