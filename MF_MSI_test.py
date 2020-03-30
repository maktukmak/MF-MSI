from utils import eval_res
from DatasetPrep import movie100kprep

from MF_MSI import MF_MSI

# load Movielens 100K dataset with cold setting
Dataset, DatasetSpec = movie100kprep(option = 'cold')

# Initialize MF-MSI model with dataset specs
model = MF_MSI(DatasetSpec)

# Perform inference with dataset entries
model.fit(Dataset, iterno = 5)
Rpred = model.predict()

# Evaulate average recall at L = 2
Test_MSE, Train_MSE, Test_recall, Train_recall,_ = eval_res(Rpred, 
                                                            Dataset['Rtest'], 
                                                            Dataset['Rtrain'], 
                                                            Dataset['Otest'], 
                                                            Dataset['Otrain'], 
                                                            at_K = 2)

# Print test performance
print("Recall = {:.3f}".format(Test_recall))

