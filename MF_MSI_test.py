from utils import eval_res
from DatasetPrep import movie100kprep

from MF_MSI import MF_MSI

# load Movielens 100K dataset with cold setting
Xnorm, Ynorm, Du, Mu, Pnorm, Znorm, Di, Mi, Rtrain, Rtest, Otrain, Otest, I, J = movie100kprep(option = 'cold')

# Initialize MF-MSI model
model = MF_MSI(Du, Mu, Di, Mi, I, J)

# Perform inference by feeding Dataset
Rpred = model.fit(Xnorm, Ynorm, Znorm, Pnorm, Rtrain, Otrain,Rtest, Otest, iterno = 5)

# Evaulate average recall at L = 2
Test_MSE, Train_MSE, Test_recall, Train_recall,_ = eval_res(Rpred, Rtest, Rtrain, Otest, Otrain, 2)

# Print resulting test performance
print("Recall = {:.3f}".format(Test_recall))

