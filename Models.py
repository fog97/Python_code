
#Creating Models using class.py as data preprocessing

d=data_preparation()
data=d.importer('./kickstarter.csv','r',',')
#eimino le classi del target collegate a stati temporanei
#restano 2 classi
stop=[]
for el in data.state:
  if el!='failed' and el!='successful' and el not in stop:
    stop.append(el)
for parola in stop:    
  data=data[data.state!=parola]
data1=data.iloc[:25000,]
#Ora che ho un campione maneggevole posso iniziare a pulire i dati
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OneHotEncoder
#separo tra colonne categoriali e numeriche
colname=list(data1.columns)
Y=data1.state
num=data1.select_dtypes(include=('float','int'))
char=data1.drop(list(num),axis=1).drop('state',axis=1)
cat_clean=char.drop(['ID','name','deadline','launched','category'],axis=1)
lab=["Var","Var2","Chi_quadro","P_Value"]
numm=d.num_imputation(num)
cat=d.cat_imputation(cat_clean)
catt=d.cat_encoding(cat)
#per semplicità elimino le categoriali direttamente dal dataframe, altrimenti non potrei trasformarle
#elimino le nomeriche correlate
num_clean=np.delete(numm,[1,3,4,7,8],axis=1)
f=0
s=0
for st in Y:
  if st=='failed':
    f+=1
  else:
    s+=1
prior_f=round(f/len(data1[1:]),3)
prior_s=round(s/len(data1[1:]),3)
#usa un metodo abastanza brutale per rendere numerico il target.
#1=Failed, 0=Successful
Yy=np.empty(1)
for el in Y:
  if el=='failed':
    Yy=np.append(Yy,1)
  else:
    Yy=np.append(Yy,0)
Yy=np.delete(Yy,0)
#Passo a creare il DataFrame 
from sklearn.model_selection import train_test_split
df=np.append(num_clean,catt,axis=1)
X=df
X_train, X_test, Y_train, Y_test = train_test_split(X, Yy, test_size=0.6, random_state=0)
#Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(priors=(prior_f,prior_s),  var_smoothing=1e-30)
NB_pred = gnb.fit(X_train, Y_train).predict(X_test)
print("Naive Bayes:Number of mislabeled points out of a total %d points : %d"
  % (X_test.shape[0], (Y_test != NB_pred).sum()))
gnb.get_params()
#Decision Tree
from sklearn import tree
DecTree = tree.DecisionTreeClassifier()
Tree_pred = DecTree.fit(X_train, Y_train).predict(X_test)
print("Tree:Number of mislabeled points out of a total %d points : %d"
  % (X_test.shape[0], (Y_test != Tree_pred).sum()))
#Bagging
from sklearn.ensemble import BaggingClassifier
Bagging = BaggingClassifier(gnb,max_samples=0.5, max_features=0.5)
Bagging_pred = Bagging.fit(X_train, Y_train).predict(X_test)
print("Bagging:Number of mislabeled points out of a total %d points : %d"
  % (X_test.shape[0], (Y_test != Bagging_pred).sum()))
#Bagging Di Alberi
from sklearn.ensemble import BaggingClassifier
Bagging_Tree = BaggingClassifier(DecTree,max_samples=0.5, max_features=0.6)
Bagging_Tree_pred = Bagging_Tree.fit(X_train, Y_train).predict(X_test)
print("Tree Baging :Number of mislabeled points out of a total %d points : %d"
  % (X_test.shape[0], (Y_test != Bagging_Tree_pred).sum()))
#Gradient Descent
from sklearn.linear_model import SGDClassifier
GD = SGDClassifier(loss="log", penalty="l2", max_iter=500)
GD_pred=GD.fit(X_train, Y_train).predict(X_test)
print("GD:Number of mislabeled points out of a total %d points : %d"
  % (X_test.shape[0], (Y_test != GD_pred).sum()))
#SVM
from sklearn import svm
SVM = svm.SVC()
SVM_pred = SVM.fit(X_train, Y_train).predict(X_test)
print("SVM:Number of mislabeled points out of a total %d points : %d"
  % (X_test.shape[0], (Y_test != SVM_pred).sum()))
print('Traccio curve Roc e calcolo AUC.')
#Curva Roc per tutti i modelli
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
ax = plt.gca()
NB_disp = plot_roc_curve(gnb, X_test, Y_test,ax=ax,alpha=10)
Tree_disp = plot_roc_curve(DecTree,X_test, Y_test, ax=ax, alpha=10)
Bagging_Tree_disp = plot_roc_curve(Bagging_Tree, X_test, Y_test,ax=ax,alpha=10)
GD_disp = plot_roc_curve(GD, X_test, Y_test,ax=ax,alpha=10)
#SVM_disp = plot_roc_curve(SVM, X_test, Y_test,ax=ax,alpha=10)
plt.show()
print('Precision recall Curve')
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
disp = plot_precision_recall_curve(Bagging_Tree, X_test, Y_test)
# Plot calibration plots
from sklearn.calibration import calibration_curve
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(gnb, 'Naive Bayes'),
                  (Bagging_Tree,'Bagging Di Alberi'),
                  (SVM, 'Support Vector Machine'),
                  (DecTree,'Albero')]:
    clf.fit(X_train, Y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(Y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()
print('End Code')
