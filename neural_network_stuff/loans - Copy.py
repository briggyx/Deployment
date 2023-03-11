import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import collections
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle as pk
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

class LoanData:
    
    def __init__(self,csv_path,do_cleaning=True):
        """Initialize a new LoanPipeline, loading data from the specified CSV file"""
        self.original = pd.read_csv(csv_path)
        if do_cleaning:
            self.clean()
        else:
            self.cleaned = None

    def __repr__(self):
        """Equivalent to the most complete internal dataframe's __repr__"""

        if self.cleaned is not None:
            print('CLEANED DATA:')
            display(self.cleaned)
        else:
            print('ORIGINAL DATA:')
            display(self.original)
        return('')
        
    def to_csv(self,csv_path,which='original'):
        """Write the data to a CSV; which=['original','cleaned']"""
        if which == 'original':
            self.original.to_csv(csv_path,index=False)
        elif which == 'cleaned':
            if self.cleaned is None:
                print('Error: no cleaned data to write to CSV.')
                return
            self.cleaned.to_csv(csv_path,index=False)
        else:
            print(f'Invalid value for which parameter: {which}')
    
    def clean(self):
        """Tidy column names, fill in missing data, convert data types"""
        
        # Remove trailing whitespace from strings
        self.original = self.original.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Remove underscores from column names for consistency
        self.original = self.original.rename(columns=lambda x: x.replace('_', ''))
        
        # Make index = loan ID, and remove loan ID column
        self.cleaned = self.original.copy()
        self.cleaned.index = self.cleaned.LoanID
        self.cleaned = self.cleaned[self.cleaned.columns[1:]]
        
        # Fill missing data with median, mode, or zero, as appropriate
        medians = self.cleaned[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'LoanAmountTerm']].median() # Calculate medians
        modes = self.cleaned[['Dependents','SelfEmployed']].mode() # Calculate modes
        replace = { x:medians[x] for x in medians.index } # Dict to hold column name to value to sub in place of nan mapping
        replace['Dependents'] = modes.Dependents.sample(1)[0] # Sample in case mode returned multiple values
        replace['SelfEmployed'] = modes.SelfEmployed.sample(1)[0] # Sample in case mode returned multiple values
        replace['CreditHistory'] = 0 # Assume missing values in CreditHistory mean no history is available, = 0
        replace['Gender'] = 'Unknown'
        replace['Married'] = 'Unknown'
        print('FILLING IN MISSING DATA AS FOLLOWS:')
        for col,val in replace.items():
            print(f'Missing data in {col} replaced with {val}')
        self.cleaned.fillna(replace,inplace=True) # Fill nans with medians
        
        # Convert numeric data to float32; some would be fine as integers due to lack of decimals but new data may break that assumption
        self.cleaned.ApplicantIncome = self.cleaned.ApplicantIncome.astype(np.float32)
        self.cleaned.CoapplicantIncome = self.cleaned.CoapplicantIncome.astype(np.float32)
        self.cleaned.LoanAmount = self.cleaned.LoanAmount.astype(np.float32)
        self.cleaned.LoanAmountTerm = self.cleaned.LoanAmountTerm.astype(np.float32)
        
        # Convert ordered categorical data to unsigned 8-bit integers
        self.cleaned.Dependents = self.cleaned.Dependents.map({'0':0,'1':1,'2':2,'3+':3}).astype(np.uint8)
        self.cleaned.SelfEmployed = self.cleaned.SelfEmployed.map({'Yes': 1, 'No': 0}).astype(np.uint8)
        self.cleaned.LoanStatus = self.cleaned.LoanStatus.map({'Y': 1, 'N': 0}).astype(np.uint8)
        self.cleaned.Education = self.cleaned.Education.map({'Graduate': 1, 'Not Graduate': 0}).astype(np.uint8)
        self.cleaned.rename(columns={'Education':'Graduate'},inplace=True)
        
        # One-hot-encode gender and marriage status (necessary because missing data was treated as 'other')
        gender = pd.get_dummies(self.cleaned.Gender)
        marriage = pd.get_dummies(self.cleaned.Married)
        gender.columns = [ f'Gender_{g}' for g in gender.columns ]
        marriage.columns = [ f'Married_{m}' for m in marriage.columns ]
        prop = pd.get_dummies(self.cleaned.PropertyArea)
        one_hot = pd.merge(pd.merge(gender,marriage,left_index=True,right_index=True),prop,left_index=True,right_index=True)
        
        # Merge everything
        self.cleaned = pd.merge(one_hot,self.cleaned[self.cleaned.columns[2:].drop('PropertyArea')],left_index=True,right_index=True)
    
        # Remove outliers
        print('\nREMOVING OUTLIERS:')
        cols_to_check = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        
        for col in cols_to_check:
            print(f"Initial range of {col}: {self.cleaned[col].min()} - {self.cleaned[col].max()}")
        
        q1, q4 = self.cleaned[cols_to_check].quantile(0.25), self.cleaned[cols_to_check].quantile(0.75)
        iqr = q4 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q4 + 1.5 * iqr
        
        mask = (self.cleaned[cols_to_check] >= lower_bound) & (self.cleaned[cols_to_check] <= upper_bound)
        self.cleaned = self.cleaned[mask.all(axis=1)].copy()
        
        for col in cols_to_check:
            print(f"Filtered range of {col}: {self.cleaned[col].min()} - {self.cleaned[col].max()}")

class PredictionPipeline:
    """Implements data cleaning and loan prediction functionality; initialize with a path to a CSV file"""
    
    def __init__(self):
        self.scaler = None
        self.estimator = None
        self.neural = None
    
    def _check_data_(self,data):
        if type(data) == LoanData:
            if data.cleaned is not None:
                data = data.cleaned
            else:
                print('Error: LoanData object passed to fit_scaler, but no cleaned data found; run LoanData.clean() first')
                return
        return data
    
    def _get_Xy_(self,data,label_target):
        data = pd.DataFrame(self.scaler.transform(data),columns=data.columns)
        X = data.drop(columns=[label_target])
        y = data[label_target] if label_target in data.columns else None
        return X,y
    
    def fit_scaler(self,data,scaler = MinMaxScaler()):
        """Fit a scaler, default = min-max"""
        
        data = self._check_data_(data)
        scaler.fit_transform(data)
        self.scaler = scaler

    def test_and_fit_estimator(self,data,
                               estimator=LogisticRegression(),
                               label_target='LoanStatus',
                               ncomp=np.arange(2,20,1),
                               penalties=["l1", "l2"],
                               C=np.arange(1,25,2.5)/5,
                               solvers=['liblinear']):
        """Takes an estimator and uses grid search to find the best parameters, and reports these along with model accuracy, prompting the user to accept or reject the model"""
        
        data = self._check_data_(data)
        
        if self.scaler is None:
            print('Error: No scaler implemented yet; run PredictionPipeline.fit_scaler() first.')
            return
        
        X,y = self._get_Xy_(data,label_target)
        
        param_grid = { "pca__n_components": ncomp, "lr__penalty": penalties,
        "lr__C": C, "lr__solver": solvers }

        # Combine PCA and logistic regression into a pipeline
        pipe = Pipeline([("pca", PCA()),("lr", estimator)])

        # Create a grid search object
        grid = GridSearchCV(pipe, param_grid, cv=3)
        
        # Fit the grid search object to the data
        grid.fit(X, y)

        # Print the best hyperparameters and accuracy score
        print("Best hyperparameters:", grid.best_params_)
        print("Accuracy score:", grid.best_score_)
        
        # Prompt the user to accept or reject the model
        accept = input('Do you wish to implement this model? (y/n):')
        if accept.upper() != 'Y':
            print('Model not implemented; please run PredictionPipeline.test_and_fit() again')
            return
        
        # If the model is good, fit and store it
        self.estimator = grid.best_estimator_
        self.estimator.fit(X,y)
        print('Model has been fit and implemented.')
    
    def train_neural(self,data,
                     label_target='LoanStatus',
                     layers=[30,20,10],
                     dropout=0.25,
                     optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'],
                     epochs=500,
                     val_frac=0.35):
        
        data = self._check_data_(data)
        
        if self.scaler is None:
            print('Error: No scaler implemented yet; run PredictionPipeline.fit_scaler() first.')
            return
        
        X,y = self._get_Xy_(data,label_target)
        
        tensor = X.to_numpy(dtype='float32')
        tensor_labels = y.to_numpy(dtype='float32')
        
        model = keras.Sequential()
        model.add(keras.layers.Dense(layers[0],input_dim=tensor.shape[1],activation='relu'))
        if dropout:
            model.add(keras.layers.Dropout(dropout))
        for n in layers[1:]:
            model.add(keras.layers.Dense(n,activation='relu'))
            if dropout:
                model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        
        model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        
        split = int(round(tensor.shape[0]*val_frac))
        
        # Checkpoint for saving the best model, despite over-fitting
        checkpoint = ModelCheckpoint('best_model.h5',monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)
        
        # Train the model
        history = model.fit(tensor[split:,:], tensor_labels[split:],
                            validation_data=(tensor[:split,:], tensor_labels[:split]),
                            epochs=epochs,batch_size=split,verbose=0,callbacks=[checkpoint])
        
        # Load the weights from the best model
        model.load_weights('best_model.h5')
        
        # Plot the training process
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        
        # Print the validation accuracy of the best model
        best_val_acc = max(history.history['val_accuracy'])
        print('Validation accuracy of the best model:', best_val_acc)
        
        # Prompt the user to accept/reject
        accept = input('Accept this model? (y/n):')
        
        # Implement or do not implement as specified
        if accept.upper() != 'Y':
            print('Model not implemented; run PredictionPipeline.neural() again.')
            return
        self.neural = model
        print('Model has been fit and implemented.')
    
    def predict(self,data,neural=False,label_target='LoanStatus'):
        """Run predictions; if y is supplied, accuracy is checked"""
        
        data = self._check_data_(data)
        
        if self.scaler is None:
            print('Error: No scaler implemented yet; run PredictionPipeline.fit_scaler() first.')
            return
        
        X,y = self._get_Xy_(data,label_target)
        
        if neural:
            
            if self.neural is None:
                print('Error, prediction with neural network requested but no network is implemented; run PredictionPipeline.train_neural() first.')
                return
            
            X = X.to_numpy(dtype='float32')
            
            predictions = self.neural.predict(X)
            predictions = (predictions > 0.5).astype(int).T[0]
            
        else:
            
            if self.estimator is None:
                print('Error, no estimator is implemented; run PredictionPipeline.test_and_fit() first.')
                return
            
            predictions = self.estimator.predict(X)
            
        if y is not None:
                
            predictions_0 = predictions[y==0]
            predictions_1 = predictions[y==1]

            y_0 = y[y==0]
            y_1 = y[y==1]

            accuracy_total = sum(predictions == y)/len(predictions)
            accuracy_0 = sum(predictions_0 == y_0)/len(predictions_0)
            accuracy_1 = sum(predictions_1 == y_1)/len(predictions_1)
            
            print(f'Accuracy (overall): {round(accuracy_total*100,2)}%')
            print(f'Accuracy (y=0): {round(accuracy_0*100,2)}%')
            print(f'Accuracy (y=1): {round(accuracy_1*100,2)}%')
            
        return predictions
    
    def save(self,prefix):
        pk.dump(self.scaler,open(f'{prefix}.scaler.pk','wb'))
        pk.dump(self.estimator,open(f'{prefix}.estimator.pk','wb'))
        self.neural.save(f'{prefix}.mod.h5')
        
    def load(self,prefix):
        self.scaler = pk.load(open(f'{prefix}.scaler.pk','rb'))
        self.estimator = pk.load(open(f'{prefix}.estimator.pk','rb'))
        self.neural = keras.models.load_model(f'{prefix}.mod.h5')