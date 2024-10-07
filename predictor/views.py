# from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np
import pandas as pd
# load data to model

# Create your views here.
def index(request):
    title = 'Home'
    context = {'title': title}
    return render(request, 'predictor/home.html', context)

def about(request):
    title = 'About'
    context = {'title': title}
    return render(request, 'predictor/about.html', context)

def kalman(request):
    title = 'Kalman'
    # Importing data
    df = pd.read_csv("C:/Users/PRESTIGE PC/Desktop/Projects/Artificial Intelligence/kalmantest2.csv")
    X= df.drop(["Actual duration","Estimated  duration","Actual effort", "Estimated effort"], axis = 1)
    y =np.array(df.drop(['Organization id', 'Organization type', 'Size of organization',
       'Customer organization type', 'Estimated  duration', 'Actual duration',
       'project gain (loss)', 'Development type', 'Application domain',
       'Object points', 'Other sizing method', 'Estimated size',
       'Estimated effort'], axis = 1))
   # Apply MinMaxScaler to the input features and target variable
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    original_shape_y = y.shape
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
    shape_of_y_after_scaling = y_scaled.shape
    y_scaled = y_scaled.flatten()
    shape_of_y_after_flatten = y_scaled.shape
    X_scaled = X_scaler.fit_transform(X)
    
    # train-test splitting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.3,random_state=0)
   
    # model init
    lr = LinearRegression()
    
    # model training
    lr.fit(X_train, y_train)
    # performing 10-fold cross validation
    # kf = KFold(n_splits = 10, shuffle = True)
    # splits = list(kf.split(X_scaled))
    # scores = []
    # for train_index, test_index in splits:
    #     X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    #     y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    #     # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.03)
    #     # Training the model
    #     lr.fit(X_train, y_train)
    #     scores.append(lr.score(X_test, y_test))
    #     print(np.mean(scores))
    # defining the kalman filter
    def kalman_filter(initial_state, initial_estimate_covariance, state_transition_matrix, observation_matrix, observation_covariance, observations):
        num_observations = len(observations)

        state_estimate = initial_state
        estimate_covariance = initial_estimate_covariance

        predicted_states = []
        filtered_states = []

        for observation in observations:
            # Prediction Step
            predicted_state = state_transition_matrix @ state_estimate
            predicted_covariance = state_transition_matrix @ estimate_covariance @ state_transition_matrix.T

        # Update Step
        kalman_gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(
            observation_matrix @ predicted_covariance @ observation_matrix.T + observation_covariance
        )

        state_estimate = predicted_state + kalman_gain @ (observation - observation_matrix @ predicted_state)
        estimate_covariance = (np.identity(initial_estimate_covariance.shape[0]) - kalman_gain @ observation_matrix) @ predicted_covariance

        predicted_states.append(predicted_state)
        filtered_states.append(state_estimate)

        return filtered_states

# Define initial parameters
    initial_state = np.array([[0]])  # Initial completion percentage
    initial_estimate_covariance = np.array([[0.1]])  # Initial uncertainty
    state_transition_matrix = np.array([[1]])  # Linear increase assumption
    observation_matrix = np.array([[1]])  # Direct observation
    observation_covariance = np.array([[0.01]])  # Observation noise

# Simulated observations (replace with actual data)
    # observations = np.array([3168,1584,5280,38016]) #np.array([0.1, 0.3, 0.5, 0.6])

# Apply Kalman filter
    # filtered_states = kalman_filter(initial_state, initial_estimate_covariance, state_transition_matrix, observation_matrix, observation_covariance, observations)
# prediction
    # y_pred = lr.predict(X_test)
    if request.method == "POST":
        Organisation_id = int(request.POST.get('Organisation id'))
        Organisation_type = int(request.POST.get('Organisation type'))
        Size_of_Organisation = int(request.POST.get('Size of Organisation'))
        Customer_Organisation_type = int(request.POST.get('Customer Organisation type'))
        Project_gain = int(request.POST.get('Project gain (loss)'))
        Development_type = int(request.POST.get('Development type'))
        Application_domain = int(request.POST.get('Application domain'))
        Object_points = int(request.POST.get('Object points'))
        Other_sizing_methods = int(request.POST.get('Other sizing methods'))
        Estimated_size = int(request.POST.get('Estimated size'))
        y_pred = lr.predict(X_scaler.transform(np.array([Organisation_id, Organisation_type, Size_of_Organisation, Customer_Organisation_type,Project_gain, Development_type, Application_domain,Object_points, Other_sizing_methods, Estimated_size]).reshape((1,-1))))
        
        # Apply kalman filter
        y_estimated = kalman_filter(initial_state, initial_estimate_covariance, state_transition_matrix, observation_matrix, observation_covariance, y_pred)
        
        # If you want to revert the predictions to the original scale:
        y_estimated = np.array(y_estimated).reshape((-1, 1))
        original_kalman_estimations = y_scaler.inverse_transform(y_estimated).flatten()
        pred = round(original_kalman_estimations[0])
        if pred < 0:
            pred *= -1
        context = {'title':title, 'pred': pred}
        return render(request, 'predictor/kalman.html', context)

    context = {'title': title}
    return render(request, 'predictor/kalman.html', context)

def predict(request):
    title = 'Predict'
    # Importing data
    df = pd.read_csv("C:/Users/PRESTIGE PC/Desktop/Projects/Artificial Intelligence/kalmantest2.csv")
    X= df.drop(["Actual duration","Estimated  duration","Actual effort", "Estimated effort"], axis = 1)
    # creating the project duration as target variable
    y_target =np.array(df.drop(['Organization id', 'Organization type', 'Size of organization',
       'Customer organization type', 'Actual duration',
       'project gain (loss)', 'Development type', 'Application domain',
       'Object points', 'Other sizing method', 'Estimated size','Actual effort',
       'Estimated effort'], axis = 1))
    # scale the y_target variable which is the Estimated duration
    y_scaler2 = MinMaxScaler()
    y_target_scaled = y_scaler2.fit_transform(y_target)

    # split the y_target_scaled into train and test set
    X_train2,X_test2, y_train2, y_test2 = train_test_split(X,y_target_scaled, test_size=0.3, random_state=0)
    
    # initialize the second linear regression model to be used to predict the
    # estimated duration
    lr2 = LinearRegression()
    
    # train the model using the earlier X_train data and y_train2
    lr2.fit(X_train2, y_train2)

    # predict the model using the earlier X_test data and y_test2 as target
    y_pred2 = lr2.predict(X_test2)

    # Evaluate the second method using the same metrics as above using y_pred2 as target this time
    mse = mean_squared_error(y_test2, y_pred2)
    mae = mean_absolute_error(y_test2, y_pred2)
    r2 = r2_score(y_test2, y_pred2)
    rmse = np.sqrt(mse)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("R-Squared:", r2)
    
    # Apply Kalman filter to estimate the true duration
    # y_estimated2 = kalman_filter(initial_state, initial_estimate_covariance, state_transition_matrix, observation_matrix, observation_covariance, y_pred2)
    # If you want to revert the predictions to the original scale:
    #y_estimated2 = np.array(y_estimated2).reshape((-1, 1))
    #original_kalman_estimations_duration = y_scaler.inverse_transform(y_estimated2).flatten()
    if request.method == "POST":
        Organisation_id = int(request.POST.get('Organisation id'))
        Organisation_type = int(request.POST.get('Organisation type'))
        Size_of_Organisation = int(request.POST.get('Size of Organisation'))
        Customer_Organisation_type = int(request.POST.get('Customer Organisation type'))
        Project_gain = int(request.POST.get('Project gain (loss)'))
        Development_type = int(request.POST.get('Development type'))
        Application_domain = int(request.POST.get('Application domain'))
        Object_points = int(request.POST.get('Object points'))
        Other_sizing_methods = int(request.POST.get('Other sizing methods'))
        Estimated_size = int(request.POST.get('Estimated size'))
        y_pred = lr2.predict(np.array([Organisation_id, Organisation_type, Size_of_Organisation, Customer_Organisation_type,Project_gain, Development_type, Application_domain,Object_points, Other_sizing_methods, Estimated_size]).reshape((1,-1)))
        
        # Apply kalman filter
        # y_estimated = kalman_filter(initial_state, initial_estimate_covariance, state_transition_matrix, observation_matrix, observation_covariance, y_pred)
        
        # If you want to revert the predictions to the original scale:
        y_estimated = np.array(y_pred).reshape((-1, 1))
        original_kalman_estimations = y_scaler2.inverse_transform(y_estimated).flatten()
        pred = round(original_kalman_estimations[0],1)
        context = {'title':title, 'pred': pred}
        return render(request, 'predictor/predict.html', context)






    '''data = pd.read_csv("C:/Users/Hp/Desktop/AmesHousing.csv")
    # Cleaning data
    data = data.fillna(0)

    # Splitting data
    x = data.drop('SalePrice', axis=1)
    y = data['SalePrice']'''

    '''scores = []
    model = LinearRegression()
    # performing 10-fold cross validation
    kf = KFold(n_splits = 10, shuffle = True)
    splits = list(kf.split(x))
    for train_index, test_index in splits:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.03)
    # Training the model
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    #classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
    #classifier.fit(x_train, y_train)'''
    '''print(np.mean(scores))
    context = {'title': title}
    if request.method == 'POST':
        quality = int(request.POST.get('quality'))
        condition = int(request.POST.get('condition'))
        bathrooms = int(request.POST.get('bathrooms'))
        toilets = int(request.POST.get('toilets'))
        bedrooms = int(request.POST.get('bedrooms'))
        kitchens = int(request.POST.get('kitchens'))
        pred = model.predict(np.array([quality, condition, bathrooms, toilets, bedrooms, kitchens]).reshape(1,-1))
        pred = round(pred[0])'''
        # context = {'title': title}#,"pred": pred}
        # return render(request, 'predict.html', context)
    context = {'title': title}
    return render(request, 'predictor/predict.html', context)