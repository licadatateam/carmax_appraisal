# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:57:26 2023

@author: carlo

"""
# system tools
import time
import re
from datetime import datetime, timedelta
from functools import cache

# utility modules
import pandas as pd
import numpy as np
import config_lica
import bq_functions

# sklearn/model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap

# UI
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid
import streamlit.components.v1 as components

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def import_data() -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Import carmax backend data from Redash and carmax competitor data from BQ
    Cleans imported data and combines both datasets
    '''
    print ('Start task: {}'.format(datetime.now().time().strftime('%H:%M:%S')))
    start_time = time.time()
    
    # initialize BQ access
    acct = bq_functions.get_acct()
    client, credentials = bq_functions.authenticate_bq(acct)
    
    # import competitor data from BQ
    print ('Start importing competitor data')
    try:
        comp_data = bq_functions.query_bq('.'.join([credentials.project_id, 
                                                    'carmax', 
                                                    'competitors_compiled']), 
                                          client)
        if comp_data[start_cols].isnull().sum().sum():
            print ('Cleaning competitor data')
            comp_data = config_lica.clean_df(comp_data)
            
            try:
                # attempt to write new cleaned data to BQ
                bq_functions.bq_write(comp_data, credentials, 
                                      'carmax', 'competitors_compiled', client)
            except:
                try: 
                    # attempt to upload data to GCS
                    config_lica.upload_to_gcloud(comp_data, 
                                             'competitors_compiled.csv',
                                             bucket_name = 'carmax_competitors')
                except:
                    # just continue operation
                    pass   
        else:
            # no need to clean data
            pass
    
    except:
        raise Exception('Unable to import competitor data.')
    
    # import backend data from redash
    print ('Start importing backend data')
    try:
        backend_data = bq_functions.query_bq('.'.join([credentials.project_id, 
                                                    'carmax', 
                                                    'backend_compiled']), 
                                          client)
        
        
        if backend_data[start_cols].isnull().sum().sum():
            backend_data = pd.read_csv('http://app.redash.licagroup.ph/api/queries/7/results.csv?api_key=sSt3ILBkdxIbOFC5DqmQxQhBq7SiiKVZBc8FBtei')
            backend_data.columns = ['_'.join(col.lower().split(' ')) for col in backend_data.columns]
            backend_data.loc[:, 'platform'] = 'carmax'
            backend_data = backend_data.rename(columns = {'selling_price' : 'price',
                                      'vehicle_location' : 'location',
                                      'po_date' : 'date_listed',
                                      'vehicle_type' : 'body_type'})
            print ('Cleaning backend data')
            backend_data = config_lica.clean_df(backend_data)
            try:
                # attempt to write new cleaned data to BQ
                bq_functions.bq_write(comp_data, credentials, 
                                      'carmax', 'backend_compiled', client)
            except:
                try:
                    # attempt to upload data to GCS
                    config_lica.upload_to_gcloud(comp_data, 
                                             'backend_compiled.csv',
                                             bucket_name = 'carmax_competitors')
                except:
                    pass
        else:
            pass
        
    except:
        raise Exception('Unable to import backend data.')
        
    # clean & combine data
    cols = ['date_listed', 'make', 'model', 'year', 'transmission', 'mileage', 
            'fuel_type', 'body_type', 'price', 'location', 'url', 'platform']
    
    # combine backend and competitor data
    df = pd.concat([backend_data[cols], comp_data[cols]], axis=0, ignore_index = True)
    # drop rows with incomplete data in required cols
    df.dropna(subset = cols, inplace = True)
    
    ## TODO : validate data
    ## TODO : Write cleaned data to BQ
    
    print ('End task: {}'.format(datetime.now().time().strftime('%H:%M:%S')))
    print('Task duration: {0:1.2f} s'.format(time.time() - start_time))
    return df, comp_data, backend_data

@st.cache_data
def feature_engineering(df):
    '''
    Executes feature engineering for dataset to prep for model training
    '''
    print ('Start task: {}'.format(datetime.now().time().strftime('%H:%M:%S')))
    start_time = time.time()
    
    ## TODO : validate input data for available feature cols
    try:
        col_check = [col  for col in start_cols if col not in df.columns]
        if len(col_check):
            print ('''Data validation: FAILED\nMissing columns: {}'''.format(', '.join(col_check)))
        else:
            print ('Data validation: PASSED')
    except Exception as e:
        raise (e)
    
    # preserve input data
    drop_cols = [c for c in start_cols if c in df.columns]
    df_new = df.dropna(subset = drop_cols).reset_index(drop = True).copy()

    try:
        # year
        df_new['year'] = df_new.year.astype(int)
        # only include model years up to until last year
        df_new = df_new[df_new.year.between(2000, datetime.today().year - 1)]
        # convert to year diff from current year to emphasize age
        df_new.loc[:, 'model_age'] = df_new.apply(lambda x: datetime.today().year - x['year'], axis=1)
        print ('PASSED: model_age')
    except:
        raise Exception('FAILED: model_age')
    
    try:
        # price
        df_new = df_new[df_new.price.between(50000, 2000000)]
        
        # price_q = df_new.groupby('year')['price'].describe().loc[:, ['25%', '50%', '75%']]
        # price_q.loc[:, 'IQR'] = (price_q.loc[:, '75%'] - price_q.loc[:, '25%'])
        # price_q.loc[:, 'upper'] = price_q.loc[:, '75%'] + 1.5*price_q.loc[:, 'IQR']
        # df_new.loc[:, 'price_check'] = df_new.apply(lambda x: 1 if x['price'] <= price_q.loc[x['year'], 'upper'] else 0, axis=1)
        # df_new = df_new[(df_new.price_check == 1) & (df_new.price >= 50000)]
        
        print ('PASSED: price')
    except:
        print('FAILED: price')
        pass
        
    try:
        # transmission-fuel_type
        df_new = df_new[df_new.fuel_type.isin(['GASOLINE', 'DIESEL']) & \
                        df_new.transmission.isin(['AUTOMATIC', 'MANUAL'])]
        df_new.loc[:, 'transmission_fuel'] = df_new.apply(lambda x: '-'.join([x['transmission'], x['fuel_type']]), axis=1)
        print ('PASSED: transmission_fuel')
    except:
        raise Exception('FAILED: transmission_fuel')
    
    try:
        # quantile removal of outliers
        df_new = df_new[df_new.mileage.between(1000, 250000)]
        mileage_q = df_new.groupby('year')['mileage'].describe()
        mileage_q.loc[:, 'IQR'] = mileage_q.loc[:, '75%'] - mileage_q.loc[:, '25%']
        mileage_q.loc[:, 'low_limit'] = mileage_q.apply(lambda x: max(0, x['25%'] - 1.5*x['IQR']), axis=1)
        mileage_q.loc[:, 'upp_limit'] = mileage_q.apply(lambda x: min(250000, x['75%'] + 1.5*x['IQR']), axis=1)
        df_new = df_new[df_new.apply(lambda x: mileage_q.loc[x['year'], 'low_limit'] <= x['mileage'] \
                                     <= mileage_q.loc[x['year'], 'upp_limit'], axis=1)]
        
        def mileage_Z_score(mileage : float, 
                            year : [int, float], 
                            ref : pd.DataFrame) -> float:
            # ref = mileage_q
            if pd.notna(ref.loc[year, 'std']):
                return (mileage - ref.loc[year, 'mean'])/ref.loc[year, 'std']
            else:
                return (mileage - ref.loc[year, 'mean'])/ref.loc[year, 'mean']
        
        df_new.loc[:, 'mileage_grade'] = df_new.apply(lambda x: mileage_Z_score(x['mileage'], x['year'], mileage_q), axis=1)
        print ('PASSED: mileage_grade')
    except:
        raise Exception('FAILED: mileage_grade')
    
    try:
        # body_type
        df_new = df_new[df_new.body_type.isin(['SUV', 'SEDAN', 'PICKUP TRUCK',
                                        'VAN', 'HATCHBACK', 'CROSSOVER'])]
        print ('PASSED: body_type')
    except:
        raise Exception('FAILED: body_type')
    
    # collect required final features for model training
    df_new = df_new.reset_index(drop = True)
    
    if df_new.isnull().sum().sum():
        raise Exception('FAILED: Summary of non-null values\n {}'.format(df_new.isnull().sum()))
    else:
        print ('PASSED: No null values found. Total of {0} rows and {1} columns'.format(len(df_new), len(df_new.columns)))
    
    print ('End task: {}'.format(datetime.now().time().strftime('%H:%M:%S')))
    print ('Task duration: {0:1.2f} s'.format(time.time() - start_time))
    return df_new

@st.cache_data
def train_test_prep(df, 
                    test_size = 0.2, 
                    rs = 101, 
                    scaling = False) -> dict:
    '''
    Prepare data for model training

    Parameters
    ----------
    df : filtered dataframe
    test_size : float, optional
        Ratio of test size compared to all data. The default is 0.2.
    rs : int, optional
        random state. The default is 101.

    Returns
    -------
    dict containing X_train, X_test, y_train, y_test, enc, scaler

    '''
    # setup training data
    y = df['price']
    
    try:
        drop_cols = ['price', 'platform', 'url', 'num_photos', 'date_listed']
        drop_cols = [drop for drop in drop_cols if drop in df.columns]
        X = df.drop(drop_cols, axis=1)
    except:
        X = df.drop(['price'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size,
                                                        random_state = rs)
    
    # get object data types
    object_cols = X.select_dtypes(include=[object]).columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    
    # OneHotEncoding
    enc = OneHotEncoder(handle_unknown = 'ignore')
    enc.fit(X[object_cols])
    
    # Construct one hot encoded features
    X_train_obj = pd.DataFrame(enc.transform(X_train[object_cols]).toarray(), 
                               columns = enc.get_feature_names_out())
    X_test_obj = pd.DataFrame(enc.transform(X_test[object_cols]).toarray(),
                              columns = enc.get_feature_names_out())
    
    # StandardScaler
    if scaling:
        scaler = StandardScaler()
        ## construct scaled numerical features
        X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]),
                                   columns = num_cols).reset_index(drop = True)
        X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]),
                                  columns = num_cols).reset_index(drop = True)
    else:
        scaler = None
        X_train_num = X_train[num_cols].reset_index(drop = True)
        X_test_num = X_test[num_cols].reset_index(drop = True)
    
    # construct X_train and X_test
    X_train = pd.concat([X_train_obj, X_train_num], axis=1)
    X_test = pd.concat([X_test_obj, X_test_num], axis=1)
    
    #return X_train, X_test, y_train, y_test, enc
    return {'X_train' : X_train,
            'X_test' : X_test,
            'y_train' : y_train,
            'y_test' : y_test,
            'enc' : enc,
            'scaler' : scaler}

@st.cache_data
def test_prep(X : pd.DataFrame, 
              _enc, 
              scaling : bool = False,
              scaler = None) -> pd.DataFrame:
    '''
    Prepare test data for model prediction

    Parameters
    ----------
    X : feature engineered input dataframe
    enc : encoder object
    scaling : bool
        Switch whether to apply scaling to numerical data

    Returns
    -------
    pd.DataFrame of X_test
    '''
    # get object data types
    object_cols = X.select_dtypes(include=[object]).columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    
    # StandardScaler
    if scaling:
        scaler = StandardScaler()
        ## construct scaled numerical features
        X_num = pd.DataFrame(scaler.fit_transform(X[num_cols]),
                                   columns = num_cols).reset_index(drop = True)
        
    else:
        X_num = X[num_cols].reset_index(drop = True)
    
    X_obj = pd.DataFrame(_enc.transform(X[object_cols]).toarray(), 
                               columns = _enc.get_feature_names_out())
    
    X_test = pd.concat([X_obj, X_num], axis=1)
    return X_test

@st.cache_resource
def xgb_model(X_train : pd.DataFrame, 
              y_train : pd.DataFrame, 
              grid_search : bool = False, 
              random_state = 101) -> dict:
    '''
    
    Parameters
    ----------
    X_train : dataframe
    y_train : dataframe
    grid_search: boolean, optional
        option to use grid search or not. Default option is False.
    random_state : int, optional
        random number generator seed. Default value is 101.
        
    Returns
    -------
    dict
    Keys : Values
    - 'clf' : trained xgb model
    - 'runtime' : total training time
    
    '''
    print ('Training XGBoost Model')
    start_time = time.time()
    if grid_search:
        # XGBRegressor
        param_grid = {'n_estimators' : [450, 500, 550],
                      'max_depth' : [3, 6, 9],
                      'learning_rate': [0.12, 0.15, 0.18]}
        model_xgbreg = XGBRegressor(tree_method = 'hist',
                                    objective = 'reg:squarederror',
                                    random_state = random_state)
        clf = GridSearchCV(estimator = model_xgbreg,
                                       param_grid = param_grid,
                                       scoring = 'neg_mean_squared_error',
                                       cv = 5,
                                       verbose = 2,
                                       n_jobs = 4)
        
        #dump(clf, 'carmax_xgb_gridsearch.joblib')
                
    else:
        clf = XGBRegressor(objective = 'reg:squarederror',
                               random_state = random_state)
        
    clf.fit(X_train, y_train)
    runtime = time.time() - start_time
    print ('Finished training XGBoost model in {0:.3f} seconds'.format(runtime))
    return {'model' : clf,
            'runtime': runtime}

@st.cache_data
def train_models(model_list, 
                 X_train, 
                 y_train, 
                 grid_search : bool = True) -> dict:
    '''
    Train selected models using training data
    
    Parameters
    -----------
    model_list: 
        list of strings of model names, ie XGB, RF, linear, SVM, etc
    X_train: 
        independent variables training data
    y_train: 
        dependent variables training data
    grid_search: bool
        Switch whether to perform gridsearch or not
        
    Returns
    -------
    dict containing ['name', 'model', 'runtime'] for each model to be trained
    
    '''
    model_dict = {}
    for model in model_list:
        print ('Start training {0} model: {1}'.format(model.lower(), 
                                                    datetime.now().time().strftime('%H:%M:%S')))
        start_time = time.time()
        func = globals()[model.lower() + '_model']
        m = func(X_train, y_train, grid_search)
        model_dict[model] = {'name' : model,
                             'model': m['model'],
                             'runtime' : m['runtime']}
        print ('End training {0} model: {1:1.2f}'.format(model.lower(),
                                                  time.time() - start_time))
    return model_dict

@st.cache_resource
def eval_models(_model_dict : dict, 
                X_test : pd.DataFrame, 
                y_test : pd.DataFrame) -> pd.DataFrame:
    
    '''
    Parameters
    ----------
    _model_dict: dict 
        output trained models (see train_models)
    X_test: pd.DataFrame
        X Test data
    y_test: pd.DataFrame
        Y Test data
    
    Returns
    -------
    pd.DataFrame
        MAE & RMSE of each trained model
    
    '''
    d = {}
    for model in _model_dict.keys():
        d[_model_dict[model]['name']] = {'RMSE': np.sqrt(mean_squared_error(y_test, _model_dict[model]['model'].predict(X_test))),
                            'MAE' : mean_absolute_error(y_test, _model_dict[model]['model'].predict(X_test))}

    return pd.DataFrame.from_dict(d).T

@st.cache_data
def import_appraisal_requests(df_data):
    '''
    Import and clean/fill-in missing data from appraisal request data
    '''
    
    appraisal_url = 'https://docs.google.com/spreadsheets/d/1Pin8YL_kVzABE6iK1so7LN2FtyqqGqRt9UlSTDWlz9M/edit#gid=0'
    df_appraisal_reqs = config_lica.read_gsheet(appraisal_url, 'log')
    # preserve original appraisal data
    df_temp = df_appraisal_reqs.copy()
    
    # data cleaning
    df_temp.loc[:, 'date'] = pd.to_datetime(df_temp.loc[:, 'created_at'])
    
    # check import reference data presence
    global_data = ['carmax_makes', 'carmax_models', 'body_types', 'ph_locations']
    if all(True if d in globals() else False for d in global_data):
        pass
    else:
        config_lica.main('CARMAX', 'CLEAN')
    
    makes_list = config_lica.carmax_makes
    models_list = config_lica.carmax_models
    
    df_temp.loc[:, 'make'] = df_temp.apply(lambda x: config_lica.clean_make(x['make'], makes_list), axis=1)
    df_temp.loc[:, 'model'] = df_temp.apply(lambda x: config_lica.clean_model(x['model'], 
                                                                          makes_list, models_list), axis=1)
    df_temp.loc[:, 'year'] = df_temp.apply(lambda x: config_lica.clean_year(x['year']), axis=1)
    df_temp.loc[:, 'fuel_type'] = df_temp.apply(lambda x: config_lica.clean_fuel_type(x['fuel_type']), axis=1)
    df_temp.loc[:, 'transmission'] = df_temp.apply(lambda x: config_lica.clean_transmission(x['transmission']), axis=1)
    
    def get_body_type(make : str, 
                      model : str) -> [str, np.NaN]:
        sim = df_data[(df_data.make == make) & (df_data.model == model)]
        try:
            return sim['body_type'].mode().iloc[0]
        except:
            return np.NaN
    
    df_temp.loc[:, 'body_type'] = df_temp.apply(lambda x: get_body_type(x['make'], x['model']), axis=1)
    #df_temp = df_temp[df_temp.body_type.notna()]
    
    def approx_mileage(x : float, df) -> float:
        # extract mileage lower and upper bounds
        mileage_bounds = []
        for n in str(x['mileage']).split('-'):
            if n != '100000':
                mileage_bounds.append(float(re.sub(',', '', n)))
            else:
                mileage_bounds = (100001, 500000)
                break
        
        # filter from data entries which satisfies similar mileage, transmission, and make
        approx_mileage = df[df.mileage.between(mileage_bounds[0], mileage_bounds[1]) &
                      (df.transmission == x['transmission']) & 
                      (df.make == x['make'])]['mileage'].mean()
        
        # if match exists
        if pd.notna(approx_mileage):
            est_mileage = df[df.mileage.between(mileage_bounds[0], mileage_bounds[1])]['mileage'].mean()
            est_mileage = round(approx_mileage/1000.0)*1000.0
        else:
            est_mileage = np.NaN
            
        return est_mileage
        
    df_temp.loc[:, 'approx_mileage'] = df_temp.apply(lambda x: approx_mileage(x, df), axis=1)
    # standardize columns
    df_temp = df_temp.rename(columns = {'mileage' : 'mileage_bracket',
                                        'approx_mileage' : 'mileage'})
    df_temp.columns = ['_'.join(c.strip().lower().split(' ')) for c in df_temp.columns]
    
    # remove duplicates
    drop_cols = [c for c in start_cols if c in df_temp.columns]
    df_temp.drop_duplicates(subset = drop_cols, inplace = True)
    
    return df_temp

def request_select(df_data : pd.DataFrame):
    '''
    Displays appraisal request info
    Able to select specific appraisal request to predict value

    Parameters
    ----------
    df_data : dataframe

    Returns
    -------
    df_retention : dataframe
        df_retention with updated values

    '''
    # Reprocess dataframe entries to be displayed
    df_app = df_data.copy()
    
    # table settings
    df_display = df_app.sort_values('date', ascending = False)
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection('single', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_column('id', 
                        headerCheckboxSelection = True,
                        width = 100)
    gridOptions = gb.build()
    
    # selection settings
    data_selection = AgGrid(
        df_display,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=True,
        height= min(33*len(df_display), 400), 
        reload_data=False)
    
    selected = data_selection['selected_rows']
    
    if selected:           
        # row/s are selected
        selected_row = [selected[0]['_selectedRowNodeInfo']['nodeRowIndex']]
        df_selected = [df_display[df_display.index.isin([selected_row[checked_items]])]
                              for checked_items in range(len(selected_row))]
        
        df_list = pd.concat(df_selected)
        # df_list = df_display.iloc[selected]
        #st.dataframe(df_list)    

    else:
        st.write('Click on an entry in the table to display appraisal request data.')
        df_list = df_display.iloc[[0]]
        
    return df_list

def appraisal_shap(df_data, enc, shap_values):
    pass

def find_similar_cars(request_info, df):
    '''
    Find similar cars to request car in market
    
    Parameters
    -----------
    request_info : dataframe
        dataframe of row with request info of car
    df : dataframe
        dataframe of compiled car info in market and backend
    '''
    df_ = df.copy()
    df_.loc[:, 'date_listed'] = pd.to_datetime(df_['date_listed'])
    try:
        similar_cars = df_[(df_.make == request_info['make'].values[0]) & 
                          (df_.model == request_info['model'].values[0]) &
                          (df_.transmission == request_info['transmission'].values[0]) &
                          (df_.year.between(int(request_info['year'].values[0]) - 1,
                                           int(request_info['year'].values[0]) + 1)) &
                          (df_.mileage.between(request_info['mileage'].values[0] - 25000,
                                              request_info['mileage'].values[0] + 25000))].sort_values('date_listed', 
                                                                                                      ascending = False)
        # remove outliers in terms of price
        try:
            q25 = np.quantile(similar_cars.price, 0.25)
            q75 = np.quantile(similar_cars.price, 0.75)
            iqr = q75 - q25
            upper = q75 + 1.5*iqr
            lower = q25 - 1.5*iqr
            similar_cars = similar_cars[similar_cars.price.between(lower, upper)]
        except:
            pass
        
        #similar_cars.loc[:, 'date_listed'] = similar_cars.loc[:, 'date_listed'].dt.strftime('%Y-%m-%d')
        similar_cars = similar_cars.drop_duplicates(subset = ['make', 'model', 'year',
                                                              'transmission', 'mileage',
                                                              'price', 'location'],
                                                    keep = 'first')
    except:
        similar_cars = pd.DataFrame()
    
    return similar_cars
    

if __name__ == '__main__':
    # column names required for this func (no null values)
    start_cols = ['price', 'make', 'model', 'year', 'mileage',
                   'transmission', 'fuel_type', 'body_type']
    feat_cols = ['price', 'make', 'model', 'model_age',
                 'mileage_grade', 'transmission_fuel', 'body_type']
    
    ## Import backend and competitor data from BQ/Redash
    df, comp_data, backend_data = import_data()
    
    
    
    st.info('Feature engineering data')
    ## Validate input data and perform feature engineering
    df_data = feature_engineering(df)
    
    ## Retrieve/Train XGBoost model
    st.info('Preparing train test data')
    train_test_dict = train_test_prep(df_data[feat_cols])
    
    st.info('Obtaining trained XGBoost model')
    filename = 'appraisal_xgb_model.joblib'
    bucket_name = 'lica-aiml-models'
    model_file = 'carmax/' + filename
    comp_table_id = 'absolute-gantry-363408.carmax.competitors_compiled'
    model_mod_date = config_lica.check_model_date(filename)
    comp_mod_date = config_lica.check_table_date(comp_table_id)
    
    try:
        models = config_lica.get_from_gcloud(filename)
        st.info(f'XGB model ({filename}) downloaded from GCS in {bucket_name}')
        evals = eval_models(models, 
                            train_test_dict['X_test'], 
                            train_test_dict['y_test'])
    except:
        # only execute if need to train model
        st.info('Training XGB model')
        models = train_models(['XGB'], 
                              train_test_dict['X_train'], 
                              train_test_dict['y_train'], 
                              )
        try:
            
            config_lica.upload_to_gcloud(models, filename,
                                          bucket_name = bucket_name)
            st.info(f'XGB model ({filename}) uploaded to GCS in {bucket_name}')
        except:
            pass
        
        evals = eval_models(models, 
                            train_test_dict['X_test'], 
                            train_test_dict['y_test'])
    # if ((datetime.today().date() - model_mod_date.date()).days <= 7) or not \
    #     ((datetime.today().date() - comp_mod_date.date()).days <= 7):
    #     models = {'XGB' : config_lica.get_from_gcloud(model_file)}
    # else:
    #     ## only execute if need to train model
    #     models = train_models(['XGB'], 
    #                           train_test_dict['X_train'], 
    #                           train_test_dict['y_train'], 
    #                           )
    #     try:
    #         filename = 'appraisal_xgb_model.joblib'
    #         bucket_name = 'lica-aiml-models'
    #         config_lica.upload_to_gcloud(models, filename,
    #                                      bucket_name = bucket_name)
    #         print(f'XGB model ({filename}) uploaded to GCS in {bucket_name}')
    #     except:
    #         pass
    
    ## TODO : Import carmax appraisal requests
    st.info('Importing appraisal requests data')
    df_appraisal_reqs = import_appraisal_requests(df_data)
    st.info('Feature engineering appraisal request data')
    df_appraisal = feature_engineering(df_appraisal_reqs)
    # prepare test data
    st.info('Preparing Appraisal Request test data')
    appraisal_test = test_prep(df_appraisal[feat_cols[1:]], 
                               _enc = train_test_dict['enc'])
    st.info('Predicting values for appraisal requests')
    appraisal_pred = models['XGB']['model'].predict(appraisal_test)
    df_appraisal.loc[:, 'predicted_value'] = appraisal_pred
    
    ## TODO : Output predicted appraised value for each entry (with shap breakdown)
    show_cols = ['id', 'date', 'status', 'intention', 'client_name',
                 'make', 'model', 'year', 'transmission', 'fuel_type',
                 'mileage', 'mileage_bracket', 'body_type', 'issues',
                 'with_inspection', 'initial_appraised_value', 
                 'min_value', 'max_value']
    
    df_request = request_select(df_appraisal[show_cols + ['predicted_value']])
    ### Give rating for each appraisal request/entry
    if df_request is not None:
        # visualize shap
        explainer = shap.Explainer(models['XGB']['model'].best_estimator_)
        shap_values = explainer(appraisal_test.iloc[[df_request.index[0]]])
        
        st.subheader('Appraised Value Breakdown')
        # waterfall plot for first observation
        n_categories = []
        for feat in train_test_dict['enc'].feature_names_in_:
            n = df_data[feat].nunique()
            n_categories.append(n)
        
        new_shap_values = []
        for values in shap_values.values:
            
            #split shap values into a list for each feature
            values_split = np.split(values , np.cumsum(n_categories))
            
            #sum values within each list
            values_obj_sum = [sum(l) for l in values_split[:-1]]
        
        values_num_sum = shap_values.values[0, -2:]
        
        new_shap_values.append(list(values_obj_sum) + list(values_num_sum))
        
        #replace shap values
        shap_values.values = np.array(new_shap_values)
        
        #replace data with categorical feature values
        row = df_appraisal[df_appraisal.id == df_request.id.values[0]][feat_cols[1:]]
        new_data = np.array(row)
        shap_values.data = np.array(new_data)
        
        #update feature names
        shap_values.feature_names = list(df_appraisal[feat_cols[1:]].columns)
        fig = shap.plots.waterfall(shap_values[0], show = True)
        st.pyplot(fig)
        
        # get original car info
        request_info = df_appraisal_reqs[df_appraisal_reqs.id == df_request.id.values[0]]
        
        ## Find similar cars in the market & perform outlier removal
        similar_cars = find_similar_cars(request_info, df)
        
        gp_lower, gp_upper = 0.05, 0.35
        offer_rate = 0.8
        demand_lower, demand_upper = 14, 90
        if len(similar_cars):
            st.dataframe(similar_cars)
            
            with st.expander('INFORMATION', expanded = True):
                market_value = round(similar_cars['price'].mean(), 2)
                st.info(f'Market value: {market_value}')
                predicted_value = round(df_request['predicted_value'].values[0], 2)
                st.info(f"Predicted Appraisal value: {predicted_value}")
                
                # decision tree
                
                
                ## gp
                profit_margin = market_value - predicted_value * offer_rate
                projected_gp = profit_margin/(predicted_value*offer_rate)
                st.info('Projected GP: {:.2f}'.format(projected_gp))
                
                ## demand
                similar_cars.loc[:,'date_listed'] = pd.to_datetime(similar_cars.loc[:, 'date_listed'],
                                                                   yearfirst = True,
                                                                   errors = 'ignore')
                st.dataframe(similar_cars.info())
                similar_cars_recent = similar_cars[similar_cars.date_listed.dt.date >= (datetime.today().date() - timedelta(days = 180))]
                
                if len(similar_cars_recent):
                    avg_time_between_listings = abs(similar_cars_recent.date_listed.diff().mean().days)
                else:
                    avg_time_between_listings = np.NaN
                    
                st.info('Average days between listings of similar cars: {}'.format(avg_time_between_listings))
                
            with st.expander('VERDICT', expanded = True):
                if pd.notna(projected_gp):
                    if (gp_lower <= projected_gp <= gp_upper):
                        gp_msg = 'Projected GP WITHIN accepted range.'
                        st.info(gp_msg)
                        gp_cond = True
                    else:
                        gp_msg = 'Projected GP OUTSIDE accepted range'
                        st.warning(gp_msg)
                        gp_cond = False
                else:
                    gp_msg = 'Projected GP cannot be calculated'
                    st.warning(gp_msg)
                    gp_cond = False
                
                
                if pd.notna(avg_time_between_listings):
                    if (demand_lower <= avg_time_between_listings <= demand_upper):
                        demand_msg = 'Estimated demand is WITHIN accepted range.'
                        st.info(demand_msg)
                        demand_cond = True
                    else:
                        demand_msg = 'Estimated demand is OUTSIDE accepted range.'
                        st.warning(demand_msg)
                        demand_cond = False
                else:
                    demand_msg = 'Estimated demand cannot be calculated.'
                    st.warning(demand_msg)
                    demand_cond = False
            
        else:
            st.warning('No similar cars in the market. This car transaction is risky due to no data on potential profit and demand.')
        
        
    with st.expander('TOOL PARAMETERS', expanded = False):
        st.markdown(f'''
                    **Gross Profit**:\n
                    -offer rate: {offer_rate}\n
                    -GP lower limit: {gp_lower}\n
                    -GP upper limit: {gp_upper}\n
                    
                    **Demand Thresholds**:\n
                    -Demand lower limit: {demand_lower}\n
                    -Demand upper limit: {demand_upper}\n
                    ''')
        
                    
                
