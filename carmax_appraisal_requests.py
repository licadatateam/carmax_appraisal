# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:57:26 2023

@author: carlo

"""
# system tools
import time
import re
from datetime import datetime, timedelta

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

from pytrends.request import TrendReq

# UI
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid
#import streamlit.components.v1 as components
import extra_streamlit_components as stx

st.set_option('deprecation.showPyplotGlobalUse', False)

global_data = ['carmax_makes', 'carmax_models', 'body_types', 'ph_locations']
if all(True if d in globals() else False for d in global_data):
    pass
else:
    config_lica.main('CARMAX', 'CLEAN')

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
    
    print ('End task: {}'.format(datetime.now().time().strftime('%H:%M:%S')))
    print('Task duration: {0:1.2f} s'.format(time.time() - start_time))
    return df, comp_data, backend_data

@st.cache_data
def feature_engineering(df_, df_ref = None):
    '''
    Executes feature engineering for dataset to prep for model training
    '''
    print ('Start task: {}'.format(datetime.now().time().strftime('%H:%M:%S')))
    start_time = time.time()
    
    try:
        col_check = [col  for col in start_cols if col not in df_.columns]
        if len(col_check):
            print ('''Data validation: FAILED\nMissing columns: {}'''.format(', '.join(col_check)))
        else:
            print ('Data validation: PASSED')
    except Exception as e:
        raise (e)
    
    # preserve input data
    drop_cols = [c for c in start_cols if c in df_.columns]
    df_new = df_.dropna(subset = drop_cols).reset_index(drop = True).copy()
    if df_ref is None:
        df_ref = df_.copy()
    else:
        pass

    try:
        # year
        df_new['year'] = df_new.year.astype(int)
        # only include model years up to until last year
        df_new = df_new[df_new.year.between(1990, datetime.today().year)]
        # convert to year diff from current year to emphasize age
        df_new.loc[:, 'model_age'] = df_new.apply(lambda x: datetime.today().year - x['year'], axis=1)
        print ('PASSED: model_age')
    except:
        raise Exception('FAILED: model_age')
    
    try:
        # price
        df_new = df_new[df_new.price.between(100000, 2500000)]
        
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
        df_q = df_ref[df_ref.mileage.between(1000, 250000)]
        mileage_q = df_q.groupby('year')['mileage'].describe()
        mileage_q.loc[:, 'IQR'] = mileage_q.loc[:, '75%'] - mileage_q.loc[:, '25%']
        mileage_q.loc[:, 'low_limit'] = mileage_q.apply(lambda x: max(0, x['25%'] - 1.5*x['IQR']), axis=1)
        mileage_q.loc[:, 'upp_limit'] = mileage_q.apply(lambda x: min(250000, x['75%'] + 1.5*x['IQR']), axis=1)
        # df_new = df_new[df_new.apply(lambda x: mileage_q.loc[x['year'], 'low_limit'] <= x['mileage'] \
        #                              <= mileage_q.loc[x['year'], 'upp_limit'], axis=1)]
        
        def mileage_Z_score(mileage : float, 
                            year : [int, float], 
                            ref : pd.DataFrame) -> float:
            # ref = mileage_q
            try:
                if (pd.notna(ref.loc[year, 'std'])) and (ref.loc[year, 'std'] > 0):
                    return (mileage - ref.loc[year, 'mean'])/ref.loc[year, 'std']
                else:
                    return (mileage - ref.loc[year, 'mean'])/ref.loc[year, 'mean']
            except:
                return 0
        
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
    
    if df_new[feat_cols[1:]].isnull().sum().sum():
        raise Exception('FAILED: Summary of non-null values\n {}'.format(df_new[feat_cols[1:]].isnull().sum()))
    else:
        print ('PASSED: No null values found. Total of {0} rows and {1} columns'.format(len(df_new), 
                                                                                        len(df_new[feat_cols[1:]].columns)))
    
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
        param_grid = {'n_estimators' : [650, 700, 750],
                      'max_depth' : [6],
                      'learning_rate': [0.16]}
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

@st.cache_resource
def import_model():
    
    
    filename = 'appraisal_xgb_model.joblib'
    bucket_name = 'lica-aiml-models'
    comp_table_id = 'absolute-gantry-363408.carmax.competitors_compiled'
    model_mod_date = config_lica.check_model_date(filename)
    comp_mod_date = config_lica.check_table_date(comp_table_id)
    # updated competitor data but not updated model, retrain model
    #if (comp_mod_date.day <= 7) and (model_mod_date.day > 7):
    try:
        if (model_mod_date.day - comp_mod_date.day) >= 0:
            models = config_lica.get_from_gcloud(filename)
            print(f'SUCCESS: XGB model ({filename}) from {bucket_name} GCS bucket.')
            
        else:
            print('Competitor data updated. XGB model need to be retrained.')
            models = train_models(['XGB'], 
                                  train_test_dict['X_train'], 
                                  train_test_dict['y_train'], 
                                  )

    except:
        print(f'FAILED: Import XGB model {filename} from {bucket_name} GCS bucket.')
        
        # only execute if need to train model
        print('Training XGB model..')
        models = train_models(['XGB'], 
                              train_test_dict['X_train'], 
                              train_test_dict['y_train'], 
                              )
        try:

            config_lica.upload_to_gcloud(models, filename,
                                          bucket_name = bucket_name)
            print(f'SUCCESS: Upload XGB model {filename} to {bucket_name} GCS bucket.')
        except:
            print(f'FAILED: Upload XGB model {filename} to {bucket_name} GCS bucket.')
            pass
    
    finally:
        evals = eval_models(models, 
                            train_test_dict['X_test'], 
                            train_test_dict['y_test'])
        return models, evals

@st.cache_data
def approx_mileage(mileage, make, transmission, df) -> float:
    # extract mileage lower and upper bounds
    mileage_bounds = []
    try:
        for n in str(mileage).split('-'):
            if n != '100,000':
                mileage_bounds.append(float(re.sub(',', '', n)))
            else:
                mileage_bounds = (100001, 250000)
                break
    except:
        mileage_bounds = (0, 250000)
    
    # filter from data entries which satisfies similar mileage, transmission, and make
    approx_mileage = df[df.mileage.between(mileage_bounds[0], mileage_bounds[1]) &
                  (df.transmission == transmission) & 
                  (df.make == make)]['mileage'].mean()
    
    # if match exists
    if pd.notna(approx_mileage):
        est_mileage = df[df.mileage.between(mileage_bounds[0], mileage_bounds[1])]['mileage'].mean()
        est_mileage = round(approx_mileage/1000.0)*1000.0
    else:
        est_mileage = np.NaN
        
    return est_mileage

@st.cache_data
def get_body_type(df, 
                  make : str, 
                  model : str) -> [str, np.NaN]:
    sim = df[(df.make == make) & (df.model == model)]
    try:
        return sim['body_type'].mode().iloc[0]
    except:
        return 'SEDAN'

@st.cache_data
def import_appraisal_requests(df_data):
    '''
    Import and clean/fill-in missing data from appraisal request data
    '''
    
    # appraisal_url_gs = 'https://docs.google.com/spreadsheets/d/1Pin8YL_kVzABE6iK1so7LN2FtyqqGqRt9UlSTDWlz9M/edit#gid=0'
    # df_appraisal_reqs_gs = config_lica.read_gsheet(appraisal_url_gs, 'log')
    acct = bq_functions.get_acct()
    client, credentials = bq_functions.authenticate_bq(acct)
    try:
        mod_date = config_lica.check_table_date('absolute-gantry-363408.carmax.appraisal_request')
        
        if (datetime.today().date() - mod_date.date()).days <= 7:
            df_temp = bq_functions.query_bq('.'.join([credentials.project_id, 
                                                      'carmax', 
                                                      'appraisal_request']), client)
            print ('Appraisal requests queried from BQ.')
        else:
            raise Exception ('Carmax appraisal requests due for update.')
        
    except:
        appraisal_url = 'http://app.redash.licagroup.ph/api/queries/211/results.csv?api_key=aUXSNW7pPb6fHfS8EmnTjnU2aR27OdDkTz6i4XOp'
        df_appraisal_reqs = pd.read_csv(appraisal_url, parse_dates = ['created_at', 'updated_at'])
        
        # preserve original appraisal data
        df_temp = df_appraisal_reqs.copy()
        
        # data cleaning
        df_temp.loc[:, 'date'] = pd.to_datetime(df_temp.loc[:, 'created_at'])
        
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
                return 'SEDAN'
        
        df_temp.loc[:, 'body_type'] = df_temp.apply(lambda x: get_body_type(x['make'], x['model']), axis=1)
        #df_temp = df_temp[df_temp.body_type.notna()]
        
        # remove entries with no mileage
        df_temp = df_temp[~df_temp.mileage.isnull()]
        df_temp.loc[:, 'approx_mileage'] = df_temp.apply(lambda x: approx_mileage(x['mileage'], 
                                                                                  x['make'], 
                                                                                  x['transmission'], 
                                                                                  df), axis=1)
        # standardize columns
        df_temp = df_temp.rename(columns = {'mileage' : 'mileage_bracket',
                                            'approx_mileage' : 'mileage'})
        df_temp.columns = ['_'.join(c.strip().lower().split(' ')) for c in df_temp.columns]
        
        # remove duplicates
        drop_cols = [c for c in start_cols if c in df_temp.columns]
        df_temp.drop_duplicates(subset = drop_cols, 
                                inplace = True)
        # remove NaN
        df_temp.dropna(subset = ['make', 'model', 'year', 'mileage', 'transmission',
                                 'fuel_type'], inplace = True)
        
        df_temp = df_temp.reset_index(drop = True)
        
        try:
            bq_functions.bq_write(df_temp,
                                  credentials,
                                  'carmax',
                                  'appraisal_request',
                                  client)
            print ('SUCCESS: Carmax appraisal requests updated on BQ')
            
        except:
            print ('FAILED: Unable to update Carmax appraisal requests on BQ')
            pass
        
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
    df_display.loc[:, 'date'] = df_display.date.dt.strftime('%Y/%m/%d')
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection('single', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_default_column(width = 100)
    gb.configure_column('date', 
                        headerCheckboxSelection = True)
    gb.configure_column('id', hide = True)
    if 'url' in df_app.columns:
        gb.configure_column('url', width = 450,
                            singleClickEdit = True)
    try:
        custom_col_size = ['date',
                           'model', 
                           'market_value',
                           'market_value_min',
                           'market_value_max',
                           'market_value_std']
        gb.configure_columns(column_names = custom_col_size,
                             width = [150]*len(custom_col_size))
    except:
        pass
    
    gridOptions = gb.build()
    

    # selection settings
    data_selection = AgGrid(
        df_display,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        height= min(33*len(df_display), 400), 
        reload_data=False)
    
    selected = data_selection['selected_rows']
    
    if selected is not None:           
        # row/s are selected
        st.write(selected)
        selected_row = [selected[0]['_selectedRowNodeInfo']['nodeRowIndex']]
        df_list = df_display.iloc[selected_row]
        
    else:
        #df_list = df_display.iloc[[0]]
        df_list = None
        
    return df_list

@st.cache_data
def import_available_units():

    df_cmx = pd.read_csv('http://app.redash.licagroup.ph/api/queries/8/results.csv?api_key=AHzu1kwBcFORZzW2kYe6YVbK5m5juVrPp6oTdueC',
                         parse_dates = ['PO Date'])
    
    df_cmx.columns = ['_'.join(c.split(' ')).lower() for c in df_cmx.columns]
    
    # df_cmx.loc[:, 'make'] = df_cmx.make.str.upper()
    # df_cmx.loc[:, 'model'] = df_cmx.model.str.upper()
    # df_cmx.loc[:, 'transmission'] = df_cmx.transmission.str.upper()
    # df_cmx.loc[:, 'fuel_type'] = df_cmx.fuel_type.str.upper()
    try:
        makes_list = config_lica.carmax_makes
        models_list = config_lica.carmax_models
    except:
        config_lica.main('CARMAX', 'CLEAN')
        makes_list = config_lica.carmax_makes
        models_list = config_lica.carmax_models
    
    df_cmx.loc[:, 'make'] = df_cmx.apply(lambda x: config_lica.clean_make(x['make'], makes_list), axis=1)
    df_cmx.loc[:, 'model'] = df_cmx.apply(lambda x: config_lica.clean_model(x['model'], 
                                                                          makes_list, models_list), axis=1)
    df_cmx.loc[:, 'year'] = df_cmx.apply(lambda x: config_lica.clean_year(x['year']), axis=1)
    df_cmx.loc[:, 'fuel_type'] = df_cmx.apply(lambda x: config_lica.clean_fuel_type(x['fuel_type']), axis=1)
    df_cmx.loc[:, 'transmission'] = df_cmx.apply(lambda x: config_lica.clean_transmission(x['transmission']), axis=1)
    df_cmx.loc[:, 'body_type'] = df_cmx.apply(lambda x: get_body_type(df,
                                                                      x['make'], 
                                                                      x['model']), axis=1)
    df_cmx.loc[:, 'is_sale'] = df_cmx.is_sale.apply(lambda x: True if x else False)
    df_cmx.loc[:, 'plate_no'] = df_cmx.plate_no.apply(lambda x: x.upper().strip() if pd.notna(x) else np.NaN)
    df_cmx.reset_index(inplace = True)
    df_cmx = df_cmx.rename(columns = {'index' : 'id',
                                      'po_date': 'date'})
    df_cmx.loc[:, 'sale_price'] = df_cmx.apply(lambda x: x['sale_price'] if (pd.notna(
        x['sale_price']) and (x['is_sale'] == True)) else np.nanmin([x['sale_price'], x['selling_price']]) - 10000, axis=1)
    # website hardcode
    # df_cmx.loc[:, 'asking_price'] = df_cmx - 10000

    return df_cmx

@st.cache_data
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
    df_.loc[:, 'date_listed'] = pd.to_datetime(df_.loc[:,'date_listed'])
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
        similar_cars.loc[:, 'date_listed'] = pd.to_datetime(similar_cars.loc[:, 'date_listed'])
    except:
        similar_cars = pd.DataFrame()
    
    return similar_cars


@st.cache_data
def import_bookings():
    '''
    Import all bookings data (financing requests and test drive bookigns)
    '''
    
    acct = bq_functions.get_acct()
    client, credentials = bq_functions.authenticate_bq(acct)
    
    try:
        mod_date = config_lica.check_table_date('absolute-gantry-363408.carmax.all_bookings')
        
        if (datetime.today().date() - mod_date.date()).days <= 7:
            df_bookings = bq_functions.query_bq('.'.join([credentials.project_id, 
                                                      'carmax', 
                                                      'all_bookings']), client)
            print ('all_bookings queried from BQ.')
        else:
            raise Exception ('Carmax all_bookings table due for update.')
        
    except:
        ## import data
        # http://app.redash.licagroup.ph/queries/178#191
        bookings_url = 'http://app.redash.licagroup.ph/api/queries/178/results.csv?api_key=oosB9hNzQZd5eDqfIGIBw2C29lM87YPd1TFpK95X'
        df_bookings = pd.read_csv(bookings_url, parse_dates = ['application_date'])
        
        ## clean data
        df_bookings.loc[:, 'make'] = df_bookings.apply(lambda x: config_lica.clean_make(x['make'], 
                                                                                        config_lica.carmax_makes), axis=1)
        df_bookings.loc[:, 'model'] = df_bookings.apply(lambda x: config_lica.clean_model(x['model'], 
                                                                                        config_lica.carmax_makes,
                                                                                        config_lica.carmax_models), axis=1)
        df_bookings.loc[:, 'year'] = df_bookings.apply(lambda x: config_lica.clean_year(x['year']), axis=1)
        
        ## remove NaNs and duplicates
        df_bookings = df_bookings.dropna(subset = ['make', 'model' , 'year'])
        df_bookings = df_bookings.drop_duplicates(subset = ['application_date', 'firstname', 
                                                            'lastname', 'vehicle_id', 'booking_date'],
                                                  keep = 'first')
        
        ## cleanup column names if needed
        df_bookings.columns = ['_'.join(c.split(' ')).lower() for c in df_bookings.columns]
        
        df_bookings = df_bookings.rename(columns = {'age_to_booking_date' : 'days_to_booking_date',
                                      'age_to_booking_date_(h)' : 'hours_to_booking_date'})
        
        def correct_hours(d):
            d = d.split(',')[-1].strip().split(':')
            h, m = float(d[0]), float(d[1])
            m = m/60.0
            return round(h + m, 1)
        
        df_bookings.loc[:, 'hours_to_booking_date'] = df_bookings['days_to_booking_date'].apply(correct_hours)
        
        def correct_days(d):
            d = d.split('day')[0].strip()
            
            try:
                return int(d)
            except:
                return 0
        
        df_bookings.loc[:, 'days_to_booking_date'] = df_bookings.days_to_booking_date.apply(correct_days)
        

        ## cleanup index
        df_bookings = df_bookings.reset_index(drop = True)
        
        try:
            bq_functions.bq_write(df_bookings,
                                  credentials,
                                  'carmax',
                                  'all_bookings',
                                  client)
            print('all_bookings table updated in BQ.')
        except:
            print('Failed to update all_bookings table in BQ.')
            
    return df_bookings

@st.cache_data
def import_tradeins():
    
    acct = bq_functions.get_acct()
    client, credentials = bq_functions.authenticate_bq(acct)
    
    try:
        mod_date = config_lica.check_table_date('absolute-gantry-363408.carmax.tradeins_consignment')
        
        if (datetime.today().date() - mod_date.date()).days <= 7:
            df_tc = bq_functions.query_bq('.'.join([credentials.project_id, 
                                                      'carmax', 
                                                      'tradeins_consignment']), client)
            print ('tradeins_consignments queried from BQ.')
        else:
            raise Exception ('Carmax tradeins_consignments table due for update.')
    
    except:
        key = 'http://app.redash.licagroup.ph/api/queries/134/results.csv?api_key=kET32aJo8w7r3U4Dy0xYYGDJ1RKbLHFwxq2AsvdC'
        df_tc = pd.read_csv(key, parse_dates = ['application_date'])
        
        ## clean data
        df_tc.loc[:, 'make'] = df_tc.apply(lambda x: config_lica.clean_make(x['make'], 
                                                                                        config_lica.carmax_makes), axis=1)
        df_tc.loc[:, 'model'] = df_tc.apply(lambda x: config_lica.clean_model(x['model'], 
                                                                                        config_lica.carmax_makes,
                                                                                        config_lica.carmax_models), axis=1)
        df_tc.loc[:, 'year'] = df_tc.apply(lambda x: int(config_lica.clean_year(x['year'])), axis=1)
        
        ## remove NaNs and duplicates
        df_tc = df_tc.dropna(subset = ['make', 'model' , 'year'])
        df_tc = df_tc.drop_duplicates(subset = ['application_date', 'firstname', 
                                                            'lastname', 'plate_no'],
                                                  keep = 'first')
        
        ## cleanup column names if needed
        df_tc.columns = ['_'.join(c.split(' ')).lower() for c in df_tc.columns]
        
        ## cleanup index
        df_tc = df_tc.reset_index(drop = True)
        
        try:
            bq_functions.bq_write(df_tc,
                                  credentials,
                                  'carmax',
                                  'tradeins_consignment',
                                  client)
            print ('tradeins_consignment table updated in BQ.')
        except:
            pass
            
        
    return df_tc
    
@st.cache_data
def calc_gp_score(market_value, 
                  appraised_value, 
                  asking_price = None):
    '''
    # maximum profit/score: appraised_value << market_value, asking_price << market_value, asking_price < appraised_value
    # minimum profit/score: asking_price >> market_value
    '''
    if (asking_price is None) or pd.isnull(asking_price):
        asking_price = appraised_value
    else:
        pass
    
    avg_value = (market_value + appraised_value)*0.5
    gp_score = 1 - 1/(1 + np.exp(-(5*(asking_price - avg_value)/avg_value))**2)
    # to map the scores

    return gp_score

@st.cache_data
def get_gtrends_score(request_info):

    search_terms = [' '.join([str(request_info[c].iloc[0])
                       for c in ['make', 'model', 'year']]).title()]
    
    geo = 'PH' # country
    #timeframe = '{} {}'.format(past_six, datetime.today().date())
    pytrends = TrendReq(hl='en-PH', tz = 480, retries = 3, backoff_factor = 0.25)
    pytrends.build_payload(search_terms, cat = 47, geo = geo,
                           timeframe='today 12-m')

    interest_over_time = pytrends.interest_over_time()
    
    recent_score = interest_over_time.iloc[26:, 0].sum()/interest_over_time.iloc[:,0].sum()
    
    return recent_score

@st.cache_data
def calc_demand_score(request_info, 
                      bookings, 
                      listings, 
                      appraisals,
                      tradeins):
    '''
    Parameters
    ----------
    request_info : DataFrame
        Single dataframe row of relevant car data
    bookings : DatFrame
        Bookings data (financing & test drive bookings)
    listings : DataFrame
        Filtered similar cars from output of find_similar_cars (request info)
    appraisals : DataFrame
    tradeins : DataFrame
    '''

    bookings.loc[:, 'year'] = bookings.loc[:, 'year'].apply(lambda x: int(x) if pd.notna(x) else np.NaN)
    bookings_filtered = bookings[(bookings.make == request_info.make.iloc[0]) & \
                                 (bookings.model == request_info.model.iloc[0]) & \
                                 (bookings.year.between(int(request_info.year.iloc[0]) - 1,
                                                        int(request_info.year.iloc[0]) + 1))]
    # utilize data from 6 months
    bookings_filtered_recent = bookings_filtered[bookings_filtered.application_date.apply(lambda x: True if x.date() >= (datetime.today().date() - timedelta(days = 180)) else False)]
    
    
    denom = sum([len(l) if len(l) != 0 else 0 for l in [listings, appraisals, tradeins]])
    
    try:
        bookings_listings_score = 1 - np.exp(-len(bookings_filtered_recent)/denom)
    
    except:
        bookings_listings_score = 1 - np.exp(-len(bookings_filtered_recent))
    
    #st.write(f'Bookings/Listings Score: {int(round(bookings_listings_score, 2)*100)}')
    
    # get views of similar listings within last 6 months over all similar listings/bookings
    if len(bookings_filtered_recent):
        
        views = bookings_filtered_recent.views.sum()    
        
        views_score = 1 - np.exp(-views/bookings_filtered.views.sum())
    
    else:
        
        views_score = 0
    
    #st.write(f'Views Score : {int(round(views_score, 2)*100)}')

    try:
        # 0-1 ratio
        interest_over_time = get_gtrends_score(request_info)
        gtrends_score = 1 - np.exp(-interest_over_time)
    
    except:
        gtrends_score = 0
    
    #st.write(f'Google Trends Score : {int(round(gtrends_score, 2)*100)}')
    
    demand_score = bookings_listings_score * 0.45 + \
        gtrends_score * 0.45 + views_score * 0.1

    return demand_score

def check_mileage(mileage, df):

    # extract mileage lower and upper bounds
    mileage_bounds = []
    try:
        for n in str(mileage).split('-'):
            if n != '100,000':
                mileage_bounds.append(float(re.sub(',', '', n)))
            else:
                mileage_bounds = (100001, 250000)
                break
            
        #mileage_bounds = [int(i) for i in re.findall('[0-9]+', re.sub(',', '', mileage))]
    except:
        mileage_bounds = (0, 250000)
    
    #st.write(mileage_bounds)
    num_cars = len(df[df.mileage.between(mileage_bounds[0], mileage_bounds[1])])
    return num_cars

@st.cache_data
def get_market_value(row, df):
    '''
    Get market value statistics of selected car listing
    
    Parameters
    ----------
    row : pandas dataframe row
        From .apply
    df : pandas dataframe
        main pandas dataframe containing all carmax competitor info
    
    '''
    similar_cars = find_similar_cars(row.to_frame().T, df)
    
    if len(similar_cars) >= 2:
        row['market_value'] = round(similar_cars.price.mean())
        row['market_value_min'] = round(min(similar_cars.price))
        row['market_value_max'] = round(max(similar_cars.price))
        row['market_value_std'] = round(similar_cars.price.std(), 2)
    else:
        try:
            market_value = round(similar_cars.price.iloc[0])
            if pd.isna(market_value):
                raise Exception
            else:
                pass
        except:
            market_value = row['predicted_value']
        
        finally:
            row['market_value'] = market_value
            row['market_value_min'] = market_value
            row['market_value_max'] = market_value
            row['market_value_std'] = 0
    
    return row

@st.cache_data
def mv_scores_apply(df, df2):
    
    df2 = df2.apply(lambda x: get_market_value(x, df), axis=1)
    
    if 'po_value' in df2.columns:
        df2.loc[:, 'projected_gp'] = df2.apply(
            lambda x: np.round(x['market_value'] - x['po_value']), axis=1)
    else:
        df2.loc[:, 'projected_gp'] = df2.apply(lambda x: np.round(
            x['market_value'] - x['asking_price']), axis=1)
        
    df2 = df2.apply(lambda x: calc_scores(x, df, df2), axis=1)

    return df2

@st.cache_data
def calc_scores(row, df, df2):
    '''
    
    Parameters:
    -----------
    - request_info: pandas dataframe single row
    - df: pandas dataframe
        all data
    - df2: pandas dataframe
        specific table (cmx listings, appraisals)
    '''
    try:                                                                                    
        request_info = row.to_frame().T
    except:
        request_info = row
    
    similar_listings = find_similar_cars(request_info, df)
    
    ## Find similar appraisal requests aside from selected
    similar_appraisals = df2[(df2.make == request_info.make.iloc[0]) &\
                     (df2.model == request_info.model.iloc[0]) &\
                     (df2.year.between(int(request_info.year.iloc[0]) - 1,
                                       int(request_info.year.iloc[0]) + 1))]
    
    similar_tradeins = df_tc[(df_tc.make == request_info.make.iloc[0]) &
                             (df_tc.model == request_info.model.iloc[0]) &
                             (df_tc.year.between(int(request_info.year.iloc[0]) - 1,
                                                 int(request_info.year.iloc[0]) + 1))]
    
    ## DEMAND
    try:
        listings_recent = similar_listings[similar_listings.date_listed.apply(lambda x: True if x.date() >= (datetime.today().date() - timedelta(days = 180)) else False)]
    except:
        listings_recent = pd.DataFrame()
    try:
        appraisals_recent = similar_appraisals[similar_appraisals.created_at.apply(lambda x: True if x.date() >= (datetime.today().date() - timedelta(days = 180)) else False)]
    except:
        appraisals_recent = pd.DataFrame()
    try:
        tradeins_recent = similar_tradeins[similar_tradeins.application_date.apply(lambda x: True if x.date() >= (datetime.today().date() - timedelta(days = 180)) else False)]
    except:
        tradeins_recent = pd.DataFrame()
    
    predicted_value = request_info['predicted_value'].values[0]
    market_value = request_info['market_value'].values[0]
    
    if pd.notna(market_value):
        pass
    else:
        market_value = predicted_value
    
    try:
        request_info['gp_score'] = int(calc_gp_score(market_value,
                                                 predicted_value,
                                                 request_info['asking_price'].values[0]) * 100)
    except:
        request_info['gp_score'] = int(calc_gp_score(market_value,
                                                 predicted_value,
                                                 request_info['po_value'].values[0]) * 100)
    
    request_info['demand_score'] = int(calc_demand_score(request_info,
                                     df_bookings,
                                     listings_recent,
                                     appraisals_recent,
                                     tradeins_recent) * 100)
    
    request_info['overall_score'] = int(request_info['gp_score'] * 0.8 + request_info['demand_score'] * 0.2)
    
    # convert back to series
    return request_info.iloc[0]

@st.cache_data
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

if __name__ == '__main__':
    st.title('Carmax Price Evaluation App')
    # column names required for this func (no null values)
    start_cols = ['price', 'make', 'model', 'year', 'mileage',
                   'transmission', 'fuel_type', 'body_type']
    feat_cols = ['price', 'make', 'model', 'model_age',
                 'mileage_grade', 'transmission_fuel', 'body_type']
    
        
    ## Import backend and competitor data from BQ/Redash
    df, comp_data, backend_data = import_data()
    ## Import bookings data
    df_bookings = import_bookings()
    ## Import tradeins and consignments
    df_tc = import_tradeins()
    
    setup_container = st.empty()
    
    setup_container.info('Feature engineering data')
    ## Validate input data and perform feature engineering
    df_data = feature_engineering(df)
    
    ## Retrieve/Train XGBoost model
    setup_container.info('Preparing train test data')
    train_test_dict = train_test_prep(df_data[feat_cols])
    
    setup_container.info('Obtaining trained XGBoost model')
    
    models, evals = import_model()
    setup_container.empty()
    
    chosen_tab = stx.tab_bar(data = [
        stx.TabBarItemData(id = '1', title = 'Appraisal Requests', description = ''),
        stx.TabBarItemData(id = '2', title = 'CMX Listings Market Survey', description = ''),
        stx.TabBarItemData(id = '3', title = 'Custom Input', description = ''),
        ], default = '2')
    
    placeholder = st.container()
    
    # Appraisal requests
    if chosen_tab == '1':
        with placeholder:
            appraisal_container = st.empty()
            ## Import carmax appraisal requests
            appraisal_container.info('Importing appraisal requests data')
            df1 = import_appraisal_requests(df_data)
            appraisal_container.info('Feature engineering appraisal request data')
            df2 = feature_engineering(df1, df).sort_values('id', ascending = False)
            
            ## prepare test data
            appraisal_container.info('Preparing Appraisal Request test data')
            df_test = test_prep(df2[feat_cols[1:]], 
                                       _enc = train_test_dict['enc'])
            appraisal_container.info('Predicting values for appraisal requests')
            df_pred = models['XGB']['model'].predict(df_test)
            df2.loc[:, 'predicted_value'] = df_pred.round()
            df2 = mv_scores_apply(df, df2)
        
            ## Output predicted appraised value for each entry (with shap breakdown)
            show_cols = ['date', 'id', 'status', 'intention', 'make', 'model', 
                         'year', 'transmission', 'fuel_type', 'mileage', 
                         'mileage_bracket', 'body_type', 'issues', 'with_inspection', 
                         'asking_price', 'predicted_value', 'market_value', 
                         'market_value_min', 'market_value_max', 'market_value_std',
                         'gp_score', 'demand_score', 'overall_score'
                         ]
            
            appraisal_container.empty()
            
            table = df2[show_cols]
            
            st.caption(f'Showing **{len(table)}** items. Select checkbox to show more info.')
            
            df_request = request_select(table)
            
            st.download_button(label = 'Export to CSV',
                                data = convert_df(table),
                                file_name = 'cmx_appraisal_requests.csv',
                                key = 'appraisal_dl')
    
    # CMX Listings Market Survey
    elif chosen_tab == '2':
        with placeholder:
            cmx_container = st.empty()
            ## Import carmax appraisal requests
            cmx_container.info('Importing CMX listings data')
            df1 = import_available_units()
            
            cmx_container.info('Feature engineering CMX listings data')
            df2 = feature_engineering(df1, df).sort_values('id', ascending = False)
            
            ## prepare test data
            cmx_container.info('Preparing CMX listings test data')
            df_test = test_prep(df2[feat_cols[1:]], 
                                       _enc = train_test_dict['enc'])
            
            cmx_container.info('Predicting values for CMX listings')
            df_pred = models['XGB']['model'].predict(df_test)
            df2.loc[:, 'predicted_value'] = df_pred.round()
            
            # market value, gp & demand scores
            df2 = mv_scores_apply(df, df2)
        
            ## Output predicted appraised value for each entry (with shap breakdown)
            show_cols = ['date', 'id', 'status', 'vehicle_id', 'make', 'model', 'year', 
                         'transmission', 'fuel_type', 'mileage', 'body_type', 
                         'plate_no', 'days_on_hand', 'po_value', 'predicted_value', 'market_value', 
                         'market_value_min', 'market_value_max', 'market_value_std', 'sale_price', 
                         'projected_gp', 'gp_score', 'demand_score', 'overall_score', 'saleability', 'url']
    
            cmx_container.empty()
            
            table = df2[show_cols]
            
            st.caption(f'Showing **{len(table)}** items. Select checkbox to show more info.')
            
            df_request = request_select(table)
            
            st.download_button(label = 'Export to CSV',
                                data = convert_df(table),
                                file_name = 'cmx_listings.csv',
                                key = 'listings_dl')
            
        
    elif chosen_tab == '3':
        with placeholder:
            with st.form('Car Feature Form 1'):
                makes_list = config_lica.carmax_makes.name.str.upper().tolist()
                make = st.selectbox('Make',
                                    options = makes_list,
                                    index = makes_list.index('TOYOTA'))
                
                submitted1 = st.form_submit_button('Enter')
                
        #with placeholder2:
            models_list = config_lica.carmax_models.name.str.upper().tolist()
            models_filter = sorted([m.split(make)[-1].strip() for m in models_list if make in m])
            models_filter = [m for m in models_filter if m != '']
            
            with st.form('Car Feature Form 2'):
                
                if len(models_filter):
                    model = st.selectbox('Model',
                                        options = models_filter,
                                        index = models_filter.index('VIOS') if 'VIOS' in models_filter else 0)
                else:
                    st.warning(f'No models associated with {make}.')
                
                ## year
                # year is kept as str datatype, list of year is generated from min year
                yr_list = sorted(list(map(str, range(int(df.year.sort_values().iloc[0]), 
                                       datetime.today().year + 1))), 
                                        reverse = True)
                
                year = st.selectbox('Year',
                                    options = yr_list,
                                    index = 0)
                
                ## transmission
                trans_list = ['AUTOMATIC', 'MANUAL']
                transmission = st.selectbox('Transmission',
                                            options = trans_list,
                                            index = 0)
                transmission = transmission.upper()
                
                ## fuel type
                ft_list = ['GASOLINE', 'DIESEL']
                fuel_type = st.selectbox('Fuel Type',
                                         options = ft_list,
                                         index = 0)
                fuel_type = fuel_type.upper()
                
                df_temp = df[(df.make == make) & (df.model == model) & (df.year == int(year)) & (
                    df.transmission == transmission) & (df.fuel_type == fuel_type)]
                
                submitted2 = st.form_submit_button('Enter')
                
                    
        #with placeholder3:
            with st.form('Car Feature Form 3'):
                ## mileage
                mileage_list = ['0-10,000', '10,001-20,000', '20,001-30,000', '30,001-40,000', 
                                '40,001-50,000', '50,001-60,000', '60,001-70,000', '70,001-80,000', 
                                '80,001-90,000', '90,001-100,000', '100,001+']
                
                #mileage_opts = [m for m in mileage_list if check_mileage(m, df_temp) > 1]
                
                mileage = st.selectbox('Mileage range',
                                       options = mileage_list,
                                       index = 0,
                                       help = 'options change depending on the previous form inputs.')
                
                
                mileage_nums = re.findall('[0-9]+', re.sub(',', '', str(mileage)))
                
                if len(mileage_nums) == 2:
                    min_mileage = int(mileage_nums[0])
                    max_mileage = int(mileage_nums[1])
                else:
                    min_mileage = int(mileage_nums[0])
                    max_mileage = 250000
                
                ## body type
                body_type_list = ['SUV', 'SEDAN', 'VAN', 'PICKUP TRUCK', 
                                  'HATCHBACK', 'CROSSOVER']
                
                # get default/popular body type for given make, model
                body_type_def = get_body_type(df, make, model)
                
                body_type = st.selectbox('Body Type',
                                         options=body_type_list,
                                         index=body_type_list.index(body_type_def) if body_type_def in body_type_list else 0)

                body_type = body_type.upper()
                
                asking_price = st.number_input('Asking Price',
                                               min_value = 0,
                                               max_value = 5000000,
                                               step = 500,
                                               value = 450000,
                                               help = 'price customer aims to sell to Carmax')
                
                specs_dict = {'id' : 1,
                              'make' : make,
                              'model' : model,
                              'year' : int(year),
                              'transmission' : transmission,
                              'fuel_type' : fuel_type,
                              'body_type' : body_type,
                              'mileage' : approx_mileage(mileage, make, transmission, df),
                              'asking_price' : asking_price}
                
                submitted3 = st.form_submit_button('Enter')
                
                if submitted3:
                
                    df1 = pd.DataFrame(list(specs_dict.values()), index = specs_dict.keys()).T
                    
                    df2 = feature_engineering(df1, df)
                    
                    df_test = test_prep(df2[feat_cols[1:]], 
                                               _enc = train_test_dict['enc'])
                    
                    df2.loc[:, 'predicted_value'] = models['XGB']['model'].predict(df_test)
                    
                    df_request = df2.copy()
                
                else:
                    df_request = None
    
    else:
        df_request = None
        placeholder.empty()
        #placeholder2.empty()
        #placeholder3.empty()
    
    ### Give rating for each appraisal request/entry
    if df_request is not None:
        with st.expander('Predicted Value Breakdown', expanded = False):
            # visualize shap
            explainer = shap.Explainer(models['XGB']['model'].best_estimator_)
            #shap_values = explainer(df_test.iloc[[df_request.index[0]]])
            shap_values = explainer(df_test[df_test.index.isin(df2[df2.id == df_request.id.values[0]].index)])
            
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
            row = df2[df2.id == df_request.id.values[0]][feat_cols[1:]]
            new_data = np.array(row)
            shap_values.data = np.array(new_data)
            
            #update feature names
            shap_values.feature_names = list(df2[feat_cols[1:]].columns)
            fig = shap.plots.waterfall(shap_values[0], show = True)
            st.pyplot(fig)
        
        # get original car info
        request_info = df2[df2.id == df_request.id.values[0]]
        
        ## Find similar cars in the market & perform outlier removal
        similar_listings = find_similar_cars(request_info, df)
        
        ## Find similar appraisal requests aside from selected
        similar_appraisals = df2[(df2.make == df_request.make.iloc[0]) &\
                         (df2.model == df_request.model.iloc[0]) &\
                         (df2.year.between(int(df_request.year.iloc[0]) - 1,
                                           int(df_request.year.iloc[0]) + 1))]
        
        similar_tradeins = df_tc[(df_tc.make == df_request.make.iloc[0]) &
                                 (df_tc.model == df_request.model.iloc[0]) &
                                 (df_tc.year.between(int(df_request.year.iloc[0]) - 1,
                                                     int(df_request.year.iloc[0]) + 1))]
        
        with st.expander('**Similar Listings**', expanded = False):
            if len(similar_listings):
                st.dataframe(similar_listings)
            else:
                st.warning('No similar cars in the market.')
        
        
        if chosen_tab == '3':
            # only applies to Manual mode
            with st.expander('**Information**', expanded = True):
                ## market value
                if len(similar_listings):
                    market_value = round(similar_listings['price'].mean(), 2)
                    
                    st.metric('Market Value',
                              value = round(similar_listings['price'].mean(), 2),
                              help = 'average from prices in similar listings tab')
                    
                    mv2, mv3, mv4 = st.columns(3)
                    
                    with mv2:
                        st.metric('Market Value (min)',
                                  value = min(similar_listings['price']),
                                  help = 'minimum from prices in similar listings tab')
                    with mv3:
                        st.metric('Market Value (max)',
                                  value = max(similar_listings['price']),
                                  help = 'maximum from prices in similar listings tab')
                    with mv4:
                        st.metric('Market Value (std)',
                                  value = round(similar_listings['price'].std()),
                                  help = 'standard deviation of prices in similar listings tab')
                
                else:
                    # same as predicted value
                    market_value = round(df_request['predicted_value'].values[0], 2)
                
                ## predicted appraised value
                predicted_value = round(float(df_request['predicted_value'].values[0]), 2)
                if pd.notna(market_value):
                    st.metric('Appraised value',
                              value = predicted_value,
                              delta = round(predicted_value - market_value, 2),
                              help = 'vs market value')
                else:
                    st.metric('Appraised value',
                              value = np.round(predicted_value, 2),
                              help = 'based on ML model predictions')
                
                ask_col, gp_col = st.columns(2)
                
                with ask_col:
                    asking_price = df_request.asking_price.values[0]
                    st.metric('Asking Price',
                              value = asking_price,
                              delta = round(asking_price - predicted_value, 2),
                              delta_color = 'normal',
                              help = 'vs appraised value'
                              )
                
                with gp_col:
                    avg_value = round(0.5*(market_value+predicted_value), 2)
                    est_gp = round(avg_value - asking_price, 2)
                    st.metric('Estimated GP',
                              value = est_gp,
                              delta = f'{round((est_gp/asking_price)*100, 2)}%',
                              help = f'compared to average of market value and appraised value: {avg_value}')
                
                
                ## DEMAND
                # bar plot of car listings
                try:
                    listings_recent = similar_listings[similar_listings.date_listed.apply(lambda x: True if x.date() >= (datetime.today().date() - timedelta(days = 180)) else False)]
                except:
                    listings_recent = pd.DataFrame()
                try:
                    appraisals_recent = similar_appraisals[similar_appraisals.created_at.apply(lambda x: True if x.date() >= (datetime.today().date() - timedelta(days = 180)) else False)]
                except:
                    appraisals_recent = pd.DataFrame()
                try:
                    tradeins_recent = similar_tradeins[similar_tradeins.application_date.apply(lambda x: True if x.date() >= (datetime.today().date() - timedelta(days = 180)) else False)]
                except:
                    tradeins_recent = pd.DataFrame()
                
                if len(listings_recent):
                    avg_time_between_listings = abs(listings_recent.date_listed.diff().mean().days)
                else:
                    avg_time_between_listings = np.NaN
                    
                #st.write('Average days between listings of similar cars: {}'.format(avg_time_between_listings))
                
                
            with st.expander('**Scores**', expanded = True):
                
                gp_score = calc_gp_score(market_value,
                                         predicted_value,
                                         df_request.asking_price.iloc[0])
                if int(gp_score*100) >= 50:
                    st.info(f'**GP Score**: {int(gp_score*100)}')
                else:
                    st.warning(f'**GP Score**: {int(gp_score*100)}')
                
                demand_score = calc_demand_score(request_info,
                                                 df_bookings,
                                                 listings_recent,
                                                 appraisals_recent,
                                                 tradeins_recent)
                if int(demand_score*100) >= 50:
                    st.info(f'**Demand Score**: {int(demand_score*100)}')
                else:
                    st.warning(f'**Demand Score**: {int(demand_score*100)}')
                
                overall_score = gp_score * 0.8 + demand_score * 0.2
                if int(overall_score*100) >= 50:
                    st.info(f'**Overall Score**: {int(overall_score*100)}')
                else:
                    st.warning(f'**Overall Score**: {int(overall_score*100)}')
                
                
            
        

                    
                
