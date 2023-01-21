import sys
from sklearn.linear_model import LinearRegression





def make_models(minmax_y=False,
                preserve_order=False):
    '''
    Uses local vars to produce an output dataframe.
    References pre-defined train/test split.
    '''
    
    _records_dict, _records = dict(), []

    shuffle=not(preserve_order)
    
    if not minmax_y:
        X_train, X_test, y_train, y_test = get_train_test_split(minmax_y=minmax_y, shuffle=shuffle)
    else:
        X_train, X_test, y_train, y_test, y_scaler = get_train_test_split(minmax_y=minmax_y, shuffle=shuffle)

    outputs,model_outputs = [],dict()
    ar = np.array
    models = {
        'ridge': RidgeCV, 
        #'svm lin': LinearSVR,
        #'svm': SVR,
       # 'linear_reg': LinearRegression,
        'elasticnet': ElasticNetCV
    }
    
    dep_vars = y_test.columns

    for dep_var in ['casual', 'registered']:
        for modelname, m in models.items():
            label = modelname+','+dep_var
            model = m()
         
            if modelname == 'lasso':
                model = LassoCV()
            elif modelname == 'elasticnet':
                model = ElasticNetCV()

            y_true = y_test[dep_var]
            y_tr = y_train[dep_var]

            try:
         
                print(f'fitting {label}')
                model.fit(X_train, y_tr)
                y_pred = model.predict(X_test)

                ys = y_true, y_pred
                
                e1,e2,e3,e4 = mean_absolute_error(*ys),mean_absolute_percentage_error(*ys),mean_squared_error(*ys),np.nan#ar([_**0.5 for _ in mean_squared_error(*ys)])

                #e1,e2,e3,e4 = mean_absolute_error(*ys),mean_absolute_percentage_error(*ys),mean_squared_error(*ys),mean_squared_log_error(*ys)#ar([_**0.5 for _ in mean_squared_error(*ys)])

                # label = modelname+','+dep_var

                try:
                    coefs = model.coef_
                except:
                    coefs = np.nan

                if modelname == 'svm':
                    coefs = np.nan


                this_output = {'label': label,
                               'modelname': modelname,
                               'model': model,
                               'dep_var': dep_var,
                               'strmodel': str(model),
                               'coefs': coefs,
                               'mean_absolute_error': float(e1),
                               'mean_absolute_percentage_error': float(e2),
                               'mean_squared_error': float(e3),
                               'mean_squared_log_error': float(e4),
                               'y_true': y_true,
                               'y_pred': y_pred,
                               'error': ''}

                # raw_model_outputs[modelname+','+dep_var] = (results)
                _records_dict[label]=this_output
                _records.append(this_output)

                # print(this_output)
    #             print(*results)
            except:
                print(sys.exc_info())
                keys_ = ['modelname',
                         'model',
                         'dep_var',
                         'strmodel',
                         'coefs',
                         'mean_absolute_error',
                         'mean_absolute_percentage_error',
                         'mean_squared_error',
                         'y_true',
                         'y_pred',
                         'error']
                this_output = { _:np.nan for _ in keys_ }
                this_output['error'] = str(sys.exc_info())
                _records_dict[label] = this_output
                _records.append(this_output)
                
    dm = pd.DataFrame.from_records(_records)
    dm['y_true']=dm.y_true.apply(lambda x: x.values)
    dm['ytr']=dm.y_true.apply(lambda x: ','.join(list(map(str, x))))
    dm['y_pred']=dm.y_pred.apply(lambda x: np.array(list(map(lambda y: round(y,1), x))))
    dm['ypr']=dm.y_pred.apply(lambda x: ','.join(list(map(str, x))))

    return dm, _records_dict, _records