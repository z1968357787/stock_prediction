dataset_select=0
datasets=['000615-data.csv','000628-data.csv','000629-data.csv','000635-data.csv','000659-data.csv','000663-data.csv','000665-data.csv','000666-data.csv','000670-data.csv','000679-data.csv','000680-data.csv']
dataset=datasets[dataset_select].split("-")[0]
# model_select=0
# model_names=['TransAm','MLP','MLP2','MLP3','LSTM','GRU','RNN']
# model_name= model_names[model_select]
predictions={'TransAm':f'output/predict_TransAm_{dataset}.csv',
        'MLP':f'output/predict_MLP_{dataset}.csv',
        'MLP2':f'output/predict_MLP2_{dataset}.csv',
        'MLP3':f'output/predict_MLP3_{dataset}.csv',
        'LSTM':f'output/predict_LSTM_{dataset}.csv',
        'GRU':f'output/predict_GRU_{dataset}.csv',
        'RNN':f'output/predict_RNN_{dataset}.csv'}

evals={'TransAm':f'output/eval_TransAm_{dataset}.csv',
        'MLP':f'output/eval_MLP_{dataset}.csv',
        'MLP2':f'output/eval_MLP2_{dataset}.csv',
        'MLP3':f'output/eval_MLP3_{dataset}.csv',
        'LSTM':f'output/eval_LSTM_{dataset}.csv',
        'GRU':f'output/eval_GRU_{dataset}.csv',
        'RNN':f'output/eval_RNN_{dataset}.csv'}
time_file=f'output/time_{dataset}.csv'