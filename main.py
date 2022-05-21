from flask import *
import json, time, math
import numpy as np
import requests
import pandas as pd
import itertools
import os
from pymongo import MongoClient
from pycaret.classification import *

app = Flask("Smooth Driving ML API")

BROKERDB_CONNECTION_STRING  = 'mongodb://helix:H3l1xNG@15.228.222.191:27000/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false'

def get_database(con_string):    
    client = MongoClient( con_string )
    db = client.sth_helixiot
    return db

def get_collection(db, id):
    collection = db[id]
    return collection


def getStartEnd(VehicleId, TripId):
  collection = get_collection(get_database(BROKERDB_CONNECTION_STRING),VehicleId)
  result = collection.aggregate([
      {
          '$match': {
              'attrName': 'IdViagem', 
              'attrValue': TripId                         
          }
      }, 
      { 
          '$group': {
              '_id': None, 
              'maxTripDate': {
                  '$max': '$recvTime'
              }, 
              'minTripDate': {
                  '$min': '$recvTime'
              }
          }
      }
  ])
  return list(result)[0]

def getData(dic,VehicleId):
  collection = get_collection(get_database(BROKERDB_CONNECTION_STRING),VehicleId)
  filter={
      'recvTime': {
          '$gt': dic["minTripDate"], 
          '$lt': dic["maxTripDate"]
      }
  }
  sort=list({
      'recvTime': 1
  }.items())

  result = collection.find(
    filter=filter,
    sort=sort
  )
  return pd.DataFrame(list(result))

def setTypes(df=None):
    return df.astype({
             "EixoXAcelerometro": float
            ,"EixoYAcelerometro": float
            ,"EixoZAcelerometro": float
            ,"EixoXGiroscopio": float
            ,"EixoYGiroscopio": float
            ,"EixoZGiroscopio": float
            ,"RPMveiculo": float
            ,"Velocidade": float
            }, errors='raise')

def renameFromMongo(df=None):
    df = df.rename(columns={'EixoXAcelerometro': 'x_ace', 'EixoYAcelerometro': 'y_ace', 'EixoZAcelerometro':'z_ace', 'EixoXGiroscopio':'x_giro', 'EixoYGiroscopio':'y_giro', 'EixoZGiroscopio':'z_giro'})
    return df

def getSecondsColumnFromDate(df=None):
  seconds_column = (df['data'] - df['data'][0]).dt.total_seconds()
  return seconds_column


def adjustDataframe(df=None):
  df.drop(['attrType','_id'], axis=1)
  df = df.pivot(index='recvTime', columns='attrName', values='attrValue')
  df['data'] = df.index.values
  df = df.reset_index(drop=True)
  df = setTypes(df)
  df['seconds'] = getSecondsColumnFromDate(df)
  df = renameFromMongo(df)
  return df

def getDataFrameFromMongo(VehicleId, TripId):
  start_end_dic = getStartEnd(VehicleId, TripId)  
  df = getData(start_end_dic,VehicleId)
  adjusted_df = adjustDataframe(df)
  return adjusted_df

def removeUnusedColumns(df=None):
  df = df.drop(['IdViagem','RPMveiculo','Velocidade','data'], axis=1)
  df['y_ace'] = df['y_ace'] -9.6
  return df

def roundSeconds(df=None):
  df['second'] = df.apply(lambda row: math.floor(row.seconds), axis=1)
  return df

def groupSeconds(df=None):
  df = df.groupby(['second'])
  counts_df = df.size().to_frame(name='counts')
  aggregated_df = (counts_df
  .join(df.agg({'x_ace': 'mean'}).rename(columns={'x_ace': 'x_ace_mean'}))
  .join(df.agg({'x_ace': 'median'}).rename(columns={'x_ace': 'x_ace_median'}))
  .join(df.agg({'x_ace': 'min'}).rename(columns={'x_ace': 'x_ace_min'}))
  .join(df.agg({'x_ace': 'max'}).rename(columns={'x_ace': 'x_ace_max'}))
  .join(df.agg({'x_ace': 'std'}).rename(columns={'x_ace': 'x_ace_std'}))
  .join(df.agg({'x_ace': 'var'}).rename(columns={'x_ace': 'x_ace_var'}))
  .join(df.agg({'y_ace': 'mean'}).rename(columns={'y_ace': 'y_ace_mean'}))
  .join(df.agg({'y_ace': 'median'}).rename(columns={'y_ace': 'y_ace_median'}))
  .join(df.agg({'y_ace': 'min'}).rename(columns={'y_ace': 'y_ace_min'}))
  .join(df.agg({'y_ace': 'max'}).rename(columns={'y_ace': 'y_ace_max'}))
  .join(df.agg({'y_ace': 'std'}).rename(columns={'y_ace': 'y_ace_std'}))
  .join(df.agg({'y_ace': 'var'}).rename(columns={'y_ace': 'y_ace_var'}))
  .join(df.agg({'z_ace': 'mean'}).rename(columns={'z_ace': 'z_ace_mean'}))
  .join(df.agg({'z_ace': 'median'}).rename(columns={'z_ace': 'z_ace_median'}))
  .join(df.agg({'z_ace': 'min'}).rename(columns={'z_ace': 'z_ace_min'}))
  .join(df.agg({'z_ace': 'max'}).rename(columns={'z_ace': 'z_ace_max'}))
  .join(df.agg({'z_ace': 'std'}).rename(columns={'z_ace': 'z_ace_std'}))
  .join(df.agg({'z_ace': 'var'}).rename(columns={'z_ace': 'z_ace_var'}))

  .join(df.agg({'x_giro': 'mean'}).rename(columns={'x_giro': 'x_giro_mean'}))
  .join(df.agg({'x_giro': 'median'}).rename(columns={'x_giro': 'x_giro_median'}))
  .join(df.agg({'x_giro': 'min'}).rename(columns={'x_giro': 'x_giro_min'}))
  .join(df.agg({'x_giro': 'max'}).rename(columns={'x_giro': 'x_giro_max'}))
  .join(df.agg({'x_giro': 'std'}).rename(columns={'x_giro': 'x_giro_std'}))
  .join(df.agg({'x_giro': 'var'}).rename(columns={'x_giro': 'x_giro_var'}))
  .join(df.agg({'y_giro': 'mean'}).rename(columns={'y_giro': 'y_giro_mean'}))
  .join(df.agg({'y_giro': 'median'}).rename(columns={'y_giro': 'y_giro_median'}))
  .join(df.agg({'y_giro': 'min'}).rename(columns={'y_giro': 'y_giro_min'}))
  .join(df.agg({'y_giro': 'max'}).rename(columns={'y_giro': 'y_giro_max'}))
  .join(df.agg({'y_giro': 'std'}).rename(columns={'y_giro': 'y_giro_std'}))
  .join(df.agg({'y_giro': 'var'}).rename(columns={'y_giro': 'y_giro_var'}))
  .join(df.agg({'z_giro': 'mean'}).rename(columns={'z_giro': 'z_giro_mean'}))
  .join(df.agg({'z_giro': 'median'}).rename(columns={'z_giro': 'z_giro_median'}))
  .join(df.agg({'z_giro': 'min'}).rename(columns={'z_giro': 'z_giro_min'}))
  .join(df.agg({'z_giro': 'max'}).rename(columns={'z_giro': 'z_giro_max'}))
  .join(df.agg({'z_giro': 'std'}).rename(columns={'z_giro': 'z_giro_std'}))
  .join(df.agg({'z_giro': 'var'}).rename(columns={'z_giro': 'z_giro_var'}))
  .reset_index()
  )
  return aggregated_df

#get seconds with a single measure and duplicate values
def duplicateSingleMeasureSeconds(df=None):
  df['count'] = df.groupby('second')['second'].transform('count')
  single_rows = df[df['count']==1]
  df = pd.concat([df, single_rows])
  df = df.sort_values(by=['seconds'])
  df = df.reset_index(drop=True)
  return df

def sliding_window(iterable, n=3):
    iterables = itertools.tee(iterable, n)
    
    for iterable, num_skipped in zip(iterables, itertools.count()):
        for _ in range(num_skipped):
            next(iterable, None)
    
    return zip(*iterables)

def rows_to_windows(df=None, groupby_name="", num_features=0, window_size=0, features_starting_column=1):
    features_columns = df.columns[features_starting_column:num_features+features_starting_column]
    begins =[]
    ends =[]
    #print(features_columns)
    conv = pd.DataFrame()
    for Name, item in df.groupby(groupby_name):
        window_counter = 0
        for window in sliding_window(item.values.tolist(),window_size):
            x = pd.DataFrame(list(window))            
            begins.append(x.iloc[0,-1])
            ends.append(x.iloc[-1,-1])
            x[0]= window_counter
            window_counter+=1
            conv = pd.concat([conv,x],axis=0)
    conv.reset_index(drop = True, inplace=True)
    return conv, list(features_columns), begins, ends

def flatten_rows(df=None, groupby_name="", features_columns=[], num_features=0, window_size=0, features_starting_column=1):
    df_x = df.iloc[:,:num_features+features_starting_column]
    cc = df_x.groupby([0]).cumcount() + 1
    dfk = df_x.set_index([0, cc]).unstack().sort_index(1, level=1)
    
    # join features names with window number
    columns_with_window_num = []
    for i in range(window_size):
        columns_with_window_num = columns_with_window_num + [s + "_"+str(i) for s in features_columns]
    dfk.columns = columns_with_window_num

    dfk.reset_index(inplace=True)
    return dfk.rename(columns={0:"window_name"})

def set_classes(df=None, gt=None):
    class_array = []
    for index, row in df.iterrows():
        event = gt.loc[(gt["start"] >= row["begins"]) &  (gt["end"] <= row["ends"]), ['evento']] #contém 
        outros = gt.loc[(gt["start"] <= row["begins"]) &  (gt["end"] >= row["ends"]), ['evento']] # está contido
        event.reset_index(drop = True, inplace=True)
        outros.reset_index(drop = True, inplace=True)
        if (len(event)>0):
            class_array.append(event.at[0,'evento'])
        elif (len(outros)>0):
            class_array.append(outros.at[0,'evento'])
        else:
            class_array.append('none')
    return class_array


model = load_model('knn_model')

@app.route("/",methods=["GET"])
def home_page():
    try:
        body = request.get_json()
        EntityId = str(body["entity_id"])
        TripId = str(body["trip_id"])

        startEnd_dic = getStartEnd(EntityId,TripId)

        start = startEnd_dic["minTripDate"]
        end = startEnd_dic["maxTripDate"]

        data = getDataFrameFromMongo(EntityId,TripId)

        velocidadeMax = max(data['Velocidade'])
        velocidadeMedia = data['Velocidade'].mean()
        rpmMax = max(data['RPMveiculo'])
        rpmMedio = data['RPMveiculo'].mean()

        data = removeUnusedColumns(data)
        data = roundSeconds(data)
        data = duplicateSingleMeasureSeconds(data)
        aggregated_data = groupSeconds(data)
        aggregated_data = aggregated_data.dropna() #Remove linhas com valores NAN (dependendo do 'axis')

        del aggregated_data['counts']
        aggregated_data.insert(0, 'Name', 'W')
        aggregated_data['Second'] = aggregated_data['second']
        del aggregated_data['second']

        window_size = 9
        num_features = len(aggregated_data.columns)-2
        features_starting_column = 1
        name_column = "Name"

        data_windows_data , features_columns_data , begins_data , ends_data = rows_to_windows(aggregated_data, name_column, num_features, window_size, features_starting_column)
        data_flatten_data = flatten_rows(data_windows_data, 0, features_columns_data, num_features, window_size, features_starting_column)
        data_flatten_data = data_flatten_data.drop(['window_name'], axis=1)

        data_flatten_data['max_ace'] = data_flatten_data[['x_ace_max_0','y_ace_max_0','z_ace_max_0','x_ace_max_1','y_ace_max_1','z_ace_max_1','x_ace_max_2','y_ace_max_2','z_ace_max_2','x_ace_max_3','y_ace_max_3','z_ace_max_3','x_ace_max_4','y_ace_max_4','z_ace_max_4','x_ace_max_5','y_ace_max_5','z_ace_max_5','x_ace_max_6','y_ace_max_6','z_ace_max_6','x_ace_max_7','y_ace_max_7','z_ace_max_7','x_ace_max_8','y_ace_max_8','z_ace_max_8']].max(axis=1)
        data_flatten_data['min_ace'] = data_flatten_data[['x_ace_min_0','y_ace_min_0','z_ace_min_0','x_ace_min_1','y_ace_min_1','z_ace_min_1','x_ace_min_2','y_ace_min_2','z_ace_min_2','x_ace_min_3','y_ace_min_3','z_ace_min_3','x_ace_min_4','y_ace_min_4','z_ace_min_4','x_ace_min_5','y_ace_min_5','z_ace_min_5','x_ace_min_6','y_ace_min_6','z_ace_min_6','x_ace_min_7','y_ace_min_7','z_ace_min_7','x_ace_min_8','y_ace_min_8','z_ace_min_8']].min(axis=1)
        data_flatten_data['max_giro'] = data_flatten_data[['x_giro_max_0','y_giro_max_0','z_giro_max_0','x_giro_max_1','y_giro_max_1','z_giro_max_1','x_giro_max_2','y_giro_max_2','z_giro_max_2','x_giro_max_3','y_giro_max_3','z_giro_max_3','x_giro_max_4','y_giro_max_4','z_giro_max_4','x_giro_max_5','y_giro_max_5','z_giro_max_5','x_giro_max_6','y_giro_max_6','z_giro_max_6','x_giro_max_7','y_giro_max_7','z_giro_max_7','x_giro_max_8','y_giro_max_8','z_giro_max_8']].max(axis=1)
        data_flatten_data['min_giro'] = data_flatten_data[['x_giro_min_0','y_giro_min_0','z_giro_min_0','x_giro_min_1','y_giro_min_1','z_giro_min_1','x_giro_min_2','y_giro_min_2','z_giro_min_2','x_giro_min_3','y_giro_min_3','z_giro_min_3','x_giro_min_4','y_giro_min_4','z_giro_min_4','x_giro_min_5','y_giro_min_5','z_giro_min_5','x_giro_min_6','y_giro_min_6','z_giro_min_6','x_giro_min_7','y_giro_min_7','z_giro_min_7','x_giro_min_8','y_giro_min_8','z_giro_min_8']].min(axis=1)

        data_flatten_data['begins'] = begins_data
        data_flatten_data['ends'] = ends_data

        unseen_predictions_knn = predict_model(model,data=data_flatten_data)
        filtered = unseen_predictions_knn[(unseen_predictions_knn['Label']!='parado') & (unseen_predictions_knn['Label']!='normal') & (unseen_predictions_knn['Score']>=0.9)].groupby(['max_ace','Label']).size().rename("count").reset_index()

        trocas = len(filtered[filtered['Label']=='troca_agressiva'])
        curvas = len(filtered[filtered['Label']=='curva_agressiva'])

        result = {
            "EntityId": EntityId,
            "TripId": TripId,
            "DateTimeStart": str(start),
            "DateTimeEnd": str(end),
            "EventsCount" : {
                "CurvasAgressivas": curvas,
                "TrocasAgressivas": trocas,
                "RPMmedio": math.floor(rpmMedio),
                "VelocidadeMax": math.floor(velocidadeMax),
                "VelocidadeMedia": math.floor(velocidadeMedia)
            }

        }

        return result
    except:
        return {"erro":"Não foi possível realizar a operação"},500

if __name__ == "__main__":
    app.run()