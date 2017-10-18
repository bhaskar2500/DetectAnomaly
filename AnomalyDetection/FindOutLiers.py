import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from io import BytesIO
from django.http import HttpResponse
from django.shortcuts import render_to_response,render
from AnomalyDetection.settings import PROJECT_ROOT
import os
from time import sleep
import base64
import json
from sklearn import svm
from django.views.decorators.csrf import csrf_exempt
import csv,codecs

def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)

@csrf_exempt
def importAnamolyData(request):
    label_data=None
    if request.POST and request.FILES:
        tr_data=return_numpy_array_csv("training_file",request)
        label_data=return_numpy_array_csv("labels_file",request)
    else: 
        tr_data = read_dataset(os.path.join(PROJECT_ROOT,'tr_server_data.csv'))
        label_data = read_dataset(os.path.join(PROJECT_ROOT,'gt_server_data.csv'))
        
    normal,abnormal,plt=train_classifier(tr_data,label_data)
    plt=label_anomalies(plt,abnormal)
    figfile=BytesIO()
    plt.savefig(figfile, format='png')
    strHM=base64.b64encode(figfile.getvalue())
    return  render(request, "result.html", {"img": "data:image/png;base64,"+str(strHM,"utf-8"),"anomaly_count":len(abnormal)} )

def return_numpy_array_csv(filename,request):
      training_file = request.FILES[filename]
      reader = training_file.read().decode('utf-8').splitlines()
      tr_data_list=[]
      for each in reader:
          tr_data_list.append([float(every_float_value) for every_float_value in (each.split(","))])
      tr_data=np.array(tr_data_list)
      return tr_data

def train_classifier(tr_data,label_data):
    clf = svm.OneClassSVM(nu=0.0
    5, kernel="rbf", gamma=0.1)
    clf.fit(tr_data,label_data)
    pred = clf.predict(tr_data) 
    normal = tr_data[pred == 1]
    abnormal = tr_data[pred == -1]
    plt.plot(normal[:,0],normal[:,1],'bx')
    plt.plot(abnormal[:,0],abnormal[:,1],'ro')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    return normal,abnormal,plt

def label_anomalies(plt,abnormal):
    labels = ['{0},{1}'.format(str(abnormal[i,0]),str(abnormal[i,1])) for i in range(len(abnormal))]
    for label, x, y in zip(labels, abnormal[:, 0], abnormal[:, 1]):
            plt.annotate(label,xy=(x,y) ,xytext=(100, 10),
            textcoords='offset points', ha='right', va='bottom',
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    return plt