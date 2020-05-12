from openvino.inference_engine import IENetwork, IECore
import numpy as np
import os
import tensorflow as tf
import cv2
import sys
from preproc import preprocess, normalization, adjust_to_see
from tensorflow.keras import backend as K
from generator import Tokenizer
import string
from timeit import default_timer as timer
import ntpath

def predict(img):
    model_xml = "C:\modeloptimizing\HTRModel.xml"
    model_bin = "C:\modeloptimizing\HTRModel.bin"
    ie = IECore()
    net = IENetwork(model=model_xml,weights=model_bin)
    input_blob = next(iter(net.inputs))
    n,c,h,w = net.inputs[input_blob].shape
    exec_net = ie.load_network(network=net, device_name="CPU")
    input_size=(1024,128,1)
    img = preprocess(img, input_size=input_size)
    img = normalization([img])
    img = np.squeeze(img,axis=3)
    img = np.expand_dims(img, axis=0)
    start = timer()
    print("Starting inference...")
    res = exec_net.infer(inputs={input_blob: img})
    end = timer()
    print("End inference time: ", 1000*(end-start))
    output_data = res['dense/BiasAdd/Softmax']
    print(output_data)

    steps_done = 0
    steps=1
    batch_size = int(np.ceil(len(output_data) / steps))
    input_length = len(max(output_data, key=len))
    predicts, probabilities = [], []

    while steps_done < steps:
                index = steps_done * batch_size
                until = index + batch_size

                x_test = np.asarray(output_data[index:until])
                x_test_len = np.asarray([input_length for _ in range(len(x_test))])

                decode, log = K.ctc_decode(x_test,
                                           x_test_len,
                                           greedy=False,
                                           beam_width=10,
                                           top_paths=3)
                probabilities.extend([np.exp(x) for x in log])
                decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
                predicts.extend(np.swapaxes(decode, 0, 1))

                steps_done += 1


    for p in predicts:
        print(str(p))

    for pb in probabilities:
        print(str(pb))


    #interpretation of the data

    max_text_length = 128
    charset_base = string.printable[:95]
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)


    predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

    print("\n####################################")
    for i, (pred, prob) in enumerate(zip(predicts, probabilities)):
        print("\nProb.  - Predict")
        for (pd, pb) in zip(pred, prob):
            print(f"{pb:.4f} - {pd}")
            if i==0:
                pbperc = pb*100
                pdfinal = pd
            i=1+i
    print("\n####################################")
    return pdfinal,pbperc
    
