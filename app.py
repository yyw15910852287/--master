# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'src')
# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pickle
import random

import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing import sequence
#从config文件中引入一些参数 包括token最大长度 测试文件夹长度 最优的模型参数
from config import max_token_length, test_a_image_folder, best_model
from forward import build_model
from generated import test_gen

#使用训练好的模型对图片进行测试
def beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=3):
    start = [word2idx["<start>"]]
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_token_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_token_length, padding='post')
            e = encoding_test[image_name]
            preds = model.predict([np.array([e]), np.array(par_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ''.join(final_caption[1:])
    return final_caption


if __name__ == '__main__':
    channel = 3
    model_weights_path = os.path.join('models', best_model)
    print('Model loading...')
    model = build_model()
    model.load_weights(model_weights_path)
    print('The model is loaded...')

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))
    print('The corpus has been loaded...')

    test_gen()

    encoding_test = pickle.load(open('data/encoded_test_a_images.p', 'rb'))
    names = [f for f in encoding_test.keys()]
    samples = names
    sentences = []

    for i in range(len(samples)):
        image_name = samples[i]

        image_input = np.zeros((1, 2048))
        image_input[0] = encoding_test[image_name]
        filename = os.path.join(test_a_image_folder, image_name)
        # print('Start processing image: {}'.format(filename))
        print('The picture described is:',image_name)

        candidate1=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=1)
        print('Beam Search, k=1:',candidate1)
        sentences.append(candidate1)

        candidate2=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=2)
        print('Beam Search, k=2:',candidate2)
        sentences.append(candidate2)

        candidate3=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=3)
        print('Beam Search, k=3:',candidate3)
        sentences.append(candidate3)

        candidate4=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=5)
        print('Beam Search, k=5:',candidate4)
        sentences.append(candidate4)

        candidate5=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=7)
        print('Beam Search, k=7:',candidate5)
        sentences.append(candidate5)

        candidate6=beam_search_predictions(model, image_name, word2idx, idx2word, encoding_test, beam_index=9)
        print('Beam Search, k=9:',candidate6)
        sentences.append(candidate6)

        img = cv.imread(filename)
        #img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
        if not os.path.exists('images'):
            os.makedirs('images')
        cv.imwrite('images/{}_bs_image.jpg'.format(i), img)

    with open('demo.txt', 'w') as file:
        file.write('\n'.join(sentences))
