import numpy as np
import scipy.signal
import timeit
from python_speech_features import mfcc
import pygame
import time
from scipy.io import wavfile
import subprocess
import panda as import pdb; pdb.set_trace()
from tflite_runtime.interpreter import interpreter

pygame.init()

display_height = 600
display_width = 600

black = (0,0,0)
alpha = (0,88,255)
white = (255,255,255)
red = (200,0,0)
green = (0,200,0)
bright_red = (255,0,0)
bright_green = (0,255,0)

gameDisplay = pygame.display.set_mode((display_width, display_height)
pygame.display.set_caption("Musical instrument recognition system")

gameDisplay.fill(black)

guitarImg = pygame.image.load('./instruments/acoustic_guitar.jpg')
bassImg = pygame.image.load('./instruments/bass_drum.jpg')
celloImg = pygame.image.load('./instruments/cello.jpg')
clarinetImg = pygame.image.load('./instruments/clarinet.jpg')
doubleImg = pygame.image.load('./instruments/double_bass.jpg')
fluteImg = pygame.image.load('./instruments/flute.jpg')
saxImg = pygame.image.load('./instruments/saxophone.jpg')
snareImg = pygame.image.load('./instruments/snare_drum.jpg')
violinImg = pygame.image.load('./instruments/violin.jpg')
hihatImg = pygame.image.load('./instruments/hihat.jpg')

# List of instruments trained
'''
Acoustic Guitar
Bass Drum
Cello
Clarinet
Double Bass
Flute
Hi-Hat
Saxophone
Snare Drum
Violin
'''

# Parameters
sample_rate = 44100
step = 1600
nfeat = 13
nfilt = 26
nfft = 512
minScaler = -71.02
maxScaler = 69.70
model_path = 'sound_model.tflite'

# Load model (interpreter)
interpreter = interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf',30)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()

def close():
    pygame.quit()
    quit()

def text_objects(text, font):
    textSurface = font.render(text, True, alpha)
    return textSurface, textSurface.get_rect()

def button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(gameDisplay, ac, (x,y,w,h))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(gameDisplay, ic, (x,y,w,h))
    smallText = pygame.font.SysFont('comicsansms', 20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ((x+(w/2)),(y+(h/2)))
    gameDisplay.blit(textSurf, textRect)

def showInstruments(value):
    instruments_images = [guitarImg, bassImg, celloImg, clarinetImg, doubleImg, fluteImg, hihatImg, saxImg, snareImg, violinImg]
    gameDisplay.blit(instruments_images[value],(0,0))

def s2t():
    intruments = ['0 - Acoustic_guitar','1 - Bass_drum','2 - Cello', '3 - Clarinet','4 - Double_bass','5 - Flute','6 - Hi=hat','7 - Saxophone','8 - Snare_drum','9 - Violin']
    rate, wav = wavfile.read('./wavfiles/teste.wav');
    y_pred = build_predictions(rate, wav)
    value = max(set(y_pred), key = y_pred.count)
    print('Prediction: ', instruments[value])
    print('y_pred: ',y_pred)
    showInstruments(value)

def s4t():
    intruments = ['0 - Acoustic_guitar','1 - Bass_drum','2 - Cello', '3 - Clarinet','4 - Double_bass','5 - Flute','6 - Hi=hat','7 - Saxophone','8 - Snare_drum','9 - Violin']
    record = 'arecord -d 2 --rate 16000 soundrecognition.wav'
    p = subprocess.Popen(record, shell=True)
    time.sleep(4)
    rate, wav = wavfile.read('./soundrecognition.wav')
    y_pred = build_predictions(rate, wav)
    value = max(set(y_pred), key = y_pred.count)
    print('Prediction: ', instruments[value])
    print('y_pred: ',y_pred)
    showInstruments(value)

def build_predictions(rate, wav):
    mask = envelope(wav, rate, 0.0005)
    wav = wav[mask]
    y_pred = []
    print('Extracting MFCC features...')
    y_prob = []
    print('Rate: ', rate)

    for i in range(0, wav.shape[0] - int(rate/10), int(rate/10)):
        sample = wav[i:i+int(rate/10)]
        x = mfcc(sample, rate, numcep = nfeat, nfilt = nfilt, nfft = nfft)
        x = (x - minScaler) / (maxScaler - minScaler)
        x = x.reshape(1, x.shape[0], x.shape[1], 1)
        input_shape = input_details[0]['index']
        in_tensor = np.float32(x)
        interpreter.set_tensor(input_shape, in_tensor)
        interpreter.invoke()
        y_hat = interpreter.get_tensor(output_details[0]['index'])
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))

    return y_pred

def envelope(y, rate, threshold):
    print('Enveloping the audio signal...')
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window = int(rate/10), min_periods  = 1, center = True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    
    return mask

def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT;
            pygame.quit()
            quit()
        button("Abrir", 150,450,100,50,green,bright_green,s2t)    
        button("Gravar", 550,450,100,50,green,bright_red,s4t)
        pygame.display.update()

if __name__ == '__main__':
    main()    

