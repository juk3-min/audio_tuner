import math
import time
import turtle
from copy import copy

from PIL import Image, ImageTk
from tkinter import PhotoImage
import audio_tuner
import numpy as np
from scipy.interpolate import interp1d

if __name__ == "__main__":
    LYRICS=("We wish you a merry Christmas\nWe wish you a merry Christmas\nWe wish you a merry Christmas\n"+
            "And a happy new year\n\nGood tidings we bring\nTo you and your kin\nGood tidings for Christmas\n"+
            "And a happy new year")
    SANTA_LEVELS=2
    score=0
    FRAME_WIDTH=1920
    FRAME_HEIGHT=900
    DELTA_PITCH_CALC=5
    MEDIAN_LENGTH=20
    MAGNITUDE_THRESHHOLD=180000
    RATE=4000
    MIN_FREQ=20
    CHUNKTIME=1/MIN_FREQ
    CHUNK_NR_FFT=10
    CUT_OFF_FREQ=25
    wn = turtle.Screen()
    wn.tracer(0)
    bg_c=(0,32,67)
    # blue=np.array((0,32,67))
    blue=np.array((255,0,0))
    green=np.array((19,162,21))
    color_fit = interp1d([0,1], np.vstack([green, blue]), axis=0)
    pitch_calc = 0
    debug = True

concert_pitch=440
ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]


def find_closest_note(pitch):
  """
  This function finds the closest note for a given pitch
  Parameters:
    pitch (float): pitch given in hertz
  Returns:
    closest_note (str): e.g. a, g#, ..
    closest_pitch (float): pitch of the closest note in hertz
  """
  i = int(np.round(np.log2(pitch/concert_pitch)*12))
  k = (np.log2(pitch/concert_pitch)*12 > i)*(i+1)+(np.log2(pitch/concert_pitch)*12 < i)*(i-1)
  other_pitch =    concert_pitch*2**(k/12)
  closest_note = ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
  closest_pitch = concert_pitch*2**(i/12)
  return i, closest_pitch, other_pitch

def auto_pitch(pitch, notes):
    ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    closest_note, closest_pitch, other_pitch = find_closest_note(pitch)
    if closest_pitch> pitch:
        switch=False
    else: switch= True
    for buff in range(0,12):
        i=(closest_note+buff*(1-switch*2))
        if ALL_NOTES[i % 12] in notes:
            return concert_pitch*2**(i/12)
        switch = not switch
    else:
        return closest_pitch



def change_concert_pitch(**kwargs):
    global concert_pitch
    global pitch_calc
    if 'pitch' in kwargs:
        pitch=kwargs.get('pitch')
    else: pitch=pitch_calc
    concert_pitch=pitch

def change_debug():
    global debug
    debug = not debug

def main():
    canvas = wn.getcanvas()  # or, equivalently: turtle.getcanvas()
    root = canvas.winfo_toplevel()
    def on_close():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)


    wn.title("note finder")


    wn.colormode(255)
    wn.bgcolor(bg_c)
    wn.setup(width=FRAME_WIDTH, height=FRAME_HEIGHT)
    wn.tracer(0)
    pen= turtle.Turtle()
    santa=turtle.Turtle()
    pen.width(4)
    santa.up()
    pen.up()
    pen.hideturtle()

    lyrics=turtle.Turtle()
    lyrics.up()
    lyrics.hideturtle()
    lyrics.goto(-FRAME_WIDTH/2+100,-FRAME_HEIGHT/2+400)
    lyrics.write(LYRICS, font=("Roboto",
        35, "bold"))

    wn.listen()
    wn.onkeypress(reset_score, 'space')
    wn.onkeypress(change_concert_pitch, 't')
    wn.onkeypress(change_debug, 'd')
    wn.onkeypress(lambda pitch=440: change_concert_pitch(pitch=pitch), 'r')


    tuner=audio_tuner.Tuner(RATE,MIN_FREQ,"Hear")
    tuner.open_stream()
    data = tuner.record_for_chunk()
    fourier = np.fft.rfft(data)

    dtsz = data.size
    data_buffer=np.zeros(dtsz*CHUNK_NR_FFT)


    timestep = 1 / RATE
    freq = np.fft.rfftfreq(dtsz, d=timestep)


    freq = np.fft.rfftfreq(data_buffer.size, d=timestep)
    fourier_sz=freq.size

    running = True
    counter=0

    HANN_WINDOW = np.hanning(dtsz*CHUNK_NR_FFT)
    factor_list=np.zeros((MEDIAN_LENGTH))
    PITCH_POINTS=40
    pitch_list=np.zeros(PITCH_POINTS)
    pitch_list_log =np.zeros(PITCH_POINTS)
    x_delta = FRAME_WIDTH / PITCH_POINTS
    y_delta = FRAME_HEIGHT / (math.log(2000)-math.log(25))
    x_delta_fourier=FRAME_WIDTH/fourier_sz
    y_delta_fourier = FRAME_HEIGHT
    factor = 0

    image1 = Image.open('santa.gif').convert('RGBA')
    image_names = []
    for i in range(0,SANTA_LEVELS):
        image=copy(image1)
        image_names.append(("santa_"+str(i)))
        image.thumbnail((10 * i + 50, 10 * i + 50))
        photo_image = ImageTk.PhotoImage(image)
        wn.addshape(image_names[i], turtle.Shape("image",photo_image))

    while running:
        pen.clear()
        data=tuner.record_for_chunk()
        data_buffer[(CHUNK_NR_FFT-1)*dtsz:CHUNK_NR_FFT*dtsz]=data
        data_buffer=np.roll(data_buffer, -dtsz)
        fourier = np.fft.rfft(data_buffer* HANN_WINDOW)
        fourier_magnitudes=np.abs(fourier)
        fourier_magnitudes[freq<CUT_OFF_FREQ]=0

        max_idx=np.argmax(fourier_magnitudes)
        strongest_freq=(freq[max_idx])

        over_sound_thresh=np.sum(fourier_magnitudes) > MAGNITUDE_THRESHHOLD


        normalized_fourier=fourier_magnitudes/np.max(fourier_magnitudes)
        delta = min(max_idx, DELTA_PITCH_CALC, CHUNK_NR_FFT * dtsz - max_idx)
        for i in range(0,3):
            lower_octave = strongest_freq/2
            lower_idx = np.where(freq>=int(lower_octave))[0][0]
            delta2 = min(lower_idx, DELTA_PITCH_CALC, CHUNK_NR_FFT * dtsz - lower_idx)
            delta3 = min(delta2,delta)
            if np.sum(fourier_magnitudes[lower_idx-delta3:lower_idx+delta3]) > (1/3 *np.sum(fourier_magnitudes[max_idx-delta3:max_idx+delta3])):
                max_idx=lower_idx
                strongest_freq = strongest_freq/2
            else:
                break


        global pitch_calc
        pitch_calc= np.sum(fourier_magnitudes[max_idx-delta:max_idx+delta]*freq[max_idx-delta:max_idx+delta])/ np.sum(fourier_magnitudes[max_idx-delta:max_idx+delta])

        pitch_list=np.roll(pitch_list,-1)
        pitch_list[-1] = (pitch_calc)
        pitch_list_log=np.roll(pitch_list_log,-1)
        pitch_list_log[-1]=math.log(pitch_calc)

        pitch, closest_freq, other_freq= find_closest_note(pitch_calc)


        pitch_delta=(pitch_calc - closest_freq) / math.fabs(closest_freq - other_freq)



        global score
        if pitch_delta<0.5 and pitch_delta>-0.5:
            score_add = (100 - abs(pitch_delta) * 400) * CHUNKTIME
        else:
            score_add = 0
        if over_sound_thresh:  score+=score_add
        else: score_add=0
        santa.shape(image_names[int(max(0, min(SANTA_LEVELS-1, score/20)))])

            #         score += score_add
        # if abs(pitch_list_log[-1]-pitch_list_log[-2])>
        #
        #     inkey_counter +=1
        #     if inkey_counter >1:
        #         score_add= (100-pitch_delta*400)*CHUNKTIME
        #         score += score_add
        # else:
        #     inkey_counter=0


        draw_and_write(over_sound_thresh,score_add,pitch,pitch_delta,factor,other_freq,closest_freq,pen,PITCH_POINTS, pitch_list_log, y_delta,x_delta, fourier_sz, x_delta_fourier, normalized_fourier, fourier_magnitudes, y_delta_fourier, pitch_calc, strongest_freq)
        wn.update()



    wn.mainloop()
    tuner.close_stream()

def draw_and_write(over_sound_thresh,score_add,pitch,pitch_delta,factor,other_freq,closest_freq,pen,PITCH_POINTS,pitch_list_log, y_delta,x_delta, fourier_sz, x_delta_fourier, normalized_fourier, fourier_magnitudes, y_delta_fourier, pitch_calc, strongest_freq):
    pen.goto(0 - FRAME_WIDTH / 2, (pitch_list_log[0] - math.log(30)) * y_delta - FRAME_HEIGHT / 2)
    pen.width(4)
    pen.down()

    for i in range(0, PITCH_POINTS):
        x =i * x_delta - FRAME_WIDTH / 2
        y =(pitch_list_log[i] - math.log(30)) * y_delta - FRAME_HEIGHT / 2
        pen.goto(x, y)

    pen.up()
    pen.goto(0 - FRAME_WIDTH / 2, 0)
    pen.width(1)
    pen.down()

    for i in range(0, fourier_sz):
        pen.goto(i * x_delta_fourier - FRAME_WIDTH / 2, (normalized_fourier[i] * y_delta_fourier) - FRAME_HEIGHT / 2+10)
    pen.up()

    # pitch_calc=np.average(pitch_list[-5:])

    if over_sound_thresh:
        factor += (2 * abs((pitch_calc - closest_freq) / (closest_freq - other_freq)) - factor) / 3
        wn.bgcolor(tuple(color_fit(min(1, max(factor, 0))).astype(int)))


    if debug:
        pen.goto(-50, 0)
        pen.write((f'Strongest_F:{strongest_freq:.1f},Calc: {pitch_calc:.1f}'), font=("Verdana",
                                                                     15, "normal"))
        pen.goto(-50, 50)
        pen.write((f'Score {score:.0f}, {pitch} '), font=("Verdana",
                                                                15, "normal"))
        pen.goto(-50, 100)
        pen.write((f'Added Score: {score_add:.1f},Closest Pitch {closest_freq:.1f}, {other_freq:.1f}'), font=("Verdana",
        15, "normal"))

        pen.goto(-50, 150)
        pen.write((f'Concert Pitch is: {concert_pitch:.1f}'), font=("Verdana",
        15, "normal"))

        pen.goto(0,0)
        pen.down()
        pen.goto(math.sin(pitch_delta/0.5*math.pi/4)*300,math.cos(pitch_delta/0.5*math.pi/4)*300)
        pen.up()

        if not over_sound_thresh:
            wn.bgcolor('grey')


def word_wrap(text, char_nr):
    char_list= ["," , ".", " ", ":", "?" , "!"]
    output="2"
    i=char_nr
    k=0
    while i< text.__len__():
        while i >= k:
            if text[i] in char_list:
                output +=text[k:i].strip() + "\n"
                k=i
                i += char_nr
                break
            else: i -=1

            if i == k:
                i += char_nr
                output += text[k:i].strip() + "-\n"
                k=i
                i += char_nr
    return output


def reset_score():
    global score
    score=0
    print('XXXXXXXXXXXXXXXXX')

if __name__ == "__main__":
    main()
