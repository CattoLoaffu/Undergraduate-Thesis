from moviepy.editor import *


def generate(path,file) :
    shift = 4
    st = 0
    end = 5
    vc = VideoFileClip(file)
    ac = vc.audio
    ac.write_audiofile(path + "\\chunck.wav")

    d = vc.audio.duration
    print(d)
    i = 0
    while end < d:
        px = ac.subclip(st,end)
        px.write_audiofile(path+"\\"+str(i)+".wav")
        i+=1
        st+=shift
        end+=shift
        if end > d :
            end = d
    px = ac.subclip(st,end)
    px.write_audiofile(path+"\\last.wav")
    os.remove(path + "\\chunck.wav")