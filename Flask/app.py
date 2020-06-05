from flask import Flask, render_template,request
app = Flask(__name__)
from keras.models import load_model

model = None

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)


def load_model0():
    global model
    model = load_model('chatbot_Bert_9090.h5')



def intro ():
    import videocap
    face = videocap.meth()
    #face = 'sad'
    if face == 'happy':
        return 'Do you want to share with me the reason that makes you %s ?' % (face)
    elif face == 'neutral':
        return "Today is the same like yesterday, I see that on your face. Nothing is happend, isn't it?"
    elif face == 'sad':
        return 'I understand that, you are not in a good mood. What happened?'
    elif face == 'angry':
        return 'I believe that you need to calm down, you look a bit %s. What makes you feel that way?' % (face)
    elif face == 'surprise':
        return 'What is the reason that you look %s.' % (face)
    else:
        return 'No feeling today...'



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/exec')
def parse(name=None):
    a = intro()
    return render_template('chatbot.html',kadir = a)


@app.route("/get")
def get_bot_response():

    import train_chatbot

    userText = request.args.get('msg')
    result = train_chatbot.chat(str(userText),model)

    return result

if __name__ == '__main__':
    load_model0()
    app.run(threaded = False)
    app.debug = False