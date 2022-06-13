
import json
from flask import Flask, flash, request, redirect, url_for, render_template
from chatbot import Chatbot



data=dict()
history= list()
print("Starting bot")
bot=Chatbot()
bot.loadQtable('trained.npy')
app = Flask(__name__)

@app.route('/')
def handleRoot():
    return render_template('chat.html')

@app.route('/reward')
def giveReward():
    reward= int(request.args['value'])
    print("recieved "+ str(reward))
    bot.reward(reward)
    return 'ok'


@app.route('/style.css', methods=['GET', 'POST'])
def stylesheet():
    return render_template('style.css')


@app.route('/history', methods=['POST'])
def handleHistory():
    bot.update()
    userInput=request.form['chatMessage']
    print(userInput)
    answer=bot.getAction(userInput)
    history.append({"a_user":userInput, "bot":answer})
    data['history']=history
    return render_template('history.html',data=data)

@app.route('/save')
def saveData():
    bot.saveQtable('trained.npy')
    print("Saved the data")
    return "Saved the data"


if __name__ == '__main__':
    try: 
        app.run()
    except:
        bot.saveQtable('trained.npy')