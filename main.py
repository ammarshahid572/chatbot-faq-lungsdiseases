
import json
from flask import Flask, flash, request, redirect, url_for, render_template




data=dict()
history= list()


app = Flask(__name__)

@app.route('/')
def handleRoot():
    return render_template('chat.html')

@app.route('/history', methods=['POST'])
def handleHistory():
    history.append({"a_user":request.form['chatMessage'], "bot":"response"})
    data['history']=history
    return render_template('history.html',data=data)

if __name__ == '__main__':
   app.run()
