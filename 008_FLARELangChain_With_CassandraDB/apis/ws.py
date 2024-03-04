from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import subprocess
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
client_connections = {}
app.debug = True 

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('request')
def handle_request(data):   
    print(data)     
    query = data.get('query')
    category = data.get('category')    
    print(['python3', '-u', 'query.py', '--query', query, '--category',category])
    process = subprocess.Popen(['python3', '-u', 'query.py', '--query', query, '--category',category], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    empty_line_count = 0 
    response = ''
    flare_result_flag = False
    for line in iter(process.stdout.readline, b''):
        log_message = line.decode().rstrip()
        
        if flare_result_flag:
            if log_message.strip() == '':
                empty_line_count += 1
                if empty_line_count >= 2:
                    break   
            else: 
                response += log_message + '\n' 
        if log_message.startswith('FLARE RESULT:'):
            flare_result_flag = True
            
        print(log_message)      
        emit('log', {'log': log_message})
    emit('response', {'response': response})

if __name__ == '__main__':
    #app.config['SECRET_KEY'] = 'your_secret_key'
    socketio.run(app)
