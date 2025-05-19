from flask import Flask, render_template, request, session, jsonify


app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    session['username'] = data['username']
    return '', 200

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if data['password'] != data['confirm_password']:
        return jsonify({'error': 'Passwords do not match'}), 400
    session['username'] = data['username']
    return '', 200

@app.route('/logout')
def logout():
    session.pop('username', None)
    return '', 200

@app.route('/get', methods=['POST'])
def get_response():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 403
    user_input = request.json.get('message')
    response = process_message(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
