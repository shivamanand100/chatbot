from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from chatbot_py import process_message

app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/chat')
def chat():
    if not session.get('username'):
        flash('Please log in or register first.', 'error')
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    session['username'] = username
    flash('Login successful!', 'success')
    return redirect(url_for('chat'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        session['username'] = username
        flash('Registration successful!', 'success')
        return redirect(url_for('chat'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/get', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    response = process_message(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
