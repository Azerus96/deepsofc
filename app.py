from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/ai_move', methods=['POST'])
def ai_move():
    return jsonify({'message': 'AI move route is working'})

@app.route('/update_state', methods=['POST'])  # <--- Добавлен route /update_state
def update_state():                               # <--- Функция для /update_state
    return jsonify({'status': 'success'})       # <--- Возвращает JSON с status: success

if __name__ == '__main__':
    app.run(debug=True, port=10000)
