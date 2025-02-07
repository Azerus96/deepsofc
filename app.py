from flask import Flask, jsonify, request
import logging

# Настройка логирования на DEBUG для максимальной детализации
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/')
def home():
    app.logger.debug("Entering home route")
    return "Test home route is working!"

@app.route('/training')
def training():
    app.logger.debug("Entering training route")
    return "Test training route is working!"

@app.route('/ai_move', methods=['POST'])
def ai_move():
    app.logger.debug("Entering ai_move route")  # Лог входа в функцию ai_move

    try:
        game_state_data = request.get_json()
        app.logger.debug(f"Received game_state_data: {game_state_data}") # Лог полученных данных

        if not game_state_data:
            app.logger.warning("Warning: No game_state_data received in request") # Предупреждение, если нет данных
        elif not isinstance(game_state_data, dict):
            app.logger.error("Error: game_state_data is not a dictionary") # Ошибка, если данные не словарь
            return jsonify({'error': 'Invalid game state data format'}), 400
        else:
            app.logger.debug("Game state data validation passed") # Лог успешной валидации

        # Здесь можно было бы добавить логику AI (в реальном приложении), но в тесте просто возвращаем сообщение
        response_data = {'message': 'AI move route is working', 'received_data': game_state_data}
        app.logger.debug(f"Sending response: {response_data}") # Лог отправляемого ответа
        return jsonify(response_data)

    except Exception as e:
        app.logger.exception("Exception in ai_move route:") # Лог исключения с трассировкой
        return jsonify({'error': f"Error in ai_move: {e}"}), 500

@app.route('/update_state', methods=['POST'])
def update_state():
    app.logger.debug("Entering update_state route")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=10000)
