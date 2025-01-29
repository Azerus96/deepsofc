=== FILE: app.py ===
from flask import Flask, render_template, jsonify, session, request
from threading import Thread, Event
import os
import time
import json
from ai_engine import GameState, Card, AIPlayer
import utils
import github_utils

app = Flask(__name__name)
app.secret_key = os.urandom(24)
ai_player = AIPlayer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/training')
def training():
    init_game_state()
    return render_template('training.html', game_state=session['game_state'])

@app.route('/update_state', methods=['POST'])
def update_state():
    try:
        game_state = request.get_json()
        validate_game_state(game_state)
        
        session['game_state'] = game_state
        session.modified = True
        
        if game_state['ai_settings'] != session.get('prev_ai_settings'):
            update_ai_settings(game_state['ai_settings'])
            
        return jsonify({'status': 'success'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/ai_move', methods=['POST'])
def ai_move():
    try:
        game_data = request.get_json()
        validate_ai_move_request(game_data)
        
        game_state = convert_to_game_state(game_data)
        timeout = int(game_data.get('ai_settings', {}).get('aiTime', 5))
        
        result = {'move': None, 'error': None}
        ai_thread = Thread(target=execute_ai_move, args=(game_state, result, timeout))
        
        ai_thread.start()
        ai_thread.join(timeout=timeout + 1)
        
        if ai_thread.is_alive():
            return jsonify({'error': 'AI move timeout'}), 504
            
        if result['error']:
            return jsonify({'error': result['error']}), 500
            
        update_session_state(result['move'])
        return jsonify(serialize_move(result['move']))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def init_game_state():
    if 'game_state' not in session:
        session['game_state'] = {
            'selected_cards': [],
            'board': {'top': [], 'middle': [], 'bottom': []},
            'discarded_cards': [],
            'ai_settings': {
                'fantasyType': 'normal',
                'fantasyMode': False,
                'aiTime': '5',
                'iterations': '1000',
                'stopThreshold': '0.001'
            }
        }

def validate_game_state(state):
    required_keys = ['selected_cards', 'board', 'discarded_cards', 'ai_settings']
    if not all(key in state for key in required_keys):
        raise ValueError("Invalid game state structure")

def update_ai_settings(settings):
    ai_player.__init__()
    session['prev_ai_settings'] = settings.copy()

def convert_to_game_state(data):
    return GameState(
        selected_cards=[Card(c['rank'], c['suit']) for c in data['selected_cards']],
        board=parse_board(data['board']),
        discarded_cards=[Card(c['rank'], c['suit']) for c in data['discarded_cards']],
        ai_settings=data.get('ai_settings', {})
    )

def parse_board(board_data):
    board = {}
    for line in ['top', 'middle', 'bottom']:
        board[line] = [Card(c['rank'], c['suit']) for c in board_data.get(line, [])]
    return board

def execute_ai_move(game_state, result, timeout):
    try:
        event = Event()
        result['move'] = ai_player.get_action(game_state)
    except Exception as e:
        result['error'] = str(e)

def update_session_state(move):
    if move and move.action_type == 'place':
        session['game_state']['board'][move.line].append({
            'rank': move.card.rank,
            'suit': move.card.suit
        })
    elif move and move.action_type == 'discard':
        session['game_state']['discarded_cards'].append({
            'rank': move.card.rank,
            'suit': move.card.suit
        })
    session.modified = True

def serialize_move(move):
    return {
        'action_type': move.action_type,
        'card': {'rank': move.card.rank, 'suit': move.card.suit},
        'line': move.line,
        'reason': move.reason
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
