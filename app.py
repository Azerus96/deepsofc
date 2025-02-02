from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
from ai_engine import CFRAgent, RandomAgent, Card, Board, GameState
import utils
import github_utils
import time
import json
from threading import Thread, Event

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Глобальные экземпляры агентов
cfr_agent = None
random_agent = RandomAgent()

def initialize_ai_agent(ai_settings):
    global cfr_agent
    iterations = int(ai_settings.get('iterations', 1000))
    stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
    cfr_agent = ai_engine.CFRAgent(iterations=iterations, stop_threshold=stop_threshold)
    if os.environ.get("AI_PROGRESS_TOKEN"):
        try:
            cfr_agent.load_progress()
            print("AI progress loaded successfully.")
        except Exception as e:
            print(f"Error loading AI progress: {e}")
    else:
        print("AI_PROGRESS_TOKEN not set. Progress loading disabled.")

initialize_ai_agent({})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/training')
def training():
    if 'game_state' not in session:
        session['game_state'] = {
            'selected_cards': [],
            'board': {'top': [None] * 3, 'middle': [None] * 5, 'bottom': [None] * 5},
            'discarded_cards': [],
            'ai_settings': {
                'fantasyType': 'normal',
                'fantasyMode': False,
                'aiTime': '5',
                'iterations': '1000',
                'stopThreshold': '0.001',
                'aiType': 'mccfr'
            }
        }
    if cfr_agent is None or session['game_state']['ai_settings'] != session.get('previous_ai_settings'):
        initialize_ai_agent(session['game_state']['ai_settings'])
        session['previous_ai_settings'] = session['game_state']['ai_settings'].copy()
    return render_template('training.html', game_state=session['game_state'])

@app.route('/update_state', methods=['POST'])
def update_state():
    if not request.is_json:
        return jsonify({'error': 'Content type must be application/json'}), 400
    game_state = request.get_json()
    if not isinstance(game_state, dict):
        return jsonify({'error': 'Invalid game state format'}), 400
    session['game_state'] = game_state
    session.modified = True
    if game_state['ai_settings'] != session.get('previous_ai_settings'):
        initialize_ai_agent(game_state['ai_settings'])
        session['previous_ai_settings'] = game_state['ai_settings'].copy()
    return jsonify({'status': 'success'})

@app.route('/ai_move', methods=['POST'])
def ai_move():
    global cfr_agent, random_agent
    game_state_data = request.get_json()
    num_cards = len(game_state_data.get('selected_cards', []))
    ai_settings = game_state_data.get('ai_settings', {})
    ai_type = ai_settings.get('aiType', 'mccfr')
    try:
        selected_cards = [Card(card['rank'], card['suit']) for card in game_state_data['selected_cards']]
        discarded_cards = [Card(card['rank'], card['suit']) for card in game_state_data.get('discarded_cards', [])]
        board = Board()
        for line in ['top', 'middle', 'bottom']:
            for card_data in game_state_data['board'].get(line, []):
                if card_data:
                    board.place_card(line, Card(card_data['rank'], card_data['suit']))
        game_state = GameState(
            selected_cards=selected_cards,
            board=board,
            discarded_cards=discarded_cards,
            ai_settings=ai_settings,
            deck=Card.get_all_cards()
        )
        if game_state.is_terminal():
            payoff = game_state.get_payoff()
            print(f"Game over. Payoff: {payoff}")
            if cfr_agent and ai_settings.get('aiType') == 'mccfr':
                try:
                    cfr_agent.save_progress()
                    print("AI progress saved successfully.")
                except Exception as e:
                    print(f"Error saving AI progress: {e}")
            return jsonify({'message': 'Game over', 'payoff': payoff}), 200
    except (KeyError, TypeError) as e:
        return jsonify({'error': f"Invalid game state data format: {e}"}), 400
    game_state = GameState(
        selected_cards=selected_cards,
        board=board,
        discarded_cards=discarded_cards,
        ai_settings=ai_settings,
        deck=Card.get_all_cards()
    )
    timeout_event = Event()
    result = {'move': None}
    if ai_type == 'mccfr':
        if cfr_agent is None:
            return jsonify({'error': 'MCCFR agent not initialized'}), 500
        ai_thread = Thread(target=cfr_agent.get_move, args=(game_state, num_cards, timeout_event, result))
    else:
        ai_thread = Thread(target=random_agent.get_move, args=(game_state, num_cards, timeout_event, result))
    ai_thread.start()
    ai_thread.join(timeout=int(ai_settings.get('aiTime', 5)))
    if ai_thread.is_alive():
        timeout_event.set()
        ai_thread.join()
        print("AI move timed out")
        return jsonify({'error': 'AI move timed out'}), 504
    move = result['move']
    if 'error' in move:
        return jsonify({'error': move['error']}), 500
    def serialize_card(card):
        return card.to_dict() if card else None
    def serialize_move(move):
        return {key: [serialize_card(card) for card in cards] if isinstance(cards, list) else serialize_card(cards)
                for key, cards in move.items()}
    serialized_move = serialize_move(move)
    if move:
        for line in ['top', 'middle', 'bottom']:
            placed_cards = move.get(line, [])
            slot_index = 0
            for card in placed_cards:
                serialized_card = serialize_card(card)
                while slot_index < len(session['game_state']['board'][line]) and session['game_state']['board'][line][slot_index] is not None:
                    slot_index += 1
                if slot_index < len(session['game_state']['board'][line]):
                    session['game_state']['board'][line][slot_index] = serialized_card
                    slot_index += 1
        discarded_card = move.get('discarded')
        if discarded_card:
            session['game_state']['discarded_cards'].append(serialize_card(discarded_card))
        session.modified = True
    if cfr_agent and cfr_agent.iterations % 100 == 0:
        try:
            cfr_agent.save_progress()
            print("AI progress saved successfully.")
        except Exception as e:
            print(f"Error saving AI progress: {e}")
    return jsonify(serialized_move)

if __name__ == '__main__':
    app.run(debug=True)

from github import Github, GithubException
import base64

GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or "Azerus96"
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY") or "deepsofc"
AI_PROGRESS_FILENAME = "cfr_data.pkl"

def save_progress_to_github(filename=AI_PROGRESS_FILENAME):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        print("AI_PROGRESS_TOKEN not set. Progress saving disabled.")
        return False
    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        try:
            contents = repo.get_contents(filename, ref="main")
            with open(filename, 'rb') as f:
                content = f.read()
            repo.update_file(contents.path, "Update AI progress", base64.b64encode(content).decode('utf-8'), contents.sha, branch="main")
            print(f"AI progress saved to GitHub: {GITHUB_REPOSITORY}/{filename}")
            return True
        except GithubException as e:
            if e.status == 404:
                with open(filename, 'rb') as f:
                    content = f.read()
                repo.create_file(filename, "Initial AI progress", base64.b64encode(content).decode('utf-8'), branch="main")
                print(f"Created new file for AI progress on GitHub: {GITHUB_REPOSITORY}/{filename}")
                return True
            else:
                print(f"Error saving progress to GitHub (other than 404): {e}")
                return False
    except GithubException as e:
        print(f"Error saving progress to GitHub: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
        return False

def load_progress_from_github(filename=AI_PROGRESS_FILENAME):
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        print("AI_PROGRESS_TOKEN not set. Progress loading disabled.")
        return False
    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        contents = repo.get_contents(filename, ref="main")
        file_content = base64.b64decode(contents.content)
        with open(filename, 'wb') as f:
            f.write(file_content)
        print(f"AI progress loaded from GitHub: {GITHUB_REPOSITORY}/{filename}")
        return True
    except GithubException as e:
        if e.status == 404:
            print("Progress file not found in GitHub repository.")
            return False
        else:
            print(f"Error loading progress from GitHub: {e}")
            return False
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return False
