from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
import utils
import github_utils
import time
import json
from threading import Thread, Event

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global AI agent instance
cfr_agent = None

# Function to initialize the AI agent with settings
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

# Initialize AI agent with default settings on app start
initialize_ai_agent({})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/training')
def training():
    # Initialize game state if it doesn't exist
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

    # Initialize AI agent if it's not initialized or settings have changed
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
    global cfr_agent
    if cfr_agent is None:
        return jsonify({'error': 'AI agent not initialized'}), 500

    game_state_data = request.get_json()
    print("Received game_state_data:", game_state_data)

    num_cards = len(game_state_data.get('selected_cards', []))
    ai_settings = game_state_data.get('ai_settings', {})

    try:
        selected_cards = [ai_engine.Card(card['rank'], card['suit']) for card in game_state_data['selected_cards']]
        discarded_cards = [ai_engine.Card(card['rank'], card['suit']) for card in game_state_data.get('discarded_cards', [])]
        board = ai_engine.Board()
        for line in ['top', 'middle', 'bottom']:
            for card_data in game_state_data['board'].get(line, []):
                if card_data: # Check if the slot is not empty
                    board.place_card(line, ai_engine.Card(card_data['rank'], card_data['suit']))

        # Check if the board is full before the AI makes a move
        if board.is_full():
            # Calculate royalties and update AI progress
            game_state = ai_engine.GameState(
                selected_cards=selected_cards,
                board=board,
                discarded_cards=discarded_cards,
                ai_settings=ai_settings,
                deck=ai_engine.Card.get_all_cards()
            )
            payoff = game_state.get_payoff()
            print(f"Game over. Payoff: {payoff}")

            # Update AI progress based on the game result
            if cfr_agent:
                try:
                    cfr_agent.update_strategy_for_game_over(game_state, payoff)
                    cfr_agent.save_progress()
                    print("AI progress updated and saved successfully.")
                except Exception as e:
                    print(f"Error updating AI progress: {e}")

            return jsonify({'message': 'Game over', 'payoff': payoff}), 200

    except (KeyError, TypeError) as e:
        return jsonify({'error': f"Invalid game state data format: {e}"}), 400

    game_state = ai_engine.GameState(
        selected_cards=selected_cards,
        board=board,
        discarded_cards=discarded_cards,
        ai_settings=ai_settings,
        deck=ai_engine.Card.get_all_cards()
    )

    timeout_event = Event()
    result = {'move': None}

    ai_thread = Thread(target=cfr_agent.get_move, args=(game_state, num_cards, timeout_event, result))
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

    # Update game state in session (correctly handling occupied slots)
    if move:
        for line in ['top', 'middle', 'bottom']:
            placed_cards = move.get(line, [])
            slot_index = 0  # Start checking from the first slot in each line
            for card in placed_cards:
                serialized_card = serialize_card(card)

                # Find the next available slot in the line
                while slot_index < len(session['game_state']['board'][line]) and session['game_state']['board'][line][slot_index] is not None:
                    slot_index += 1

                # Place the card if an available slot is found
                if slot_index < len(session['game_state']['board'][line]):
                    session['game_state']['board'][line][slot_index] = serialized_card
                    slot_index += 1  # Move to the next slot for the next card

        discarded_card = move.get('discarded')
        if discarded_card:
            session['game_state']['discarded_cards'].append(serialize_card(discarded_card))

        session.modified = True

    # Save AI progress periodically
    if cfr_agent and cfr_agent.iterations % 100 == 0:
        try:
            cfr_agent.save_progress()
            print("AI progress saved successfully.")
        except Exception as e:
            print(f"Error saving AI progress: {e}")

    return jsonify(serialized_move)

if __name__ == '__main__':
    app.run(debug=True)
