from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
from ai_engine import CFRAgent, RandomAgent, Card
import utils
import github_utils  # Import the github_utils module
import time
import json
from threading import Thread, Event
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global AI agent instances
cfr_agent = None
random_agent = RandomAgent()

# Function to initialize the AI agent with settings
def initialize_ai_agent(ai_settings):
    global cfr_agent
    app.logger.info(f"Initializing AI agent with settings: {ai_settings}")
    try:
        iterations = int(ai_settings.get('iterations', 1000))
        stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
    except ValueError:
        app.logger.error("Invalid iterations or stopThreshold. Using defaults.")
        iterations = 1000
        stop_threshold = 0.001
    cfr_agent = CFRAgent(iterations=iterations, stop_threshold=stop_threshold)

    if os.environ.get("AI_PROGRESS_TOKEN"):
        try:
            cfr_agent.load_progress()
            app.logger.info("AI progress loaded successfully.")
        except Exception as e:
            app.logger.error(f"Error loading AI progress: {e}")
    else:
        app.logger.info("AI_PROGRESS_TOKEN not set. Progress loading disabled.")

# Initialize AI agent with default settings on app start
initialize_ai_agent({})

# Helper functions for serialization (moved outside of ai_move)
def serialize_card(card):
    return card.to_dict() if card else None

def serialize_move(move, next_slots):
    serialized = {
        key: [serialize_card(card) for card in cards] if isinstance(cards, list) else serialize_card(cards)
        for key, cards in move.items()
    }
    serialized['next_available_slots'] = next_slots
    return serialized

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/training')
def training():
    app.logger.info("Entering /training route")
    # Initialize or load game state
    if 'game_state' not in session:
        app.logger.info("Initializing new game state in session")
        session['game_state'] = {
            'selected_cards': [],
            'board': {
                'top': [None] * 3,    # 3 слота для верхней линии
                'middle': [None] * 5, # 5 слотов для средней линии
                'bottom': [None] * 5  # 5 слотов для нижней линии
            },
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
    else:
        app.logger.info("Loading existing game state from session")
        # Use a single loop for loading card data
        for key in ['selected_cards', 'discarded_cards']:
            if key in session['game_state']:
                session['game_state'][key] = [
                    Card.from_dict(card_dict) for card_dict in session['game_state'][key]
                    if isinstance(card_dict, dict)
                ]
        for line in ['top', 'middle', 'bottom']:
            if line in session['game_state']['board']:
                session['game_state']['board'][line] = [
                    Card.from_dict(card_dict) for card_dict in session['game_state']['board'][line]
                    if isinstance(card_dict, dict)
                ]

    # Initialize AI agent if needed
    if cfr_agent is None or session['game_state']['ai_settings'] != session.get('previous_ai_settings'):
        initialize_ai_agent(session['game_state']['ai_settings'])
        session['previous_ai_settings'] = session['game_state']['ai_settings'].copy()

    app.logger.info(f"Current game state in session AFTER LOADING: {session['game_state']}")
    return render_template('training.html', game_state=session['game_state'])

@app.route('/update_state', methods=['POST'])
def update_state():
    app.logger.info("Entering /update_state route")
    if not request.is_json:
        app.logger.error("Error: Request is not JSON")
        return jsonify({'error': 'Content type must be application/json'}), 400

    try:
        game_state = request.get_json()
        app.logger.info(f"Received game state update: {game_state}")

        if not isinstance(game_state, dict):
            app.logger.error("Error: Invalid game state format (not a dictionary)")
            return jsonify({'error': 'Invalid game state format'}), 400

        # Use session.get with a default empty dictionary
        session['game_state'] = session.get('game_state', {})
        app.logger.info(f"Session BEFORE update: {session['game_state']}")

        # Update board - APPEND, don't replace!
        if 'board' in game_state:
            for line in ['top', 'middle', 'bottom']:
                if line in game_state['board']:
                    # Extend the existing list with the new cards
                    session['game_state']['board'].setdefault(line, [None] * [3, 5, 5][['top', 'middle', 'bottom'].index(line)]).extend(game_state['board'][line]) # Initialize with None if needed and then extend
            app.logger.info(f"Updated board: {session['game_state']['board']}")

        # Update other keys
        if 'selected_cards' in game_state:
            session['game_state']['selected_cards'] = game_state['selected_cards']
        if 'discarded_cards' in game_state:
            session['game_state']['discarded_cards'] = game_state['discarded_cards']
        if 'ai_settings' in game_state:
            session['game_state']['ai_settings'] = game_state['ai_settings']

        # session.modified = True  # REMOVE THIS LINE. Let Flask handle it.

        # Reinitialize AI agent if settings have changed
        if game_state.get('ai_settings') != session.get('previous_ai_settings'):
            app.logger.info("AI settings changed, reinitializing AI agent")
            initialize_ai_agent(game_state['ai_settings'])
            session['previous_ai_settings'] = game_state.get('ai_settings', {}).copy()

        app.logger.info(f"Session AFTER update: {session['game_state']}")
        return jsonify({'status': 'success'})

@app.route('/ai_move', methods=['POST'])
def ai_move():
    global cfr_agent
    global random_agent

    game_state_data = request.get_json()
    app.logger.info(f"Received game state data for AI move: {game_state_data}")

    # Validate game_state_data
    if not isinstance(game_state_data, dict):
        app.logger.error("Error: game_state_data is not a dictionary")
        return jsonify({'error': 'Invalid game state data format'}), 400

    num_cards = len(game_state_data.get('selected_cards', []))
    ai_settings = game_state_data.get('ai_settings', {})
    ai_type = ai_settings.get('aiType', 'mccfr')

    try:
        # --- Data Validation and Card Processing ---
        selected_cards_data = game_state_data.get('selected_cards') or []  # Default to empty list
        if not isinstance(selected_cards_data, list):
            app.logger.error("Error: selected_cards is not a list")
            return jsonify({'error': 'Invalid selected_cards format'}), 400
        selected_cards = [Card.from_dict(card) for card in selected_cards_data]
        app.logger.info(f"Processed selected_cards: {selected_cards}")

        discarded_cards_data = game_state_data.get('discarded_cards') or []  # Default to empty list
        if not isinstance(discarded_cards_data, list):
            app.logger.error("Error: discarded_cards is not a list")
            return jsonify({'error': 'Invalid discarded_cards format'}), 400
        discarded_cards = [Card.from_dict(card) for card in discarded_cards_data]
        app.logger.info(f"Processed discarded_cards: {discarded_cards}")

        board_data = game_state_data.get('board')
        if not isinstance(board_data, dict):
            app.logger.error("Error: board is not a dictionary")
            return jsonify({'error': 'Invalid board format'}), 400

        board = ai_engine.Board()
        for line in ['top', 'middle', 'bottom']:
            line_data = board_data.get(line)
            if not isinstance(line_data, list):
                app.logger.error(f"Error: board[{line}] is not a list")
                return jsonify({'error': f'Invalid board[{line}] format'}), 400
            for card_data in line_data:
                if card_data:
                    board.place_card(line, Card.from_dict(card_data))
        app.logger.info(f"Processed board: {board}")

        # --- Game State Creation ---
        game_state = ai_engine.GameState(
            selected_cards=selected_cards,
            board=board,
            discarded_cards=discarded_cards,
            ai_settings=ai_settings,
            deck=ai_engine.Card.get_all_cards()
        )
        app.logger.info(f"Created game state: {game_state}")

        # --- Check for Terminal State BEFORE AI Call ---
        if game_state.is_terminal():
            app.logger.info("Game is in terminal state")
            payoff = game_state.get_payoff()  # Calculate the final payoff
            royalties = game_state.calculate_royalties()
            total_royalty = sum(royalties.values())
            app.logger.info(f"Game over. Payoff: {payoff}, Royalties: {royalties}, Total: {total_royalty}")

            # Save AI progress (if using MCCFR)
            if cfr_agent and ai_settings.get('aiType') == 'mccfr':
                try:
                    cfr_agent.save_progress()
                    app.logger.info("AI progress saved locally.")
                    if github_utils.save_progress_to_github():
                        app.logger.info("AI progress saved to GitHub.")
                    else:
                        app.logger.warning("Failed to save AI progress to GitHub.")
                except Exception as e:
                    app.logger.error(f"Error saving AI progress: {e}")

            return jsonify({
                'message': 'Game over',
                'payoff': payoff,
                'royalties': royalties,
                'total_royalty': total_royalty
            }), 200

        # --- Find the next available slots BEFORE calling the AI ---
        next_available_slots = {}
        for line in ['top', 'middle', 'bottom']:
            next_available_slots[line] = 0
            while (next_available_slots[line] < len(session['game_state']['board'][line]) and
                   session['game_state']['board'][line][next_available_slots[line]] is not None):
                next_available_slots[line] += 1
        app.logger.info(f"Next available slots BEFORE AI call: {next_available_slots}")

    except (KeyError, TypeError, ValueError) as e:
        app.logger.exception("Exception during game state setup:")  # Log the full traceback
        return jsonify({'error': f"Error during game state setup: {e}"}), 500

    timeout_event = Event()
    result = {'move': None}

    # Choose the appropriate agent based on ai_type
    try:
        if ai_type == 'mccfr':
            if cfr_agent is None:
                app.logger.error("Error: MCCFR agent not initialized")
                return jsonify({'error': 'MCCFR agent not initialized'}), 500
            ai_thread = Thread(target=cfr_agent.get_move, args=(game_state, num_cards, timeout_event, result))
        else:  # ai_type == 'random'
            ai_thread = Thread(target=random_agent.get_move, args=(game_state, num_cards, timeout_event, result))

        ai_thread.start()
        ai_thread.join(timeout=int(ai_settings.get('aiTime', 5)))

        if ai_thread.is_alive():
            timeout_event.set()
            ai_thread.join()
            app.logger.warning("AI move timed out")
            return jsonify({'error': 'AI move timed out'}), 504
    except Exception as e:
        app.logger.exception("Exception during AI agent selection/execution:")
        return jsonify({'error': f"Error during AI agent selection/execution: {e}"}), 500

    try:
        move = result['move']
        if move is None or 'error' in move:
            app.logger.error(f"AI move error: {move.get('error', 'Unknown error')}")
            return jsonify({'error': move.get('error', 'Unknown error')}), 500
    except Exception as e:
        app.logger.exception("Exception while getting move from result:")
        return jsonify({'error': f"Error getting move from result: {e}"}), 500

    # --- Serialization and Response ---
    try:
        serialized_move = serialize_move(move, next_available_slots)
        app.logger.info(f"Serialized move: {serialized_move}")

        royalties = game_state.calculate_royalties()
        total_royalty = sum(royalties.values())

        if move:
            app.logger.info("Updating game state in session with AI move")
            # Update the session with the AI move
            session['game_state']['board'] = move
            app.logger.debug(f"Updated session with new board: {session['game_state']['board']}")

        return jsonify({
            'move': serialized_move,
            'royalties': royalties,
            'total_royalty': total_royalty
        }), 200

    except Exception as e:
        app.logger.exception("Exception during move serialization and response:")
        return jsonify({'error': f"Error during move serialization: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
