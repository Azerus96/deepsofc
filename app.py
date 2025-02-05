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
    iterations = int(ai_settings.get('iterations', 1000))
    stop_threshold = float(ai_settings.get('stopThreshold', 0.001))
    cfr_agent = ai_engine.CFRAgent(iterations=iterations, stop_threshold=stop_threshold)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/training')
def training():
    app.logger.info("Entering /training route")
    # Initialize game state if it doesn't exist or reset if needed
    if 'game_state' not in session:
        app.logger.info("Initializing new game state in session")
        session['game_state'] = {
            'selected_cards': [],
            'board': {'top': [], 'middle': [], 'bottom': []},
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
    else: # IMPORTANT: Load Card objects from dictionaries
        app.logger.info("Loading existing game state from session")
        if 'selected_cards' in session['game_state']:
            session['game_state']['selected_cards'] = [
                Card.from_dict(card_dict) for card_dict in session['game_state']['selected_cards']
                if isinstance(card_dict, dict)
            ]
        if 'board' in session['game_state']:
            for line in ['top', 'middle', 'bottom']:
                if line in session['game_state']['board']:
                    session['game_state']['board'][line] = [
                        Card.from_dict(card_dict) for card_dict in session['game_state']['board'][line]
                        if isinstance(card_dict, dict)
                    ]
        if 'discarded_cards' in session['game_state']:
            session['game_state']['discarded_cards'] = [
                Card.from_dict(card_dict) for card_dict in session['game_state']['discarded_cards']
                if isinstance(card_dict, dict)
            ]


    # Initialize AI agent if it's not initialized or settings have changed
    if cfr_agent is None or session['game_state']['ai_settings'] != session.get('previous_ai_settings'):
        initialize_ai_agent(session['game_state']['ai_settings'])
        session['previous_ai_settings'] = session['game_state']['ai_settings'].copy()

    app.logger.info(f"Current game state in session: {session['game_state']}")
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

        # Initialize or update the game state in the session
        if 'game_state' not in session:
            session['game_state'] = {}

        # Update selected_cards (replace, don't append, and use to_dict)
        if 'selected_cards' in game_state:
            session['game_state']['selected_cards'] = [
                card_dict for card_dict in game_state['selected_cards']  # Keep as dictionaries
            ]
            app.logger.info(f"Updated selected_cards: {session['game_state']['selected_cards']}")

        # Update board (correctly handle Card objects, and use to_dict)
        if 'board' in game_state:
            for line in ['top', 'middle', 'bottom']:
                if line in game_state['board']:
                    session['game_state']['board'][line] = [
                        card_dict for card_dict in game_state['board'][line]  # Keep as dictionaries
                    ]
            app.logger.info(f"Updated board: {session['game_state']['board']}")

        # Update other keys (discarded_cards, ai_settings, etc.)
        #   Make sure discarded_cards are also stored as dictionaries
        if 'discarded_cards' in game_state:
            session['game_state']['discarded_cards'] = [
                card_dict for card_dict in game_state['discarded_cards']
            ]
        for key in ['ai_settings']:  # Only ai_settings is not a list of cards
            if key in game_state:
                session['game_state'][key] = game_state[key]

        session.modified = True

        # Reinitialize AI agent if settings have changed
        if game_state.get('ai_settings') != session.get('previous_ai_settings'):
            app.logger.info("AI settings changed, reinitializing AI agent")
            initialize_ai_agent(game_state['ai_settings'])
            session['previous_ai_settings'] = game_state.get('ai_settings', {}).copy()

        app.logger.info(f"Updated game state in session: {session['game_state']}")
        return jsonify({'status': 'success'})

    except Exception as e:
        app.logger.error(f"Error in update_state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ai_move', methods=['POST'])
def ai_move():
    global cfr_agent
    global random_agent

    game_state_data = request.get_json()
    app.logger.info(f"Received game state data for AI move: {game_state_data}")

    num_cards = len(game_state_data.get('selected_cards', []))
    ai_settings = game_state_data.get('ai_settings', {})
    ai_type = ai_settings.get('aiType', 'mccfr')

    try:
        # Ensure selected_cards is a list of Card objects
        selected_cards_data = game_state_data.get('selected_cards')
        if selected_cards_data is None:
            selected_cards = []
            app.logger.info("selected_cards is None, initializing to empty list")
        else:
            selected_cards = [Card.from_dict(card) for card in selected_cards_data]
        app.logger.info(f"Processed selected_cards: {selected_cards}")

        discarded_cards = [Card.from_dict(card) for card in game_state_data.get('discarded_cards', [])]
        app.logger.info(f"Processed discarded_cards: {discarded_cards}")
        board = ai_engine.Board()
        for line in ['top', 'middle', 'bottom']:
            for card_data in game_state_data['board'].get(line, []):
                if card_data:
                    board.place_card(line, Card.from_dict(card_data))
        app.logger.info(f"Processed board: {board}")

        game_state = ai_engine.GameState(
            selected_cards=selected_cards,
            board=board,
            discarded_cards=discarded_cards,
            ai_settings=ai_settings,
            deck=ai_engine.Card.get_all_cards()
        )
        app.logger.info(f"Created game state: {game_state}")


        # --- END OF GAME HANDLING ---
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

                    # Try to save to GitHub
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
        # --- END OF END OF GAME HANDLING ---


    except (KeyError, TypeError, ValueError) as e:
        app.logger.error(f"Error in ai_move during game state creation: {e}")
        return jsonify({'error': f"Invalid game state data format: {e}"}), 400

    timeout_event = Event()
    result = {'move': None}

    # Choose the appropriate agent based on ai_type
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

    move = result['move']
    if move is None or 'error' in move:
        app.logger.error(f"AI move error: {move.get('error', 'Unknown error')}")
        return jsonify({'error': move.get('error', 'Unknown error')}), 500


    # Serialize the move using Card.to_dict()
    def serialize_card(card):
        return card.to_dict() if card else None

    def serialize_move(move):
        return {key: [serialize_card(card) for card in cards] if isinstance(cards, list) else serialize_card(cards)
                for key, cards in move.items()}

    serialized_move = serialize_move(move)
    app.logger.info(f"Serialized move: {serialized_move}")

    # Calculate royalties (even if not terminal, for display)
    royalties = game_state.calculate_royalties()
    total_royalty = sum(royalties.values())


    # Update game state in session (correctly handling occupied slots)
    if move:
        app.logger.info("Updating game state in session with AI move")
        for line in ['top', 'middle', 'bottom']:
            if line in move:
                placed_cards = move.get(line, [])
                slot_index = 0
                for card in placed_cards:
                    serialized_card = serialize_card(card)
                    # Find the next available slot
                    while slot_index < len(session['game_state']['board'][line]) and session['game_state']['board'][line][slot_index] is not None:
                        slot_index += 1
                    if slot_index < len(session['game_state']['board'][line]):
                        session['game_state']['board'][line][slot_index] = serialized_card
                    else:
                        app.logger.warning(f"No slot for {serialized_card} on {line}")


        discarded_card = move.get('discarded')
        if discarded_card:
            session['game_state']['discarded_cards'].append(serialize_card(discarded_card))

        session.modified = True

    # NO PERIODIC SAVING HERE - only save at the end of the game

    app.logger.info(f"Returning AI move: {serialized_move}, Royalties: {royalties}, Total Royalty: {total_royalty}")
    return jsonify({
        'move': serialized_move,
        'royalties': royalties,
        'total_royalty': total_royalty
    })

if __name__ == '__main__':
    app.run(debug=True)
