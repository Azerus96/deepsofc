from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
from ai_engine import CFRAgent, RandomAgent, Card
import utils
import github_utils
import time
import json
from threading import Thread, Event
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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

        # Merge the incoming data with the existing session data
        if 'game_state' not in session:
            app.logger.info("Initializing game state in session from request")
            session['game_state'] = game_state
        else:
            app.logger.info("Merging received game state with session data")
            for key, value in game_state.items():
                if key == 'selected_cards':
                    app.logger.info("Updating selected_cards")
                    # Ensure that selected_cards are properly merged
                    if 'selected_cards' not in session['game_state']:
                        session['game_state']['selected_cards'] = []
                    
                    # Convert dictionaries to Card objects
                    session['game_state']['selected_cards'] = [Card.from_dict(card_dict) for card_dict in value if isinstance(card_dict, dict)]
                    app.logger.info(f"Updated selected_cards: {session['game_state']['selected_cards']}")
                elif key == 'board':
                    app.logger.info("Updating board")
                    for line in ['top', 'middle', 'bottom']:
                        if line in value:
                            session['game_state']['board'][line] = [Card.from_dict(card_dict) for card_dict in value[line] if isinstance(card_dict, dict)]
                    app.logger.info(f"Updated board: {session['game_state']['board']}")
                elif key in session['game_state'] and isinstance(session['game_state'][key], list):
                    app.logger.info(f"Extending list for key: {key}")
                    session['game_state'][key].extend(value)
                elif key in session['game_state'] and isinstance(session['game_state'][key], dict):
                    app.logger.info(f"Updating dictionary for key: {key}")
                    session['game_state'][key].update(value)
                else:
                    app.logger.info(f"Setting new value for key: {key}")
                    session['game_state'][key] = value

        session.modified = True

        # Reinitialize AI agent if settings have changed
        if game_state['ai_settings'] != session.get('previous_ai_settings'):
            app.logger.info("AI settings changed, reinitializing AI agent")
            initialize_ai_agent(game_state['ai_settings'])
            session['previous_ai_settings'] = game_state['ai_settings'].copy()

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
            deck=ai_engine.Card.get_all_cards()  # Corrected call to get_all_cards()
        )
        app.logger.info(f"Created game state: {game_state}")

        # Check if the board is full before the AI makes a move
        if game_state.is_terminal():
            app.logger.info("Game is in terminal state")
            # Calculate royalties and update AI progress
            payoff = game_state.get_payoff()
            app.logger.info(f"Game over. Payoff: {payoff}")

            # Update AI progress based on the game result (if using MCCFR)
            if cfr_agent and ai_settings.get('aiType') == 'mccfr':
                try:
                    # No need to update strategy here, just save the progress
                    cfr_agent.save_progress()
                    app.logger.info("AI progress saved successfully.")
                except Exception as e:
                    app.logger.error(f"Error saving AI progress: {e}")

            return jsonify({'message': 'Game over', 'payoff': payoff}), 200

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
    if 'error' in move:
        app.logger.error(f"AI move error: {move['error']}")
        return jsonify({'error': move['error']}), 500

    # Serialize the move using Card.to_dict()
    def serialize_card(card):
        return card.to_dict() if card else None

    def serialize_move(move):
        return {key: [serialize_card(card) for card in cards] if isinstance(cards, list) else serialize_card(cards)
                for key, cards in move.items()}

    serialized_move = serialize_move(move)
    app.logger.info(f"Serialized move: {serialized_move}")

    # Calculate royalties
    royalties = game_state.calculate_royalties()
    total_royalty = sum(royalties.values())

    # Update game state in session (correctly handling occupied slots)
    if move:
        app.logger.info("Updating game state in session with AI move")
        for line in ['top', 'middle', 'bottom']:
            placed_cards = move.get(line, [])
            slot_index = 0  # Start checking from the first slot in each line
            for card in placed_cards:
                serialized_card = serialize_card(card)
                app.logger.info(f"Placing card: {serialized_card} on line: {line} at index: {slot_index}")

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
            app.logger.info("AI progress saved successfully.")
        except Exception as e:
            app.logger.error(f"Error saving AI progress: {e}")

    app.logger.info(f"Returning AI move: {serialized_move}, Royalties: {royalties}, Total Royalty: {total_royalty}")
    return jsonify({
        'move': serialized_move,
        'royalties': royalties,
        'total_royalty': total_royalty
    })

if __name__ == '__main__':
    app.run(debug=True)
