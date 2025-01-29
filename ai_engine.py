from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from collections import defaultdict
import itertools
import random
import pickle
import math

@dataclass
class Card:
    rank: str
    suit: str

    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['♥', '♦', '♣', '♠']

    def __post_init__(self):
        if self.rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {self.rank}")
        if self.suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {self.suit}")

    def __hash__(self):
        return hash((self.rank, self.suit))

@dataclass
class GameState:
    board: Dict[str, List[Card]] = field(default_factory=lambda: defaultdict(list))
    discarded: Set[Card] = field(default_factory=set)
    remaining_deck: List[Card] = field(init=False)
    fantasy_level: int = 0
    historical_data: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self._update_remaining()

    def _update_remaining(self):
        used = set(itertools.chain(*self.board.values(), self.discarded))
        all_cards = [Card(r, s) for r in Card.RANKS for s in Card.SUITS]
        self.remaining_deck = [c for c in all_cards if c not in used]

    def get_valid_actions(self) -> List['Action']:
        actions = []
        lines = {'top': 3, 'middle': 5, 'bottom': 5}
        
        for line, limit in lines.items():
            if len(self.board[line]) < limit:
                for card in self.remaining_deck:
                    actions.append(Action('place', card, line))
        
        actions.extend([Action('discard', c) for c in self.remaining_deck])
        return [a for a in actions if self.is_valid_action(a)]

    def is_valid_action(self, action: 'Action') -> bool:
        if action.action_type == "place":
            line_limits = {'top': 3, 'middle': 5, 'bottom': 5}
            return len(self.board[action.line]) < line_limits[action.line]
        return True

    def apply_action(self, action: 'Action') -> 'GameState':
        new_state = GameState(
            board={k: v.copy() for k, v in self.board.items()},
            discarded=self.discarded.copy(),
            fantasy_level=self.fantasy_level,
            historical_data=self.historical_data
        )
        
        if action.action_type == 'place':
            new_state.board[action.line].append(action.card)
        elif action.action_type == 'discard':
            new_state.discarded.add(action.card)
        
        new_state._update_remaining()
        return new_state

    def baseline_evaluation(self) -> float:
        score = 0.0
        line_weights = {'top': 1.5, 'middle': 2.0, 'bottom': 3.0}
        
        for line, weight in line_weights.items():
            score += self._evaluate_line(line) * weight
        
        score -= len(self.discarded) * 0.5
        score += self.fantasy_level * 15
        score -= 1000 if self.is_dead_hand() else 0
        score += self.historical_data.get(self._state_key(), 0)
        
        return score

    def _evaluate_line(self, line: str) -> float:
        cards = self.board[line]
        if not cards:
            return 0
        
        if line == 'top':
            return self._evaluate_top_line(cards)
        elif line == 'middle':
            return self._evaluate_middle_line(cards)
        return self._evaluate_bottom_line(cards)

    def _evaluate_top_line(self, cards: List[Card]) -> float:
        if len(cards) != 3:
            return 0
        if self._is_three_of_a_kind(cards):
            return 10.0
        if self._is_one_pair(cards):
            return self._get_pair_value(cards)
        return self._get_high_card_value(cards)

    def _evaluate_middle_line(self, cards: List[Card]) -> float:
        if len(cards) != 5:
            return len(cards) * 0.5
        
        combinations = [
            (self._is_straight_flush, 30.0),
            (self._is_four_of_a_kind, 20.0),
            (self._is_full_house, 12.0),
            (self._is_flush, 8.0),
            (self._is_straight, 4.0),
            (self._is_three_of_a_kind, 2.0)
        ]
        
        for check_fn, value in combinations:
            if check_fn(cards):
                return value
        return 0.5

    def _evaluate_bottom_line(self, cards: List[Card]) -> float:
        if len(cards) != 5:
            return len(cards) * 0.3
        
        combinations = [
            (self._is_royal_flush, 25.0),
            (self._is_straight_flush, 15.0),
            (self._is_four_of_a_kind, 10.0),
            (self._is_full_house, 6.0),
            (self._is_flush, 4.0),
            (self._is_straight, 2.0)
        ]
        
        for check_fn, value in combinations:
            if check_fn(cards):
                return value
        return 0.3

    def _get_pair_value(self, cards: List[Card]) -> float:
        rank_counts = defaultdict(int)
        for card in cards:
            rank_counts[card.rank] += 1
        
        pair_rank = next((r for r, count in rank_counts.items() if count == 2), None)
        if not pair_rank:
            return 0.0
            
        rank_values = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, 
                      '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13}
        return rank_values.get(pair_rank, 0) * 0.5

    def _get_high_card_value(self, cards: List[Card]) -> float:
        rank_values = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
                      '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13}
        return max(rank_values[card.rank] for card in cards) * 0.1

    def _is_royal_flush(self, cards: List[Card]) -> bool:
        return self._is_straight_flush(cards) and \
               {card.rank for card in cards} == {'10', 'J', 'Q', 'K', 'A'}

    def _is_straight_flush(self, cards: List[Card]) -> bool:
        return self._is_flush(cards) and self._is_straight(cards)

    def _is_four_of_a_kind(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(rank) == 4 for rank in set(ranks))

    def _is_full_house(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return len(set(ranks)) == 2 and \
               any(ranks.count(rank) == 3 for rank in set(ranks))

    def _is_flush(self, cards: List[Card]) -> bool:
        return len({card.suit for card in cards}) == 1

    def _is_straight(self, cards: List[Card]) -> bool:
        rank_values = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
                      '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13}
        values = sorted(rank_values[card.rank] for card in cards)
        
        # Check for wheel straight (A-2-3-4-5)
        if values == [1, 2, 3, 4, 5]:
            return True
            
        return all(values[i] == values[i-1] + 1 for i in range(1, 5))

    def _is_three_of_a_kind(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(rank) == 3 for rank in set(ranks))

    def _is_one_pair(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(rank) == 2 for rank in set(ranks))

    def _line_strength(self, line: str) -> int:
        strength_order = [
            'royal_flush',
            'straight_flush',
            'four_of_a_kind',
            'full_house',
            'flush',
            'straight',
            'three_of_a_kind',
            'two_pair',
            'one_pair',
            'high_card'
        ]
        
        cards = self.board[line]
        if not cards:
            return 0
            
        checks = [
            (self._is_royal_flush, 'royal_flush'),
            (self._is_straight_flush, 'straight_flush'),
            (self._is_four_of_a_kind, 'four_of_a_kind'),
            (self._is_full_house, 'full_house'),
            (self._is_flush, 'flush'),
            (self._is_straight, 'straight'),
            (self._is_three_of_a_kind, 'three_of_a_kind'),
            (self._is_one_pair, 'one_pair')
        ]
        
        for check_fn, strength in checks:
            if check_fn(cards):
                return len(strength_order) - strength_order.index(strength)
        return 1  # high_card

@dataclass
class Action:
    action_type: str
    card: Card
    line: Optional[str] = None
    reason: Optional[str] = None

class AIPlayer:
    def __init__(self, cfr_data_path='cfr_data.pkl'):
        self.historical_data = self._load_cfr_data(cfr_data_path)
        self.learned_strategies = defaultdict(dict)
        self.risk_factor = 1.2
        self.future_discount = 0.7

    def _load_cfr_data(self, path: str) -> Dict:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                return data.get('historical', defaultdict(float))
        except (FileNotFoundError, EOFError):
            return defaultdict(float)

    def get_action(self, state: GameState) -> Action:
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return Action('discard', random.choice(list(state.remaining_deck)), reason='No valid actions')
        
        state_key = state._state_key()
        if state_key in self.learned_strategies:
            return self._best_historical_action(state_key)
        
        return self._calculate_best_action(state, valid_actions)

    def _best_historical_action(self, state_key: str) -> Action:
        strategy = self.learned_strategies[state_key]
        best_action = max(strategy, key=lambda k: strategy[k]['expected_value'])
        return Action(
            action_type=best_action['action_type'],
            card=best_action['card'],
            line=best_action.get('line'),
            reason='Historical strategy'
        )

    def _calculate_best_action(self, state: GameState, valid_actions: List[Action]) -> Action:
        action_scores = []
        for action in valid_actions:
            projected_state = state.apply_action(action)
            score = self._evaluate_state(projected_state)
            action_scores.append((action, score))
        
        best_action = max(action_scores, key=lambda x: x[1])[0]
        best_action.reason = f"Calculated score: {action_scores[0][1]:.2f}"
        return best_action

    def _evaluate_state(self, state: GameState) -> float:
        base_score = state.baseline_evaluation()
        future_score = self._estimate_future_potential(state) * self.future_discount
        risk_score = self._calculate_risk_factor(state) * self.risk_factor
        return base_score + future_score - risk_score

    def _estimate_future_potential(self, state: GameState) -> float:
        if len(state.remaining_deck) < 5:
            return 0.0
            
        needed_combinations = self._identify_needed_combinations(state)
        outs = sum(1 for card in state.remaining_deck if self._is_card_valuable(card, needed_combinations))
        return math.log(outs + 1) * 2.0

    def _identify_needed_combinations(self, state: GameState) -> Dict[str, Set[str]]:
        needed = defaultdict(set)
        for line in ['top', 'middle', 'bottom']:
            current = state.board[line]
            if len(current) < (3 if line == 'top' else 5):
                needed[line].update(self._get_missing_ranks(current))
        return needed

    def _get_missing_ranks(self, cards: List[Card]) -> Set[str]:
        if len(cards) < 1:
            return set(Card.RANKS)
        return {r for r in Card.RANKS if r not in {c.rank for c in cards}}

    def _is_card_valuable(self, card: Card, needed: Dict[str, Set[str]]) -> bool:
        for line in needed.values():
            if card.rank in line:
                return True
        return False

    def _calculate_risk_factor(self, state: GameState) -> float:
        remaining = len(state.remaining_deck)
        if remaining < 10:
            return 3.0 - (remaining * 0.2)
        return 0.5 + (state.fantasy_level * 0.3)

    def save_progress(self):
        data = {
            'historical': self.historical_data,
            'strategies': self.learned_strategies
        }
        with open('cfr_data.pkl', 'wb') as f:
            pickle.dump(data, f)

    def update_historical_data(self, state: GameState, action: Action, reward: float):
        state_key = state._state_key()
        action_key = f"{action.action_type}_{action.card.rank}{action.card.suit}"
        self.historical_data[state_key] = self.historical_data.get(state_key, 0) + reward

# Пример использования
if __name__ == "__main__":
    ai = AIPlayer()
    test_state = GameState()
    action = ai.get_action(test_state)
    print(f"Recommended action: {action}")
