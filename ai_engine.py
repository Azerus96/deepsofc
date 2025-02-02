import random
import itertools
from collections import defaultdict
import utils
from threading import Event, Thread
import time
import math

class Card:
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['♥', '♦', '♣', '♠']

    def __init__(self, rank, suit):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Rank must be one of: {self.RANKS}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Suit must be one of: {self.SUITS}")
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        if isinstance(other, dict):
            return self.rank == other.get('rank') and self.suit == other.get('suit')
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def to_dict(self):
        return {'rank': self.rank, 'suit': self.suit}

    @staticmethod
    def from_dict(card_dict):
        return Card(card_dict['rank'], card_dict['suit'])

    @staticmethod
    def get_all_cards():
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

class Hand:
    def __init__(self, cards=None):
        self.cards = cards if cards is not None else []

    def add_card(self, card):
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.append(card)

    def remove_card(self, card):
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        try:
            self.cards.remove(card)
        except ValueError:
            print(f"Card {card} not found in hand: {self.cards}")

    def __repr__(self):
        return ', '.join(map(str, self.cards))

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

class Board:
    def __init__(self):
        self.top = []
        self.middle = []
        self.bottom = []

    def place_card(self, line, card):
        if line == 'top':
            if len(self.top) >= 3:
                raise ValueError("Top line is full")
            self.top.append(card)
        elif line == 'middle':
            if len(self.middle) >= 5:
                raise ValueError("Middle line is full")
            self.middle.append(card)
        elif line == 'bottom':
            if len(self.bottom) >= 5:
                raise ValueError("Bottom line is full")
            self.bottom.append(card)
        else:
            raise ValueError(f"Invalid line: {line}. Must be one of: 'top', 'middle', 'bottom'")

    def is_full(self):
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self):
        self.top = []
        self.middle = []
        self.bottom = []

    def __repr__(self):
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

    def get_cards(self, line):
        if line == 'top':
            return self.top
        elif line == 'middle':
            return self.middle
        elif line == 'bottom':
            return self.bottom
        else:
            raise ValueError("Invalid line specified")

class GameState:
    def __init__(self, selected_cards=None, board=None, discarded_cards=None, ai_settings=None, deck=None):
        self.selected_cards = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board = board if board is not None else Board()
        self.discarded_cards = discarded_cards if discarded_cards is not None else []
        self.ai_settings = ai_settings if ai_settings is not None else {}
        self.current_player = 0
        self.deck = deck if deck is not None else self.create_deck()
        self.rank_map = {rank: i for i, rank in enumerate(Card.RANKS)}
        self.suit_map = {suit: i for i, suit in enumerate(Card.SUITS)}

    def create_deck(self):
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

    def get_current_player(self):
        return self.current_player

    def is_terminal(self):
        return self.board.is_full()

    def get_num_cards_to_draw(self):
        placed_cards = sum(len(row) for row in [self.board.top, self.board.middle, self.board.bottom])
        if placed_cards == 5:
            return 3
        elif placed_cards in [7, 10]:
            return 3
        elif placed_cards >= 13:
            return 0
        else:
            return 0

    def get_available_cards(self):
        used_cards = set(self.discarded_cards + self.board.top + self.board.middle + self.board.bottom + list(self.selected_cards))
        return [card for card in self.deck if card not in used_cards]

    def get_actions(self):
        if self.is_terminal():
            return []
        num_cards = len(self.selected_cards)
        actions = []
        # Если в режиме фантазии и ≥13 карт, генерируем полные варианты
        if num_cards >= 13 and self.ai_settings.get('fantasyMode'):
            for p in itertools.permutations(self.selected_cards.cards):
                action = {
                    'top': list(p[:3]),
                    'middle': list(p[3:8]),
                    'bottom': list(p[8:13]),
                    'discarded': list(p[13:])
                }
                actions.append(action)
            return actions
        # Если 5 карт
        if num_cards == 5:
            for p in itertools.permutations(self.selected_cards.cards):
                if self.evaluate_hand([p[0]])[0] != 7:
                    actions.append({
                        'top': [p[0]],
                        'middle': [p[1], p[2]],
                        'bottom': [p[3], p[4]],
                        'discarded': None
                    })
        # Если 3 карты (последний ход: разместить 2 и отбросить 1)
        elif num_cards == 3:
            # Вычисляем, сколько мест ещё свободно в каждом ряду
            top_remaining = 3 - len(self.board.top)
            middle_remaining = 5 - len(self.board.middle)
            bottom_remaining = 5 - len(self.board.bottom)
            for discarded_index in range(3):
                remaining_cards = [card for i, card in enumerate(self.selected_cards) if i != discarded_index]
                # Перебираем возможное распределение оставшихся карт по рядам с учётом свободных слотов
                for top_count in range(0, min(len(remaining_cards), top_remaining) + 1):
                    for middle_count in range(0, min(len(remaining_cards) - top_count, middle_remaining) + 1):
                        bottom_count = len(remaining_cards) - top_count - middle_count
                        if bottom_count <= bottom_remaining:
                            action = {
                                'top': remaining_cards[:top_count],
                                'middle': remaining_cards[top_count:top_count + middle_count],
                                'bottom': remaining_cards[top_count + middle_count:],
                                'discarded': self.selected_cards[discarded_index]
                            }
                            actions.append(action)
        # Если 1 или 2 карты – аналогично (учитываем оставшиеся слоты)
        elif num_cards in [1, 2]:
            top_remaining = 3 - len(self.board.top)
            middle_remaining = 5 - len(self.board.middle)
            bottom_remaining = 5 - len(self.board.bottom)
            for top_count in range(0, min(num_cards, top_remaining) + 1):
                for middle_count in range(0, min(num_cards - top_count, middle_remaining) + 1):
                    bottom_count = num_cards - top_count - middle_count
                    if bottom_count <= bottom_remaining:
                        action = {
                            'top': list(self.selected_cards.cards[:top_count]),
                            'middle': list(self.selected_cards.cards[top_count:top_count + middle_count]),
                            'bottom': list(self.selected_cards.cards[top_count + middle_count:]),
                            'discarded': []
                        }
                        actions.append(action)
        return actions

    def apply_action(self, action):
        new_board = Board()
        new_board.top = self.board.top + action.get('top', [])
        new_board.middle = self.board.middle + action.get('middle', [])
        new_board.bottom = self.board.bottom + action.get('bottom', [])
        new_discarded_cards = self.discarded_cards[:]
        if 'discarded' in action and action['discarded']:
            if isinstance(action['discarded'], list):
                new_discarded_cards.extend(action['discarded'])
            else:
                new_discarded_cards.append(action['discarded'])
        return GameState(
            selected_cards=Hand(),
            board=new_board,
            discarded_cards=new_discarded_cards,
            ai_settings=self.ai_settings,
            deck=self.deck[:]  # копия колоды
        )

    def get_information_set(self):
        def card_to_string(card):
            return str(card)
        def sort_cards(cards):
            return sorted(cards, key=lambda card: (self.rank_map[card.rank], self.suit_map[card.suit]))
        top_str = ','.join(map(card_to_string, sort_cards(self.board.top)))
        middle_str = ','.join(map(card_to_string, sort_cards(self.board.middle)))
        bottom_str = ','.join(map(card_to_string, sort_cards(self.board.bottom)))
        discarded_str = ','.join(map(card_to_string, sort_cards(self.discarded_cards)))
        selected_str = ','.join(map(card_to_string, sort_cards(self.selected_cards)))
        return f"T:{top_str}|M:{middle_str}|B:{bottom_str}|D:{discarded_str}|S:{selected_str}"

    def get_payoff(self):
        if not self.is_terminal():
            raise ValueError("Game is not in a terminal state")
        if self.is_dead_hand():
            return -self.calculate_royalties()
        return self.calculate_royalties()

    def is_dead_hand(self):
        if not self.board.is_full():
            return False
        top_rank, _ = self.evaluate_hand(self.board.top)
        middle_rank, _ = self.evaluate_hand(self.board.middle)
        bottom_rank, _ = self.evaluate_hand(self.board.bottom)
        return top_rank > middle_rank or middle_rank > bottom_rank

    def calculate_royalties(self):
        if self.is_dead_hand():
            return 0
        royalties = 0
        lines = {'top': self.board.top, 'middle': self.board.middle, 'bottom': self.board.bottom}
        for line_name, cards in lines.items():
            rank, _ = self.evaluate_hand(cards)
            if line_name == 'top':
                if rank == 7:
                    royalties += 10 + Card.RANKS.index(cards[0].rank)
                elif rank == 8:
                    royalties += self.get_pair_bonus(cards)
                elif rank == 9:
                    royalties += self.get_high_card_bonus(cards)
            elif line_name == 'middle':
                if rank <= 6:
                    royalties += self.get_royalties_for_hand(rank) * 2
            elif line_name == 'bottom':
                if rank <= 6:
                    royalties += self.get_royalties_for_hand(rank)
        return royalties

    def get_royalties_for_hand(self, hand_rank):
        if hand_rank == 1:
            return 25
        elif hand_rank == 2:
            return 15
        elif hand_rank == 3:
            return 10
        elif hand_rank == 4:
            return 6
        elif hand_rank == 5:
            return 4
        elif hand_rank == 6:
            return 2
        return 0

    def get_line_score(self, line, cards):
        if not cards:
            return 0
        rank, score = self.evaluate_hand(cards)
        return score

    def get_pair_bonus(self, cards):
        if len(cards) != 3:
            return 0
        ranks = [card.rank for card in cards]
        for rank in Card.RANKS[::-1]:
            if ranks.count(rank) == 2:
                return 1 + Card.RANKS.index(rank) - Card.RANKS.index('6') if rank >= '6' else 0
        return 0

    def get_high_card_bonus(self, cards):
        if len(cards) != 3 or not all(isinstance(card, Card) for card in cards):
            return 0
        ranks = [card.rank for card in cards]
        if len(set(ranks)) == 3:
            high_card = max(ranks, key=Card.RANKS.index)
            return 1 if high_card == 'A' else 0
        return 0

    def get_fantasy_bonus(self):
        bonus = 0
        top_rank, _ = self.evaluate_hand(self.board.top)
        if top_rank <= 8 and self.board.top and self.board.top[0].rank in ['Q', 'K', 'A']:
            if self.ai_settings.get('fantasyType') == 'progressive':
                if self.board.top[0].rank == 'Q':
                    bonus += 14
                elif self.board.top[0].rank == 'K':
                    bonus += 15
                elif self.board.top[0].rank == 'A':
                    bonus += 16
                elif top_rank == 7:
                    bonus += 17
            else:
                bonus += 14
            if self.is_fantasy_repeat():
                bonus += 14
        return bonus

    def is_fantasy_repeat(self):
        top_rank, _ = self.evaluate_hand(self.board.top)
        bottom_rank, _ = self.evaluate_hand(self.board.bottom)
        if top_rank == 7:
            return True
        if bottom_rank <= 3:
            return True
        return False

    def calculate_royalty_bonus(self):
        """
        Рассчитывает бонусы согласно правилам:
          - Нижняя линия: +6, если комбинация – Фулл Хаус (или лучше)
          - Средняя линия: +4, если комбинация – Стрит
          - Верхняя линия: +7, если комбинация – пара дам (без фола)
          Общий бонус = сумма бонусов.
        """
        bonus = {'top': 0, 'middle': 0, 'bottom': 0, 'total': 0}
        # Верхняя линия
        if len(self.board.top) == 3:
            rank, _ = self.evaluate_hand(self.board.top)
            # Если комбинация – One Pair и самая высокая пара не ниже дам, и рука не мёртва
            if rank == 9:
                pair_cards = [card for card in self.board.top if self.board.top.count(card) >= 2]
                if pair_cards and pair_cards[0].rank in ['Q', 'K', 'A'] and not self.is_dead_hand():
                    bonus['top'] = 7
        # Средняя линия
        if len(self.board.middle) == 5:
            rank, _ = self.evaluate_hand(self.board.middle)
            # Если комбинация – Стрит
            if rank == 6:
                bonus['middle'] = 4
        # Нижняя линия
        if len(self.board.bottom) == 5:
            rank, _ = self.evaluate_hand(self.board.bottom)
            # Если комбинация – Фулл Хаус или лучше (т.е. rank от 1 до 4)
            if 1 <= rank <= 4:
                bonus['bottom'] = 6
        bonus['total'] = bonus['top'] + bonus['middle'] + bonus['bottom']
        return bonus

    def evaluate_hand(self, cards):
        if not cards or not all(isinstance(card, Card) for card in cards):
            return 11, 0
        n = len(cards)
        if n == 5:
            if self.is_royal_flush(cards):
                return 1, 25
            if self.is_straight_flush(cards):
                return 2, 15
            if self.is_four_of_a_kind(cards):
                rank = [card.rank for card in cards if cards.count(card) == 4][0]
                return 3, 10 + Card.RANKS.index(rank) / 100
            if self.is_full_house(cards):
                rank = [card.rank for card in cards if cards.count(card) == 3][0]
                return 4, 6 + Card.RANKS.index(rank) / 100
            if self.is_flush(cards):
                score = 4 + sum(Card.RANKS.index(card.rank) for card in cards) / 1000
                return 5, score
            if self.is_straight(cards):
                score = 2 + sum(Card.RANKS.index(card.rank) for card in cards) / 1000
                return 6, score
            if self.is_three_of_a_kind(cards):
                rank = [card.rank for card in cards if cards.count(card) == 3][0]
                return 7, 2 + Card.RANKS.index(rank) / 100
            if self.is_two_pair(cards):
                ranks = sorted([Card.RANKS.index(card.rank) for card in cards if cards.count(card) == 2], reverse=True)
                return 8, sum(ranks) / 1000
            if self.is_one_pair(cards):
                rank = [card.rank for card in cards if cards.count(card) == 2][0]
                return 9, Card.RANKS.index(rank) / 1000
            score = sum(Card.RANKS.index(card.rank) for card in cards) / 10000
            return 10, score
        elif n == 3:
            if self.is_three_of_a_kind(cards):
                rank = cards[0].rank
                return 7, 10 + Card.RANKS.index(rank)
            if self.is_one_pair(cards):
                rank = [card.rank for card in cards if cards.count(card) == 2][0]
                return 8, self.get_pair_bonus(cards)
            return 9, self.get_high_card_bonus(cards)
        else:
            return 11, 0

    def is_royal_flush(self, cards):
        if not self.is_flush(cards):
            return False
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        return ranks == [8, 9, 10, 11, 12]

    def is_straight_flush(self, cards):
        return self.is_straight(cards) and self.is_flush(cards)

    def is_four_of_a_kind(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 4 for r in ranks)

    def is_full_house(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 3 for r in ranks) and any(ranks.count(r) == 2 for r in ranks)

    def is_flush(self, cards):
        suits = [card.suit for card in cards]
        return len(set(suits)) == 1

    def is_straight(self, cards):
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        if ranks == [0, 1, 2, 3, 12]:
            return True
        return all(ranks[i + 1] - ranks[i] == 1 for i in range(len(ranks) - 1))

    def is_three_of_a_kind(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 3 for r in ranks)

    def is_two_pair(self, cards):
        ranks = [card.rank for card in cards]
        pairs = [r for r in set(ranks) if ranks.count(r) == 2]
        return len(pairs) == 2

    def is_one_pair(self, cards):
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 2 for r in ranks)

class CFRNode:
    def __init__(self, actions):
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.actions = actions

    def get_strategy(self, realization_weight):
        normalizing_sum = 0
        strategy = defaultdict(float)
        for a in self.actions:
            strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
            normalizing_sum += strategy[a]
        for a in self.actions:
            if normalizing_sum > 0:
                strategy[a] /= normalizing_sum
            else:
                strategy[a] = 1.0 / len(self.actions)
            self.strategy_sum[a] += realization_weight * strategy[a]
        return strategy

    def get_average_strategy(self):
        avg_strategy = defaultdict(float)
        normalizing_sum = sum(self.strategy_sum.values())
        if normalizing_sum > 0:
            for a in self.actions:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
        else:
            for a in self.actions:
                avg_strategy[a] = 1.0 / len(self.actions)
        return avg_strategy

class CFRAgent:
    def __init__(self, iterations=1000, stop_threshold=0.001):
        self.nodes = {}
        self.iterations = iterations
        self.stop_threshold = stop_threshold

    def cfr(self, game_state, p0, p1, timeout_event, result):
        if timeout_event.is_set():
            print("CFR timed out!")
            return 0
        if game_state.is_terminal():
            payoff = game_state.get_payoff()
            print(f"cfr terminal state. Payoff: {payoff}")
            return payoff
        player = game_state.get_current_player()
        info_set = game_state.get_information_set()
        print(f"cfr for info_set: {info_set}, player: {player}")
        if info_set not in self.nodes:
            actions = game_state.get_actions()
            if not actions:
                print("No actions available.")
                return 0
            self.nodes[info_set] = CFRNode(actions)
        node = self.nodes[info_set]
        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = defaultdict(float)
        node_util = 0
        for a in node.actions:
            if timeout_event.is_set():
                print("CFR timed out in loop!")
                return 0
            next_state = game_state.apply_action(a)
            if player == 0:
                util[a] = -self.cfr(next_state, p0 * strategy[a], p1, timeout_event, result)
            else:
                util[a] = -self.cfr(next_state, p0, p1 * strategy[a], timeout_event, result)
            node_util += strategy[a] * util[a]
        if player == 0:
            for a in node.actions:
                node.regret_sum[a] += p1 * (util[a] - node_util)
        else:
            for a in node.actions:
                node.regret_sum[a] += p0 * (util[a] - node_util)
        print(f"cfr returning for info_set: {info_set}, node_util: {node_util}")
        return node_util

    def train(self, timeout_event, result):
        for i in range(self.iterations):
            if timeout_event.is_set():
                print(f"Training stopped at {i} iterations.")
                break
            all_cards = Card.get_all_cards()
            random.shuffle(all_cards)
            game_state = GameState(deck=all_cards)
            game_state.selected_cards = Hand(all_cards[:5])
            self.cfr(game_state, 1, 1, timeout_event, result)
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.iterations} complete.")
                if self.check_convergence():
                    print("CFR converged after", i + 1, "iterations.")
                    break

    def check_convergence(self):
        for node in self.nodes.values():
            avg_strategy = node.get_average_strategy()
            for action, prob in avg_strategy.items():
                if abs(prob - 1.0 / len(node.actions)) > self.stop_threshold:
                    return False
        return True

    def baseline_evaluation(self, state):
        if state.is_dead_hand():
            return -1000
        score = 0
        top_score = state.get_line_score('top', state.board.top) * 4
        middle_score = state.get_line_score('middle', state.board.middle) * 2.5
        bottom_score = state.get_line_score('bottom', state.board.bottom) * 1.5
        score += top_score + middle_score + bottom_score
        available_cards = state.get_available_cards()
        score += self.calculate_potential(state.board.top, 'top', state.board, available_cards) * 6
        score += self.calculate_potential(state.board.middle, 'middle', state.board, available_cards) * 4
        score += self.calculate_potential(state.board.bottom, 'bottom', state.board, available_cards) * 3
        if state.ai_settings.get('fantasyMode'):
            fantasy_bonus = state.get_fantasy_bonus()
            if fantasy_bonus > 0:
                score += fantasy_bonus * 10
            else:
                score -= 50
        if score > 500:
            return score * 1.2
        return score

    def calculate_potential(self, cards, line, board, available_cards):
        potential = 0
        num_cards = len(cards)
        if num_cards < 5 and line != 'top':
            if self.is_straight_potential(cards, available_cards):
                potential += 0.5
            if self.is_flush_potential(cards, available_cards):
                potential += 0.7
        if num_cards == 2 and line == 'top':
            if self.is_pair_potential(cards, available_cards):
                potential += 0.3
        return potential

    def is_flush_potential(self, cards, available_cards):
        if len(cards) < 2:
            return False
        suit_counts = defaultdict(int)
        for card in cards:
            suit_counts[card.suit] += 1
        for suit, count in suit_counts.items():
            if count >= 2:
                remaining_needed = 5 - count
                available_of_suit = sum(1 for card in available_cards if card.suit == suit)
                if available_of_suit >= remaining_needed:
                    return True
        return False

    def is_straight_potential(self, cards, available_cards):
        if len(cards) < 2:
            return False
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        for i in range(len(ranks) - 1):
            if ranks[i + 1] - ranks[i] == 1:
                return True
        if len(ranks) >= 2:
            for i in range(len(ranks) - 1):
                if ranks[i + 1] - ranks[i] == 2:
                    needed_rank = ranks[i] + 1
                    if any(Card.RANKS.index(card.rank) == needed_rank for card in available_cards):
                        return True
        if len(ranks) >= 2:
            for i in range(len(ranks) - 1):
                if ranks[i + 1] - ranks[i] == 3:
                    needed_ranks = [ranks[i] + 1, ranks[i] + 2]
                    if sum(1 for card in available_cards if Card.RANKS.index(card.rank) in needed_ranks) >= 1:
                        return True
        if ranks == [0, 1, 2, 3]:
            if any(card.rank == 'A' for card in available_cards):
                return True
        if ranks == [0, 1, 2, 12]:
            if any(card.rank == '3' for card in available_cards):
                return True
        if ranks == [0, 1, 11, 12]:
            if any(card.rank == '2' for card in available_cards):
                return True
        if ranks == [0, 10, 11, 12]:
            if any(card.rank == '1' for card in available_cards):
                return True
        return False

    def is_pair_potential(self, cards, available_cards):
        if len(cards) != 2:
            return False
        if cards[0].rank == cards[1].rank:
            rank = cards[0].rank
            if sum(1 for card in available_cards if card.rank == rank) >= 1:
                return True
        return False

    def get_move(self, game_state, num_cards, timeout_event, result):
        print("Inside get_move")
        actions = game_state.get_actions()
        print(f"Available actions: {actions}")
        if not actions:
            result['move'] = {'error': 'Нет доступных ходов'}
            return
        # Если включён режим фантазии, фильтруем варианты, чтобы избежать фола
        if game_state.ai_settings.get('fantasyMode'):
            fantasy_actions = []
            for action in actions:
                next_state = game_state.apply_action(action)
                if not next_state.is_dead_hand():
                    fantasy_actions.append(action)
            if fantasy_actions:
                actions = fantasy_actions
        best_action = None
        best_value = float('-inf')
        for action in actions:
            next_state = game_state.apply_action(action)
            value = self.baseline_evaluation(next_state)
            if value > best_value:
                best_value = value
                best_action = action
        if best_action is None:
            print("No best action found, choosing random.")
            best_action = random.choice(actions) if actions else None
        print(f"Selected move: {best_action}")
        result['move'] = best_action

    def evaluate_move(self, game_state, action, timeout_event):
        next_state = game_state.apply_action(action)
        info_set = next_state.get_information_set()
        if info_set in self.nodes:
            node = self.nodes[info_set]
            strategy = node.get_average_strategy()
            expected_value = 0
            for a, prob in strategy.items():
                if timeout_event.is_set():
                    return 0
                expected_value += prob * self.get_action_value(next_state, a, timeout_event)
            return expected_value
        else:
            return self.shallow_search(next_state, 2, timeout_event)

    def shallow_search(self, state, depth, timeout_event):
        if depth == 0 or state.is_terminal() or timeout_event.is_set():
            return self.baseline_evaluation(state)
        best_value = float('-inf')
        for action in state.get_actions():
            if timeout_event.is_set():
                return 0
            value = -self.shallow_search(state.apply_action(action), depth - 1, timeout_event)
            best_value = max(best_value, value)
        return best_value

    def get_action_value(self, state, action, timeout_event):
        num_simulations = 10
        total_score = 0
        for _ in range(num_simulations):
            if timeout_event.is_set():
                return 0
            simulated_state = state.apply_action(action)
            while not simulated_state.is_terminal():
                actions = simulated_state.get_actions()
                if not actions:
                    break
                random_action = random.choice(actions)
                simulated_state = simulated_state.apply_action(random_action)
            total_score += self.baseline_evaluation(simulated_state)
        return total_score / num_simulations if num_simulations > 0 else 0

    def save_progress(self):
        data = {
            'nodes': self.nodes,
            'iterations': self.iterations,
            'stop_threshold': self.stop_threshold
        }
        utils.save_data(data, 'cfr_data.pkl')

    def load_progress(self):
        data = utils.load_data('cfr_data.pkl')
        if data:
            self.nodes = data['nodes']
            self.iterations = data['iterations']
            self.stop_threshold = data.get('stop_threshold', 0.001)

class RandomAgent:
    def __init__(self):
        pass

    def get_move(self, game_state, num_cards, timeout_event, result):
        print("Inside RandomAgent get_move")
        actions = game_state.get_actions()
        print(f"Available actions: {actions}")
        if not actions:
            result['move'] = {'error': 'Нет доступных ходов'}
            return
        best_move = random.choice(actions) if actions else None
        print(f"Selected move: {best_move}")
        result['move'] = best_move

# Создание глобальных агентов (инициализируются в app.py)
cfr_agent = CFRAgent()
random_agent = RandomAgent()
