=== FILE: utils.py ===
import pickle
import os
import logging

def save_data(data, filename):
    """Сохраняет данные в файл с обработкой ошибок"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Данные сохранены в {filename}")
    except Exception as e:
        logging.error(f"Ошибка сохранения данных: {str(e)}")
        raise

def load_data(filename):
    """Загружает данные из файла с обработкой ошибок"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.warning(f"Файл {filename} не найден")
        return None
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {str(e)}")
        return None

def validate_card(card_data):
    """Валидация структуры данных карты"""
    required = ['rank', 'suit']
    return all(key in card_data for key in required)

def log_game_action(action):
    """Логирование игровых действий"""
    logging.info(f"Action: {action}")

def backup_data(data, prefix='backup'):
    """Создание резервной копии данных"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}_{timestamp}.pkl"
    save_data(data, filename)
    return filename
