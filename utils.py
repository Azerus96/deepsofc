import pickle
import os

def save_data(data, filename):
    """Saves data to a file using pickle."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol for better performance
        print(f"Data saved to {filename}")
        return True # Indicate success
    except (pickle.PickleError, OSError, Exception) as e:
        print(f"Error saving data to {filename}: {e}")
        return False # Indicate failure


def load_data(filename):
    """Loads data from a file using pickle."""
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return None

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            print(f"Data loaded from {filename}")
            return data
    except (pickle.PickleError, OSError, EOFError, Exception) as e:
        print(f"Error loading data from {filename}: {e}")
        return None
