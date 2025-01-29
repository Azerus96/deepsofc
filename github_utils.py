=== FILE: github_utils.py ===
from github import Github, GithubException
import base64
import os
import logging

def connect_to_github():
    """Установка соединения с GitHub"""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise ValueError("GitHub token not configured")
    return Github(token)

def save_progress_to_github(filename, repo_name, branch='main'):
    """Полная реализация сохранения на GitHub"""
    try:
        g = connect_to_github()
        repo = g.get_user().get_repo(repo_name)
        
        with open(filename, 'rb') as f:
            content = base64.b64encode(f.read()).decode('utf-8')
            
        try:
            file = repo.get_contents(filename, ref=branch)
            repo.update_file(file.path, f"Update {filename}", content, file.sha, branch=branch)
        except GithubException:
            repo.create_file(filename, f"Create {filename}", content, branch=branch)
            
        logging.info(f"Файл {filename} успешно сохранен на GitHub")
        return True
    
    except GithubException as e:
        logging.error(f"GitHub API error: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"General error: {str(e)}")
        return False

def load_progress_from_github(filename, repo_name, branch='main'):
    """Полная реализация загрузки с GitHub"""
    try:
        g = connect_to_github()
        repo = g.get_user().get_repo(repo_name)
        file = repo.get_contents(filename, ref=branch)
        content = base64.b64decode(file.content).decode('utf-8')
        
        with open(filename, 'w') as f:
            f.write(content)
            
        logging.info(f"Файл {filename} успешно загружен с GitHub")
        return True
    
    except GithubException as e:
        logging.error(f"GitHub API error: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"General error: {str(e)}")
        return False

def backup_to_github(data, repo_name):
    """Создание резервной копии данных на GitHub"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"backups/game_state_{timestamp}.json"
    return save_progress_to_github(filename, repo_name)
