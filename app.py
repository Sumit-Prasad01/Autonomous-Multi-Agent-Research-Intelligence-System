import subprocess
import threading
import time
from dotenv import load_dotenv

from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)

load_dotenv()


def run_backend():
    try:

        logger.info("Starting backend service.")
        subprocess.run(["python", "api/main.py"], check = True)

    except CustomException as e:
        logger.error("Problem with backend service")
        raise CustomException("Failed to start backend", e)
    

def run_frontend():
    try:

        logger.info("Starting frontend service")
        subprocess.run(["streamlit", "run", "frontend/ui.py"], check = True)
    
    except CustomException as e:
        logger.error("Problem with frontend service")
        raise CustomException("Failed to start frontend", e)
    


if __name__ == "__main__":
    try:

        threading.Thread(target = run_backend).start()
        time.sleep(2)
        run_frontend()

    except CustomException as e:
        logger.exception(f"Custom Exception occured : {str(e)}")