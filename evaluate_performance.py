import time
import tracemalloc
import importlib
import logging

# Setup logging
logging.basicConfig(filename="performance_metrics.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# List of scripts to evaluate
scripts = [
    "preprocess",
    "npy",
    "load_and_process_json",
    "train_model",
    "model",
    "app",
    "chatbot",
]

# Function to evaluate performance
def evaluate_script(script_name):
    try:
        logging.info(f"Evaluating {script_name}.py")
        start_time = time.time()
        tracemalloc.start()

        # Dynamically import and run the script
        module = importlib.import_module(script_name)
        if hasattr(module, "main"):  # Check if a main function exists
            module.main()

        # Performance metrics
        runtime = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logging.info(f"{script_name}.py executed successfully.")
        logging.info(f"Runtime: {runtime:.2f} seconds")
        logging.info(f"Memory Usage: Current = {current / 1024:.2f} KB, Peak = {peak / 1024:.2f} KB")
        
    except Exception as e:
        logging.error(f"Error while evaluating {script_name}.py: {e}")

# Main script to evaluate all
if __name__ == "__main__":
    for script in scripts:
        evaluate_script(script)
    logging.info("Evaluation complete. Check performance_metrics.log for details.")
