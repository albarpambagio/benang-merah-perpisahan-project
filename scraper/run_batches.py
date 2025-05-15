import subprocess
import time
import os
import logging
from tqdm import tqdm
from pathlib import Path
import json
import datetime
import math

# Configure logging
logging.basicConfig(
    filename='run_batches.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

BATCH_SLEEP = 10  # seconds to wait between batches
MAX_ATTEMPTS = 1000  # safety limit to avoid infinite loops
PROGRESS_FILE = Path('data/batch_progress.json')

completed_file = Path('data/completed_cases.txt')

def load_progress():
    if PROGRESS_FILE.exists():
        with PROGRESS_FILE.open() as f:
            return json.load(f)
    return {"last_completed": 0, "attempt": 0, "batches_completed": 0, "start_time": time.time()}

def save_progress(progress):
    with PROGRESS_FILE.open('w') as f:
        json.dump(progress, f)

def get_completed_count():
    if completed_file.exists():
        with completed_file.open() as f:
            completed = set(line.strip() for line in f)
        return len(completed)
    return 0

def run_batch_with_retries(max_retries=3):
    for attempt in range(max_retries):
        result = subprocess.run(['python', 'scraper.py'], cwd=Path(__file__).parent)
        if result.returncode == 0:
            return True
        else:
            wait_time = min(60, 2 ** attempt)  # exponential backoff, max 60s
            logging.warning(f"Batch failed (attempt {attempt+1}), retrying in {wait_time}s...")
            time.sleep(wait_time)
    return False

def main():
    print("Starting batch scraping automation...")
    logging.info("Batch scraping automation started.")
    progress = load_progress()
    last_completed = progress.get("last_completed", 0)
    attempt = progress.get("attempt", 0)
    batches_completed = progress.get("batches_completed", 0)
    start_time = progress.get("start_time", time.time())
    initial_completed = get_completed_count()
    if last_completed < initial_completed:
        last_completed = initial_completed
    try:
        with tqdm(desc='Cases Completed', initial=last_completed, unit='case') as case_pbar:
            while attempt < MAX_ATTEMPTS:
                attempt += 1
                print(f"\n[Batch {attempt}] Running scraper.py...")
                logging.info(f"Starting batch {attempt}")
                batch_success = run_batch_with_retries(max_retries=3)
                if not batch_success:
                    print(f"scraper.py failed after retries. Stopping automation.")
                    logging.error(f"scraper.py failed after retries. Stopping automation.")
                    break
                num_completed = get_completed_count()
                print(f"Completed cases so far: {num_completed}")
                logging.info(f"Completed cases after batch {attempt}: {num_completed}")
                # Update progress bar by the number of new cases completed
                case_pbar.update(num_completed - last_completed)
                batches_completed += 1
                elapsed = time.time() - start_time
                rate = num_completed / elapsed if elapsed > 0 else 0
                case_pbar.set_postfix({
                    'batches': batches_completed,
                    'rate': f"{rate:.2f} cases/s",
                })
                # Save progress state
                progress = {
                    "last_completed": num_completed,
                    "attempt": attempt,
                    "batches_completed": batches_completed,
                    "start_time": start_time
                }
                save_progress(progress)
                # Check if new cases were processed
                if num_completed == last_completed:
                    print("No new cases found in this batch. Scraping is complete.")
                    logging.info("No new cases found. Stopping automation.")
                    break
                last_completed = num_completed
                print(f"Waiting {BATCH_SLEEP} seconds before next batch...")
                time.sleep(BATCH_SLEEP)
            else:
                print("Reached maximum number of attempts. Stopping.")
                logging.warning("Reached max attempts. Stopping automation.")
    except KeyboardInterrupt:
        print("\nBatch scraping interrupted by user.")
        logging.info("Batch scraping interrupted by user.")
    print("Batch scraping automation finished.")
    logging.info("Batch scraping automation finished.")
    # --- Summary Report ---
    end_time = time.time()
    duration = end_time - start_time
    summary = (
        f"\nSummary Report\n"
        f"==============\n"
        f"Total cases completed: {last_completed}\n"
        f"Total batches run: {batches_completed}\n"
        f"Total runtime: {str(datetime.timedelta(seconds=int(duration)))}\n"
        f"Average rate: {(last_completed / duration) if duration > 0 else 0:.2f} cases/sec\n"
        f"=============="
    )
    print(summary)
    logging.info(summary)

def run():
    main()
    # Play a sound alert when scraping is complete (cross-platform)
    try:
        import platform
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 700)  # frequency, duration(ms)
        elif platform.system() == "Darwin":  # macOS
            os.system('afplay /System/Library/Sounds/Glass.aiff')
        else:  # Linux/Unix
            os.system('aplay /usr/share/sounds/alsa/Front_Center.wav')
    except Exception as e:
        print("Could not play sound alert:", e)

def launch_parallel_batches(num_batches=5, last_page=499):
    pages_per_batch = math.ceil(last_page / num_batches)
    procs = []
    for i in range(num_batches):
        start = i * pages_per_batch + 1
        end = min((i + 1) * pages_per_batch, last_page)
        print(f"Launching batch {i+1}: pages {start} to {end}")
        proc = subprocess.Popen([
            'python', 'scraper.py',
            f'--start_page={start}',
            f'--end_page={end}'
        ], cwd=Path(__file__).parent)
        procs.append(proc)
    # Wait for all batches to finish
    for proc in procs:
        proc.wait()
    print("All parallel batches finished.")

if __name__ == "__main__":
    import sys
    if '--parallel' in sys.argv:
        # Example: python run_batches.py --parallel 5
        idx = sys.argv.index('--parallel')
        num_batches = int(sys.argv[idx+1]) if len(sys.argv) > idx+1 else 5
        launch_parallel_batches(num_batches=num_batches, last_page=499)
    else:
        run() 