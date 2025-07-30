# reconstruct_order_book_snapshots.py

import pandas as pd
from pathlib import Path

def reconstruct_minute_snapshots(base_dir, stock_folders):
    all_snapshots = []

    for stock in stock_folders:
        stock_path = Path(base_dir) / stock
        if not stock_path.exists():
            print(f"Folder not found: {stock_path}")
            continue

        print(f"Processing folder: {stock}")
        for file in stock_path.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['ts_event'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df['minute'] = df['timestamp'].dt.floor('min')

                book_columns = ['minute'] + \
                    [f'ask_px_0{i}' for i in range(10)] + [f'ask_sz_0{i}' for i in range(10)] + \
                    [f'bid_px_0{i}' for i in range(10)] + [f'bid_sz_0{i}' for i in range(10)]

                df_book = df[book_columns].copy()
                df_book['stock'] = stock

                snapshot = df_book.groupby(['stock', 'minute']).last().reset_index()
                all_snapshots.append(snapshot)

            except Exception as e:
                print(f"Failed to process {file.name}: {e}")

    if all_snapshots:
        combined = pd.concat(all_snapshots, ignore_index=True)
        output_path = Path(base_dir) / "all_minute_order_book_snapshots.csv"
        combined.to_csv(output_path, index=False)
        print(f"\n✅ Snapshot saved to: {output_path}")
    else:
        print("⚠️ No snapshots were created. Check your file paths or formats.")

if __name__ == "__main__":
    # Adjust this to the actual path where your stock folders are located
    base_directory = "./"  # Use "." if running from the same folder
    stock_folder_names = ["FROG", "SOUN", "CRWV"]

    reconstruct_minute_snapshots(base_directory, stock_folder_names)
