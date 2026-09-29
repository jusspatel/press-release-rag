import csv
import os
import re
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_FILE = os.path.join(BASE_DIR, "data", "pib_urls.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "pib_pdfs")
START_ROW = 2000  # 1-based index


def generate_pdfs_from_csv():
    if not os.path.exists(CSV_FILE):
        print(f"[!] Error: {CSV_FILE} not found. Run Stage 1 first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CSV_FILE, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total_records = len(reader)
    print(f"[*] Found {total_records} items in {CSV_FILE}.")
    print(f"[*] Resuming from row {START_ROW}...")

    # Slice from START_ROW - 1 to the end
    records_to_process = reader[START_ROW - 1 :]

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
        )
        page = context.new_page()

        for idx, row in enumerate(records_to_process, start=START_ROW):
            prid = row["prid"]
            title = row["title"]
            url = row["url"]
            year = row["year"]
            month = row["month"]

            safe_title = (
                re.sub(r'[\\/*?:"<>|]', "", title).strip()[:50].replace(" ", "_")
            )
            filename = f"{year}-{month}_{prid}_{safe_title}.pdf"
            file_path = os.path.join(OUTPUT_DIR, filename)

            # Skip if already exists and is not 0 bytes
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                continue

            print(f"[{idx}/{total_records}] Downloading PRID {prid}...")
            try:
                res = page.goto(url, wait_until="networkidle", timeout=30000)
                if not res or not res.ok:
                    status = res.status if res else "None"
                    print(f"    [!] Failed to load {url} (status: {status})")
                    continue

                page.emulate_media(media="print")
                page.pdf(
                    path=file_path,
                    format="A4",
                    print_background=True,
                    margin={
                        "top": "15mm",
                        "bottom": "15mm",
                        "left": "15mm",
                        "right": "15mm",
                    },
                )
            except Exception as e:
                print(f"    [!] Error printing {url}: {e}")

        browser.close()

    print("[*] PDF export finished.")


if __name__ == "__main__":
    generate_pdfs_from_csv()