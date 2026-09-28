import csv
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
YEAR_TO_DOWNLOAD = "2026"
CSV_FILE = "pib_urls.csv"
BASE_URL = "https://www.pib.gov.in/allRel.aspx?reg=48&lang=1"

MONTHS = [
    ("1", "January"), ("2", "February"), ("3", "March"),
    ("4", "April"), ("5", "May"), ("6", "June"),
    ("7", "July"), ("8", "August"), ("9", "September"),
    ("10", "October"), ("11", "November"), ("12", "December")
]

MINISTRY_SELECT = "#ContentPlaceHolder1_ddlMinistry"
DAY_SELECT = "#ContentPlaceHolder1_ddlday"
MONTH_SELECT = "#ContentPlaceHolder1_ddlMonth"
YEAR_SELECT = "#ContentPlaceHolder1_ddlYear"
LINK_SELECTOR = "a[href*='PressReleaseDetail.aspx?PRID=']"


def wait_for_aspnet_idle(page, timeout=10000):
    """Waits for network requests to finish and ensures the DOM is settled."""
    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except Exception:
        pass


def harvest_links_to_csv():
    records = []

    with sync_playwright() as p:
        print("[*] Launching VISIBLE Microsoft Edge window...")
        browser = p.chromium.launch(
            channel="msedge",
            headless=False,     # Visible browser window
            slow_mo=1000,       # 1-second delay between actions so you can watch
            args=["--start-maximized"]
        )
        context = browser.new_context(no_viewport=True)
        page = context.new_page()

        print(f"[*] Navigating to {BASE_URL}...")
        page.goto(BASE_URL, wait_until="domcontentloaded")
        wait_for_aspnet_idle(page)

        # 1. Ministry: "0" = All Ministry
        print("[*] Selecting: Ministry -> All")
        page.select_option(MINISTRY_SELECT, value="0")
        wait_for_aspnet_idle(page)

        # 2. Year: Match by label (e.g. "2026")
        print(f"[*] Selecting: Year -> {YEAR_TO_DOWNLOAD}")
        page.select_option(YEAR_SELECT, label=YEAR_TO_DOWNLOAD)
        wait_for_aspnet_idle(page)

        # 3. Day: On PIB ASP.NET, "0" represents "All" days
        print("[*] Selecting: Day -> All")
        try:
            page.select_option(DAY_SELECT, value="0")
        except Exception:
            page.select_option(DAY_SELECT, index=0)
        wait_for_aspnet_idle(page)

        # 4. Loop Months
        for month_val, month_name in MONTHS:
            print(f"\n[*] Processing Month: {month_name} ({month_val})...")
            try:
                # Select using month value ("1" for Jan, "2" for Feb, etc.)
                page.select_option(MONTH_SELECT, value=month_val)
                wait_for_aspnet_idle(page)
                
                # Brief pause to ensure the postback finishes rendering inner tables
                page.wait_for_timeout(1500)
            except Exception as e:
                print(f"    [!] Failed to set month {month_name}: {e}")
                continue

            # Parse with BeautifulSoup
            soup = BeautifulSoup(page.content(), "html.parser")
            found_anchors = soup.find_all("a", href=re.compile(r"PressReleaseDetail\.aspx\?PRID="))

            print(f"    -> Extracted {len(found_anchors)} links for {month_name}.")
            for a in found_anchors:
                raw_href = a.get("href", "")
                full_url = urljoin(page.url, raw_href)
                title = a.get_text(strip=True)
                prid_match = re.search(r"PRID=(\d+)", raw_href)
                prid = prid_match.group(1) if prid_match else "unknown"

                records.append({
                    "year": YEAR_TO_DOWNLOAD,
                    "month": month_val.zfill(2),
                    "prid": prid,
                    "title": title,
                    "url": full_url
                })

        print("\n[*] Closing browser in 5 seconds...")
        page.wait_for_timeout(5000)
        browser.close()

    # Save to CSV
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "month", "prid", "title", "url"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[✓] Done. Saved {len(records)} total links to {CSV_FILE}.")


if __name__ == "__main__":
    harvest_links_to_csv()