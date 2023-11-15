import tempfile
import zipfile
from pathlib import Path
import mechanicalsoup


def login(browser, username, password):
    res = browser.open("https://id.tecnico.ulisboa.pt/cas/login")
    res.raise_for_status()
    browser.select_form('form[id="credential"]')
    browser["username"] = username
    browser["password"] = password

    res = browser.submit_selected()
    res.raise_for_status()

def download_file(browser, url, tmpfile):
    res = browser.session.get(url)
    res.raise_for_status()
    tmpfile.write(res.content)

def download_and_extract(browser, url, filename):
    with tempfile.TemporaryFile() as tmpfile:
        download_file(browser, url, tmpfile)
        with zipfile.ZipFile(tmpfile, 'r') as zip:
            zip.extract(filename, "./data")


if __name__ == "__main__":
    username = input("IST number: ")
    password = input("Password: ")

    browser = mechanicalsoup.StatefulBrowser()
    login(browser, username, password)

    download_and_extract(
        browser,
        "https://fenix.tecnico.ulisboa.pt/downloadFile/2815368242404009/class_pos_covid.csv.zip",
        "class_pos_covid.csv"
    )

    download_and_extract(
        browser,
        "https://fenix.tecnico.ulisboa.pt/downloadFile/2815368242404010/class_credit_score.csv.zip",
        "class_credit_score.csv"
    )





