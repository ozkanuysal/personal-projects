import requests
import luigi
from bs4 import BeautifulSoup
from collections import Counter
import pickle

class GetTopBooks(luigi.Task):
    """
    Retrieves list of the most popular books from Project Gutenberg.
    Saves the list as a .txt file.
    """
    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget("data/books_list.txt")

    def run(self) -> None:
        resp = requests.get("http://www.gutenberg.org/browse/scores/top")
        soup = BeautifulSoup(resp.content, "html.parser")
        pageHeader = soup.find_all("h2", string="Top 100 EBooks yesterday")[0]
        listTop = pageHeader.find_next_sibling("ol")

        with self.output().open("w") as f:
            for result in listTop.select("li>a"):
                if "/ebooks/" in result["href"]:
                    f.write(f"http://www.gutenberg.org{result['href']}.txt.utf-8\n")
class DownloadBooks(luigi.Task):
    """
    Downloads text from a specified Project Gutenberg book.
    Cleans and lowercases the text.
    """
    FileID = luigi.IntParameter()
    REPLACE_LIST = """.,"';_[]:*-"""

    def requires(self):
        return GetTopBooks()

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(f"data/downloads/{self.FileID}.txt")

    def run(self) -> None:
        with self.input().open("r") as i:
            url = i.read().splitlines()[self.FileID]
            response = requests.get(url)
            book_text = response.text

            for char in self.REPLACE_LIST:
                book_text = book_text.replace(char, " ")

            book_text = book_text.lower()

            with self.output().open("w") as outfile:
                outfile.write(book_text)
class CountWords(luigi.Task):
    """
    Counts word frequencies in a downloaded book using Counter.
    Dumps results into a pickle file.
    """
    FileID = luigi.IntParameter()

    def requires(self):
        return DownloadBooks(FileID=self.FileID)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(f"data/counts/count_{self.FileID}.pickle", format=luigi.format.Nop)

    def run(self) -> None:
        with self.input().open("r") as infile:
            word_count = Counter(infile.read().split())

        with self.output().open("wb") as outfile:
            pickle.dump(word_count, outfile)

class GlobalParams(luigi.Config):
    """Global configuration for the pipeline."""
    NumberBooks: int = 10
    NumberTopWords: int = 500

class TopWords(luigi.Task):
    """
    Aggregates word counts from multiple books and outputs the top words.
    """
    def requires(self):
        return [CountWords(FileID=i) for i in range(GlobalParams().NumberBooks)]

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget("data/summary.txt")

    def run(self) -> None:
        total_count = Counter()
        for input_file in self.input():
            with input_file.open("rb") as infile:
                next_counter = pickle.load(infile)
                total_count += next_counter

        with self.output().open("w") as outfile:
            for word, freq in total_count.most_common(GlobalParams().NumberTopWords):
                outfile.write(f"{word:<15}{freq}\n")