# luigi_task.py
import luigi
import requests
import time
import os

class CallAPI(luigi.Task):
    num = luigi.IntParameter()

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget(f"factorial_{self.num}_response.txt")

    def _append_response_to_file(self, data):
        old_content = ""
        if os.path.exists(self.output().path):
            with self.output().open('r') as f:
                old_content = f.read()
        with self.output().open('w') as f:
            f.write(old_content + data + '\n')

    def run(self):
        response = requests.get(f'http://localhost:8000/factorial/{self.num}')
        if response.ok:
            self._append_response_to_file(str(response.json()))

class AnotherTask(luigi.Task):
    num = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(f'another_task_{self.num}.txt')

    def run(self):
        with self.output().open('w') as f:
            f.write(f'This is another task with num : {self.num}\n')


if __name__ == '__main__':
    luigi.run()