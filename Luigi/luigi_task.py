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

    def run(self):
        response = requests.get(f'http://localhost:8000/factorial/{self.num}')
        old_content = ""
        if os.path.exists(self.output().path):
            with self.output().open('r') as f:
                old_content = f.read()
        with self.output().open('w') as f:
            f.write(old_content + str(response.json()) + '\n')
class AnotherTask(luigi.Task):
    num = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(f'another_task_{self.num}.txt')

    def run(self):
        with self.output().open('w') as f:
            f.write(f'This is another task with num : {self.num}\n')


if __name__ == '__main__':
        luigi.build([CallAPI(num=5), AnotherTask(num=10)], local_scheduler=True)   
        print('all task completed.')