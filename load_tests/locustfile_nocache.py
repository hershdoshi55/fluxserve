from locust import HttpUser, task, between
import random
import itertools

SAMPLE_TEXTS = [
    "I love this community and everyone in it!",
    "You should all go to hell, I hate you",
    "The weather is great today",
    "I will find you and make you regret this",
    "Thanks for your help, really appreciate it",
    "This is absolute garbage, worst product ever",
    "Anyone want to grab lunch today?",
    "Kill yourself you worthless piece of trash",
    "Just finished a great run!",
    "Why do you exist? You ruin everything",
]

_counter = itertools.count()


class ModerateUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.headers = {"Authorization": "Bearer dev-key"}

    @task
    def moderate(self):
        text = random.choice(SAMPLE_TEXTS) + f" [{next(_counter)}]"
        self.client.post(
            "/moderate",
            json={"text": text, "max_new_tokens": 10},
            headers=self.headers,
        )
