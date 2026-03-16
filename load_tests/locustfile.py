from locust import HttpUser, task, between
import random

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


class ModerateUser(HttpUser):
    wait_time = between(0.1, 0.5)
    headers = {
        "Authorization": "Bearer dev-key",
        "Content-Type": "application/json",
    }

    @task
    def moderate(self):
        self.client.post(
            "/moderate",
            json={"text": random.choice(SAMPLE_TEXTS)},
            headers=self.headers,
        )
