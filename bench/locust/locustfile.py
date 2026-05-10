from locust import HttpUser, task, between

class VllmUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def completions(self):
        self.client.post(
            "/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "prompt": "Summarize continuous batching in one paragraph.",
                "max_tokens": 128,
                "temperature": 0.0,
            },
            timeout=60,
        )
