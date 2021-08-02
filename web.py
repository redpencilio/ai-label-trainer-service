from job_run_loop import start_loop
from .main import train

app.threaded = True


def run(data):
    topics = train(data)
    return topics


start_loop(call_method=run)