from job_run_loop import start_loop
from main import train


def run(data):
    topics = train(data)
    return topics


start_loop(call_method=run)