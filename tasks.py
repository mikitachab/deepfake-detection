from invoke import task

@task
def reformat(c):
    c.run("black deepfake_detection main.py")

