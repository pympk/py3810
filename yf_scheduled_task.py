# Run yf_scheduled_task
# You can schedule a Python script to run on a Windows 10 PC using the built-in "Task Scheduler" tool. Here's the process:
#     Open the Task Scheduler by searching for "Task Scheduler" in the Windows search bar.
#     Click on "Create Basic Task" in the right-side panel.
#     Enter a name for the task and a description (optional).
#     Choose the frequency of running the task (e.g. daily, weekly, etc.).
#     Set the start time for the task.
#     Choose "Start a program" as the action to be performed.
#     In the "Program/script" field, enter the path to your Python executable, usually "C:\PythonXX\python.exe", where XX is the version number.
#     In the "Add arguments" field, enter the path to your Python script file.
#     Click on "Next" and then "Finish" to save the task.
# The task will now run according to the schedule you set.
# Note: Make sure your Python script runs without any errors when you run it manually before scheduling it.


import os
import datetime

filename = "_scheduled_task.txt"
filepath = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', filename)

current_datetime = datetime.datetime.now()
date_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

print(date_string)
with open(filepath, "w") as f:
    f.write(date_string)