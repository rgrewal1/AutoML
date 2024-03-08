import subprocess

program_list = ['tpotrun.py', 'autosklearn_run.py' , 'autogluon_run.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)