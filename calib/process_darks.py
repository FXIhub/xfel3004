import sys
import os
import subprocess
import configparser

class ProcessDarks():
    def __init__(self, conf_fname='exp.ini'):
        self.working_dir = None

        config = configparser.ConfigParser()
        config.read(conf_fname)
        self.script_dir = config.get('toolbox', 'script_dir', fallback='')
        self.exp_dir = config.get('toolbox', 'exp_dir')
        #self.detector_str = config.get('toolbox', 'detector_string')
        self.output_dir = config.get('toolbox', 'output_dir', fallback='.')

    def process(self, runs, test=False):
        try:
            assert len(runs) == 3
        except (AssertionError, TypeError):
            print('Need three run numbers')
            raise
        sruns = [str(int(r)) for r in runs]
        command = self.script_dir + 'sbatch_analyse.py '
        command += '--input_dir ' + self.exp_dir + ' '
        command += '--output_dir ' + self.output_dir + ' '
        #command += '--detector_string ' + self.detector_str + ' '
        command += '--type dark --run_type all '
        command += '--run_list ' + ' '.join(sruns) + ' '
        print(command)
        if not test:
            output = subprocess.check_output(command.split())
            self.working_dir = output[output.find(b'sbatch working dir'):].split(b'\n')[0].split()[-1].decode()
            print('Working directory:', self.working_dir)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Dark runs using the AGIPD toolbox')
    parser.add_argument('run_hg', help='High gain run', type=int)
    parser.add_argument('run_mg', help='Medium gain run', type=int)
    parser.add_argument('run_lg', help='Low gain run', type=int)
    parser.add_argument('-c', '--config_file', help='Path to config file. Default: exp.ini', default='exp.ini')
    args = parser.parse_args()
    
    proc = ProcessDarks(conf_fname=args.config_file)
    proc.process([args.run_hg, args.run_mg, args.run_lg])

if __name__ == '__main__':
    main()
