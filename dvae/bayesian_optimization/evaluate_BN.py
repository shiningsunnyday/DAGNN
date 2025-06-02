import numpy as np
import os

class Eval_BN(object):
    def __init__(self, save_dir, R_script='compute_score.R'):
        self.save_dir = save_dir
        self.R_script = R_script

    def eval(self, input_string):
        input_matrix = np.array([int(x) for x in input_string.split()]).reshape(8, 8)
        tmp_file = os.path.join(self.save_dir, 'temp_BN_matrix')
        np.savetxt(tmp_file, input_matrix)
        return self.compute_score(tmp_file)

    def compute_score(self, input_file, output_file=None):
        if output_file is None:
            output_file = input_file + "_score"
        input_file = os.path.abspath(input_file)
        output_file = os.path.abspath(output_file)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        cmd = (
            'bash -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate cktgnn_r && cd {} && Rscript compute_score.R {} {}"'
        ).format(current_dir, input_file, output_file)
        # cmd = 'cd ~/induction/dagnn/dvae/bayesian_optimization && Rscript compute_score.R ' + input_file + ' ' + output_file
        os.system(cmd)
        score = np.loadtxt(output_file, ndmin=1)
        print(score)
        return float(score[0])
