import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

class Excel2Fig:
    def __init__(self, input_filename, output_dir):
        _, ext = os.path.splitext(input_filename)
        if ext == ".xlsx":
            self.df = pd.read_excel(input_filename)
        elif ext == ".csv":
            self.df = pd.read_csv(input_filename)

        self.remove_empty_rows()
        
        self.output_file_basename = os.path.splitext(os.path.basename(input_filename))[0].split("_", 1)[1]
        print(self.output_file_basename)
        print(os.getcwd())
        self.output_dir = output_dir

    def remove_empty_rows(self):
        c = self.df.test_F_mean
        cc = c.map(np.isnan).map(np.bitwise_not)
        self.df = self.df[cc]

    def repeat(self, X):
        # X = "N"
        measures = ["balancedAccuracy", "F", "MCC", "AUC"]
        y_labels = ["balanced accuracy", "F-measure", "MCC", "AUC"]
        for index, (measure, y_label) in enumerate(zip(measures, y_labels)):
            self.make_fig(index, X, measure, y_label)

    def make_fig(self, index, X, measure, y_label):
        fig = plt.figure(figsize=(11.69, 8.27), dpi=200)

        labels = eval("self.df." + X)
        m = eval("self.df.test_" + measure + "_mean")
        se = eval("self.df.test_" + measure + "_SE")

        x = np.arange(len(labels))
        plt.bar(x, m,  yerr = se, capsize=0.5, tick_label=labels, width=0.5, error_kw={"elinewidth": 0.1})
        plt.ylabel(y_label, fontsize = 12)

        plt.ylim(0, 1.0)
        # for a,b,c in zip(x, m, se):
        #     plt.text(a, b+c+0.02, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)

        plt.xticks(fontsize=6, rotation=60)
        # rotation='vertical')
        # 
        # plt.margins(0.2)
        
        
        # plt.subplots_adjust(bottom=0.25)


        plt.grid(axis="y",ls='--')
        # plt.show()
        fig.savefig(os.path.join(self.output_dir, "img_" + self.output_file_basename + "_" + measure + ".pdf"))

if __name__  == "__main__":
    # top_dir = os.path.join("~", "OneDrive",
    #         "work", "methyl", "source_data", "cgi_methyl_fgo_blastocyst-maternal_unmethyl-pos")
    # input_filename = os.path.join(top_dir, "exp_aug.xlsx")
    # excel2fig = Excel2Fig(input_filename)
    # excel2fig.repeat('N')

    # Specify a path to input tabular file. 
    excel2fig = Excel2Fig(sys.argv[1], sys.argv[2])

    # Specify a column name of the above tabular file, the values in which specifies the points on the X-axis.  
    # For example, in exp_aug.xlsx, we use 'N' (degree of augmentation). 
    excel2fig.repeat(sys.argv[3])


