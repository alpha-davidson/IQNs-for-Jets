""" 
Converts a text file output by the C++ code into a .npy file
"""

import click
import numpy as np


@click.command()
@click.option('--filename', default="mlData", help='name of file to fix')

def main(filename):
    dataInitial=np.loadtxt(filename + ".txt")
    np.save(filename + ".npy", dataInitial)
    
if(__name__=="__main__"):
    main()
