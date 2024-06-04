import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu index")
    parser.add_argument("--pid",    type=int, default=0, help="process index")
    parser.add_argument("--n_p",    type=int, default=1, help="number of processes")

    args = parser.parse_args()
    return args