import argparse



def get_autoformer_argsparse(add_help=True) :
    parser = argparse.ArgumentParser(description="Pytorch AutoFormer", add_help=add_help)
    return parser