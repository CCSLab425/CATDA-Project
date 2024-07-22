import pandas as pd


def train_history():
    train_hist = {}

    train_hist['src_domain_loss'] = []
    train_hist['tgt_domain_loss'] = []
    train_hist['total_loss'] = []
    train_hist['Source_test_acc'] = []
    train_hist['Source_test_loss'] = []
    train_hist['Target_test_acc'] = []
    train_hist['Target_test_loss'] = []

    return train_hist


def DDC_train_history():
    train_hist = {}

    train_hist['src_and_tgt_mmd_loss'] = []
    train_hist['yhmmd_loss'] = []
    train_hist['total_loss'] = []
    train_hist['Source_test_acc'] = []
    train_hist['Source_test_loss'] = []
    train_hist['Target_test_acc'] = []
    train_hist['Target_test_loss'] = []

    return train_hist

def save_his(train_hist={}, save_dir='save_dir/', save_name=''):
    """
    save history data
    """
    data_df = pd.DataFrame(train_hist)
    data_df.to_csv(save_dir + save_name + '.csv')


