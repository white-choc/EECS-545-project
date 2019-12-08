import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

def read_txt(file_name):
    df = pd.read_csv(file_name, header=None, sep=' ')
    mat = df.to_numpy(dtype=float)
    return mat

def plot_traj(base, truth, pred, idx, avg_error, final_error):
    plt.figure()
    plt.scatter(base[:, 2], base[:, 3], c='k', marker='*',label='base traj')
    plt.scatter(truth[:, 2], truth[:, 3], c='k', label='truth')
    plt.scatter(pred[:, 2], pred[:, 3], c='r', label='prediction')
    plt.title('average error = {avg_err}, \nfinal error = {final_err}'.format(avg_err=avg_error, final_err=final_error))
    plt.axis('equal')
    plt.legend()
    plt.savefig('plots/fig{}'.format(idx))
    # plt.show()


def calc_avg_error(truth, pred):
    error = np.linalg.norm(truth[:, 2:] - pred[:, 2:], axis=1, ord=2)
    return np.average(error)

def calc_final_error(truth, pred):
    return np.linalg.norm(truth[-1, 2:] - pred[-1, 2:])

def calc_avg_error_sgan(truth, pred):
    error = np.linalg.norm(truth[:, 2:] - pred[:, :], axis=1, ord=2)
    return np.average(error)

def calc_final_error_sgan(truth, pred):
    return np.linalg.norm(truth[-1, 2:] - pred[-1, :])

def get_sgan_data():
    file_name = 'gate8_picked.txt'
    mat = read_txt(file_name)
    num_traj = int(mat.shape[0] / 4)
    true_traj = [None] * num_traj
    pred_traj = [None] * num_traj
    for i in range(num_traj):
        true_traj[i] = np.array([mat[i*4, :-1], mat[i*4+1, :-1]])
        pred_traj[i] = np.array([mat[i*4+2, :-1], mat[i*4+3, :-1]])
        # plot(true_traj[i][:, :8], true_traj[i][:, 8:], pred_traj[i][:, 8:])
    return pred_traj

def plot_sgan_lstm(base, truth, sgan, lstm, avg_error_sgan, final_error_sgan, avg_error_lstm, final_error_lstm, idx):
    fig, axs = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.4)
    axs[0].scatter(base[:, 2], base[:, 3], c='k', marker='*',label='base traj')
    axs[0].scatter(truth[:, 2], truth[:, 3], c='k', label='truth')
    axs[0].scatter(sgan[:, 0], sgan[:, 1], c='r', label='prediction')
    axs[0].set_title('average error = {avg_err}, \nfinal error = {final_err}'.format(avg_err=avg_error_sgan, final_err=final_error_sgan))
    axs[0].set_ylabel('sgan')
    axs[0].axis('equal')
    axs[0].legend()
    axs[1].scatter(base[:, 2], base[:, 3], c='k', marker='*',label='base traj')
    axs[1].scatter(truth[:, 2], truth[:, 3], c='k', label='truth')
    axs[1].scatter(lstm[:, 2], lstm[:, 3], c='r', label='prediction')
    axs[1].set_title('average error = {avg_err}, \nfinal error = {final_err}'.format(avg_err=avg_error_lstm, final_err=final_error_lstm))
    axs[1].set_ylabel('social lstm')
    axs[1].axis('equal')
    axs[1].legend()
    fig.savefig('plots/fig{}'.format(idx))


if __name__ == '__main__':
    pred_file = 'gates_8.txt'
    truth_file = 'truth_gates_8.txt'
    pred_mat = read_txt(pred_file)
    truth_mat = read_txt(truth_file)
    row_length = truth_mat.shape[0]
    traj_length = 20
    num_traj = int((row_length + 1) / traj_length)
    base_traj = [None] * num_traj
    truth_traj = [None] * num_traj
    pred_traj = [None] * num_traj
    for i in range(num_traj):
        tmp_base_traj = np.zeros((8, truth_mat.shape[1]))
        tmp_truth_traj = np.zeros((12, truth_mat.shape[1]))
        tmp_pred_traj = np.zeros((12, truth_mat.shape[1]))
        for j in range(8):
            tmp_base_traj[j, :] = truth_mat[i*traj_length + j]
        for j in range(12):
            tmp_pred_traj[j, :] = pred_mat[i*traj_length + j + 8]
            tmp_truth_traj[j, :] = truth_mat[i*traj_length + j + 8]
        base_traj[i] = tmp_base_traj
        pred_traj[i] = tmp_pred_traj
        truth_traj[i] = tmp_truth_traj
    idx_set = [82, 83, 45, 47, 80, 71, 72, 73, 28, 14, 15, 23, 59, 0, 1, 10, 34, 11]
    avg_error = [None] * len(idx_set)
    final_error = [None] * len(idx_set)
    avg_error_sgan = [None] * len(idx_set)
    final_error_sgan = [None] * len(idx_set)
    #plot the figures and calculate the errors
    sgan_pred = get_sgan_data()
    for i in range(len(idx_set)):
        idx = idx_set[i]
        base = base_traj[idx]
        truth = truth_traj[idx]
        pred = pred_traj[idx]
        avg_error[i] = calc_avg_error(truth, pred)
        final_error[i] = calc_final_error(truth, pred)
        sgan = sgan_pred[i]
        sgan = sgan[:, 8:].T
        avg_error_sgan[i] = calc_avg_error_sgan(truth, sgan)
        final_error_sgan[i] = calc_final_error_sgan(truth, sgan)
        # plot_traj(base, truth, pred, idx, avg_error[i], final_error[i])
        plot_sgan_lstm(base, truth, sgan, pred, avg_error_sgan[i], final_error_sgan[i], avg_error[i], final_error[i], i)
    print('average error for sgan is {}'.format(np.mean(avg_error_sgan)))
    print('final error for sgan is {}'.format(np.mean(final_error_sgan)))
    print('average error for lstm is {}'.format(np.mean(avg_error)))
    print('final error for lstm is {}'.format(np.mean(final_error)))
    # np.savetxt('average error', avg_error, delimiter=' ')
    # np.savetxt('final error', final_error, delimiter=' ')
