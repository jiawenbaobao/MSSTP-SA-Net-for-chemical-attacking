import os
import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def correlation_coefficient(y_pred, y_target):
    '''
    caculate the correlation coefficient
    '''
    return stats.pearsonr(y_pred, y_target)[0]

def compute_RMSE(y_pred, y_target):
    '''
    caculate the root mean squared error
    :param y_pred:
    :param y_target:
    :return:
    '''
    with torch.no_grad():
        RMSE_value = np.sqrt((((y_pred - y_target) ** 2).mean()))
    return RMSE_value

def FB(y_pred, y_target):
    '''
    fractional bias
    :param y_pred:
    :param y_target:
    :return:
    '''
    with torch.no_grad():
        mean_y_pred = y_pred.mean()
        mean_y_target = y_target.mean()
        FB = (mean_y_pred - mean_y_target)/ (2 * (mean_y_target + mean_y_pred))
    return FB

def accuracy(y_pred, y_target):
    y_pred_class = list(map(set_class, y_pred))
    y_target_class = list(map(set_class, y_target))
    correct = np.sum(np.array(y_pred_class) == np.array(y_target_class))
    return correct/ len(y_pred)

def set_class(x):
    if x <= 0.0189:  # safe area
        x = 0
    elif 0.0189 < x <= 0.5:    # reaction area
        x = 1
    elif 0.5 < x <= 171.5:  # light injury area
        x = 2
    elif 171.5< x <=933.9:   # dangerous area
        x = 3
    elif x > 933.9: # dead area
        x = 4
    return x

def plot_performance(train_loss, val_loss, save_dir):
    # Figure size
    fig = plt.figure()

    # Plot Loss
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title('train loss')
    ax1.plot(train_loss, label="train")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('val loss')
    ax2.plot(val_loss, label="val")

    # Save figure
    plt.savefig(os.path.join(save_dir, "performance.png"))

    # Show plots
    plt.show()


def fit_performance(y_pred, y_target):

    # plt.subplot(1, 1, 1)
    # plt.title("Concentration(ppm)")

    l = LinearRegression()
    l.fit(y_pred, y_target)
    w = np.around(l.coef_[0][0], 3)
    b = np.around(l.intercept_[0], 3)

    line = l.predict(y_pred.reshape(-1,1))
    return w, line

    # plt.xlabel('Model output')
    # plt.ylabel('CFD output')
    # plt.xlim(-1000, 10000)
    # plt.ylim(-1000, 10000)
    # plt.scatter(y_pred, y_target,label='Adjusted data')
    # plt.plot(y_pred, line, c='r', label='Fit: y='+ str(w)+'x+')


    # plt.legend(loc='lower right')

    # plt.savefig(os.path.join(save_dir, "test_performance.png"))

    # Show plots
    # plt.show()

def main():
    # MaxAbs normalization
    x1 = torch.randn(3,1580)
    x2 = torch.randn(3,1580)
    # print(FB(x1,x2))
    fit_performance(x1, x2, rmse=0, r=0.7, save_dir='222')

if __name__ == '__main__':
    main()