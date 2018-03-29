
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ===================================================================================
# Visualization the model
# ===================================================================================
def animation_train_and_test(Y_train,Y_test, predict_y_train,predict_y_test):
    # Configure the training plot
    fig = plt.figure()
    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)
    Y_train_abnormal =Y_train[Y_train == 1]
    Y_train_normal = Y_train[Y_train == 0]
    axes1.imshow([], interpolation='nearest')
    line_1_1, = axes1.plot(np.ones(Y_train_abnormal.shape)*0.5, np.zeros(Y_train_abnormal.shape), 'ro')
    line_1_2, = axes1.plot(np.ones(Y_train_normal.shape)*0.5, np.zeros(Y_train_normal.shape), 'bo')
    axes1.set_title('Train Set')
    axes1.set_xlabel('OK')
    axes1.set_ylabel('Abnormal')
    line2, = axes2.plot(np.zeros(Y_test.shape), np.zeros(Y_test.shape), 'bo')
    axes2.set_title('Test Set')
    axes2.set_xlabel('OK')
    axes2.set_ylabel('Abnormal')
    axes1.set_xlim([0, 1])
    axes2.set_xlim([0, 1])
    axes1.set_ylim([0, 1])
    axes2.set_ylim([0, 1])

    def corr_plt(data):
        data_abnormal = data[Y_train == 1]
        #line_1_1.set_xdata(1-data_abnormal)
        line_1_1.set_ydata(data_abnormal)

        data_normal = data[Y_train == 0]
        #line_1_2.set_xdata(1 - data_normal)
        line_1_2.set_ydata(data_normal)
        return line_1_1,

    def corr_plt2(data):
        line2.set_xdata(1 - data)
        line2.set_ydata(data)
        return line2,

    ani = animation.FuncAnimation(fig, corr_plt, predict_y_train, interval=1000)
    ani2 = animation.FuncAnimation(fig, corr_plt2, predict_y_test, interval=1000)
    plt.show()


def get_heat_map(actual,predict):
    # Plot it out
    fig, ax = plt.subplots()
    AUC = np.zeros((2,2))

    for s in range(predict.shape[0]):
        auc_score = 1 if predict[s,0] > 0.5 else 0
        AUC[auc_score,int(actual[s,0])] +=1

    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='YlGn', vmin=0.0, vmax=1.0)
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(["Normal","Abnormal"])
    ax.set_yticklabels(["Normal", "Abnormal"])
    #ax.set_yticklabels(["%s - %s"%(i,nex)], minor=False)

    plt.xlabel("Actual")
    plt.ylabel("Model")
    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)
    fig = plt.gcf()

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    from itertools import zip_longest
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

    #
if __name__ == '__main__':
    import random
    N = 100
    M = 50

    Y_train = np.random.randint(0,2,size=(N,1))
    Y_test =  np.random.randint(0,2,size=(M,1))

    get_heat_map(Y_train,np.random.random(size=(N,1)))
    plt.show()
    #predict_y_train = []
    #predict_y_test = []

    #for i in range(10):
        #predict_y_train.append(np.random.random(size=(N,1)))
        #predict_y_test.append(np.random.random(size=(M, 1)))

    #animation_train_and_test(Y_train,Y_test,predict_y_train,predict_y_test)