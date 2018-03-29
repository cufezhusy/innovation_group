import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

# ===================================================================================
# Visualization the model
# ===================================================================================
def animation_train_and_test(Y_train,Y_test, predict_y_train,predict_y_test):
    # Configure the training plot
    fig = plt.figure(figsize=(16, 8))
    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)
    Y_train_abnormal =Y_train[Y_train == 1]
    Y_train_normal = Y_train[Y_train == 0]
    cut = Y_train_abnormal.shape[0]/Y_train.shape[0]
    line_1_0 = axes1.plot(np.array([0,1]),np.array([0.5,0.5]),'k-')
    line_1_00 = axes1.plot(np.array([0.5, 0.5]), np.array([0, 1]), 'k-')
    line_1_1, = axes1.plot(np.linspace(0,cut,Y_train_abnormal.shape[0]), np.ones(Y_train_abnormal.shape)*0.5, 'ro')
    line_1_2, = axes1.plot(np.linspace(cut,1,Y_train_normal.shape[0]), np.ones(Y_train_normal.shape)*0.5, 'bo')
    axes1.set_title('Train Set')
    axes1.set_ylabel('Model predict')
    axes1.get_xaxis().set_ticks([])
    axes1.set_xlabel('True Answer')

    Y_test_abnormal = Y_test[Y_test == 1]
    Y_test_normal = Y_test[Y_test == 0]
    cut = Y_test_abnormal.shape[0] / Y_test.shape[0]
    line_1_0 = axes2.plot(np.array([0, 1]), np.array([0.5, 0.5]), 'k-')
    line_1_00 = axes2.plot(np.array([0.5, 0.5]), np.array([0, 1]), 'k-')
    line_2_1, = axes2.plot(np.linspace(0, cut, Y_test_abnormal.shape[0]), np.ones(Y_test_abnormal.shape) * 0.5, 'ro')
    line_2_2, = axes2.plot(np.linspace(cut, 1, Y_test_normal.shape[0]), np.ones(Y_test_normal.shape) * 0.5, 'bo')
    axes2.set_title('Test Set')
    axes2.set_ylabel('Model predict')
    axes2.get_xaxis().set_ticks([])
    axes2.set_xlabel('True Answer')


    axes1.set_xlim([0, 1])
    axes2.set_xlim([0, 1])
    axes1.set_ylim([0, 1])
    axes2.set_ylim([0, 1])

    def corr_plt(data):
        train_data, test_data = data
        data_abnormal = train_data[Y_train == 1]
        #line_1_1.set_xdata(1-data_abnormal)
        line_1_1.set_ydata(data_abnormal)

        data_normal = train_data[Y_train == 0]
        #line_1_2.set_xdata(1 - data_normal)
        line_1_2.set_ydata(data_normal)

        data_abnormal = test_data[Y_test == 1]
        # line_1_1.set_xdata(1-data_abnormal)
        line_2_1.set_ydata(data_abnormal)

        data_normal = test_data[Y_test == 0]
        # line_1_2.set_xdata(1 - data_normal)
        line_2_2.set_ydata(data_normal)
        return line_2_1,

    all_data = [(x,y) for (x,y) in zip(predict_y_train,predict_y_test)]
    ani = animation.FuncAnimation(fig, corr_plt, all_data, interval=1000)
    #ani2 = animation.FuncAnimation(fig, corr_plt2, predict_y_test, interval=1000)
    ani.save(filename="training.mp4",writer=writer)
    #ani2.save(filename="testing.mp4", writer=writer)
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

def plot_problem(X,Y,idx):
    plt.plot(X[idx, :, 1, 0], 'b*')
    plt.plot(X[idx, :, 0, 0], 'r*-')
    plt.show()
    print(Y[idx])

if __name__ == '__main__':
    import random
    N = 100
    M = 50

    Y_train = np.random.randint(0,2,size=(N,1))
    Y_test =  np.random.randint(0,2,size=(M,1))

    #get_heat_map(Y_train,np.random.random(size=(N,1)))
    #plt.show()
    predict_y_train = []
    predict_y_test = []

    for i in range(50):
        predict_y_train.append(np.random.random(size=(N,1)))
        predict_y_test.append(np.random.random(size=(M, 1)))

    animation_train_and_test(Y_train,Y_test,predict_y_train,predict_y_test)