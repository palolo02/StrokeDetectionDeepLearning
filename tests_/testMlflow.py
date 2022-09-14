import mlflow
from sqlalchemy import true
import torch
from main import LEARNING_RATE
from deepLearningModels.unet import UNET, test
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://localhost:5000")

def testMlfow():    
    print("starting MLFLOW")
    with mlflow.start_run():
        LEARNING_RATE = 0.0001
        mlflow.log_param("learningRate", LEARNING_RATE)
        # for epoch in range(0, 3):
        #     mlflow.log_metric(key="quality", value=2*epoch, step=epoch)
        #     model = UNET(in_channels=3, out_channels=1)
        #     mlflow.pytorch.log_model(model, "UNET", registered_model_name="unet")
        #     print(model._get_name())
        #     print(model.__class__)
        #     #LEARNING_RATE += 0.001

# ============= Runing Mlflow server ===================
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host localhost

def testGenerators():
    
    def return_subset_data(x, y, n, iterations):
        start_offset = 0
        end_offset = n        
        # Start reading elements       
        for _ in range(iterations):
            yield _, x[start_offset:end_offset,:,:], y[start_offset:end_offset,:,:]
            start_offset += n
            end_offset += n

    x = torch.zeros((160,128,128))
    y = torch.ones((160,128,128))
    for i, x, y in return_subset_data(x,y,32,int(160/32)):        
        print(x.shape)
        print(y.shape)
        print(i)

def plotTest():
    values = torch.rand(10)
    name = "example"
    #title = f"{name} per epochs - {values[-1]:.2f} last element"
    title = f"{name} per epochs"
    print(values)
    fig, ax = plt.subplots()
    ax.plot(range(1,len(values)+1), values)
    ax.set(xlabel='epochs', ylabel=name,title=title)    
    plt.xticks(range(1, len(values)+1,1))
    ax.set_ylim(ymin=0)
    ax.xaxis.label.set_color('gray')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('gray')          #setting up Y-axis label color to blue
    ax.tick_params(axis='x', colors='gray')    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='gray')  #setting up Y-axis tick color to black
    ax.spines['left'].set_color('gray')        # setting up Y-axis tick color to red
    ax.spines['top'].set_color('gray')  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.annotate(f"{values[-1]:.2f}",(10,values[-1]), color='black', size=8)   
    #plt.tick_params(top='off', bottom='on', left='off', right='off', labelleft='on', labelbottom='on')
    plt.savefig(f'plots/test.png', format="png")
    plt.close(fig)

if __name__=="__main__":
    #testGenerators()
    testMlfow()