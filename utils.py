import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from FrEIA.utils import force_to
import torch.distributions as D
import os
import tqdm

#Train an INN
def train(INN,p_data:Callable,device:str,lr:float,milestones:list,gamma:float,batch_size:int,n_batches:int,experiment_name:str,save_freq:int):
    """
    parameters:
        INN:                Normalizing flow to train
        p_data:             Function to get samples followikng the target distribution
        lr:                 Learning rate
        milestones:         Milestones for learning rate decay
        gamma:              Factor for learning rate decay
        batch_size:         Bacth size
        n_batches:          Number of batches
        experiment_name:    Name of the training run
        save_freq:          Frequency of saving the state dict
    """


    INN.train()
    
    folder = "./"+experiment_name
    os.mkdir(folder)

    #Latent distribution of the model
    p_0 = force_to(D.MultivariateNormal(torch.zeros(2),torch.eye(2)),device)
    
    #Initialize the optimzzer and the lr scheduler
    optimizer = torch.optim.Adam(INN.parameters(),lr = lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=milestones,gamma =gamma)

    #Storage
    loss_storage = torch.zeros(n_batches)
    jacobian_storage = torch.zeros([n_batches,2])

    #Train the model
    for i in tqdm.tqdm(range(n_batches)):
        
        #Get training data
        x = p_data.sample(N = batch_size).to(device)

        #Compute the loss
        z,jac = INN(x)
        nll = - (p_0.log_prob(z) + jac).mean()

        #Optimze
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()

        lr_scheduler.step()

        #Store the results
        loss_storage[i] = nll.item()
        jacobian_storage[i][0] = jac.mean()
        jacobian_storage[i][1] = jac.std().item()

        #Store the model
        if ((i+1) % save_freq) == 0:
            torch.save(INN.state_dict(),folder + f"/state-dict_iteration-{i+1}.pt")

    
    #Save the recorded data
    np.savetxt(folder +"/loss.txt",loss_storage.cpu().detach().numpy())
    np.savetxt(folder +"/jac.txt",jacobian_storage.cpu().detach().numpy())

#Create coordinate grid
def get_grid_n_dim(res_list:list,lim_list:list):

    '''
    parameters:
        res_list: List of intes containing the resolutions along the different dimensions
        lim_list: Lists of lists contaiing the limits of the gird along the different dimensions

    returns:
        grid_points:            Tensor of shape (N,d) containing the points on the grid
        spacings_tensor:        List containing the distance between points for each dimension
        coordinate_grids_list:  List containing the coordinate grids for each dimension
    '''

    d = len(res_list)

    #get ranges for the different dimensions
    range_list = [torch.linspace(lim_list[i][0],lim_list[i][1],res_list[i]) for i in range(d)]

    #Get the spacings between two points
    spacings_tensor = torch.zeros(d)

    for i in range(d):
        spacings_tensor[i] = range_list[i][1] - range_list[i][0]

    #Get grids for the different dimensions
    coordinate_grids = torch.meshgrid(range_list,indexing="xy")

    #Combine the grids
    coordinate_grids_list = []

    for i in range(d):
        coordinate_grids_list.append(coordinate_grids[i].reshape(-1,1))

    grid_points = torch.cat(coordinate_grids_list,-1)

    return grid_points,spacings_tensor,coordinate_grids

#Evaluate pdf on a grid
def eval_pdf_on_grid_2D(pdf:Callable,x_lims:list = [-10.0,10.0],y_lims:list = [-10.0,10.0],x_res:int = 200,y_res:int = 200):
    """
    parameters:
        pdf:    Probability density function
        x_lims: Limits of the evaluated region in x directions
        y_lims: Limits of the evaluated region in y directions
        x_res:  Number of grid points in x direction
        y_res:  Number of grid points in y direction

    returns:
        pdf_grid:   Grid of ppdf values
        x_grid:     Grid of x coordinates
        y_grid:     Grid of y coordinates
    """

    grid_points,spacings_tensor,coordinate_grids = get_grid_n_dim(res_list = [x_res,y_res],lim_list = [x_lims,y_lims])

    #Evaluat pdf
    pdf_grid = pdf(grid_points).reshape(y_res,x_res)

    x_grid = coordinate_grids[0]
    y_grid = coordinate_grids[1]

    return pdf_grid,x_grid,y_grid

#visualize the pdf 
def plot_pdf_2D(pdf_grid:torch.tensor,x_grid:torch.tensor,y_grid:torch.tensor,ax:plt.axes,fs:int = 20,title:str = ""):
    """
    parameters:
        pdf_grid:   Grid of pdf values
        x_grid:     Grid of x coordinates
        y_grid:     Grid of y coordinates
        ax:         Axes for plotting
        fs:         Fontsize
        title:      Title of the plot
    """

    ax.pcolormesh(x_grid,y_grid,pdf_grid)
    ax.set_xlabel("x",fontsize = fs)
    ax.set_ylabel("y",fontsize = fs)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.set_title(title,fontsize = fs)
    plt.tight_layout()

#Data distribution
class GMM():
    def __init__(self,means:torch.tensor,covs:torch.tensor,weights:torch.tensor = None)->None:
        """
        parameters:
            means: Tensor of shape [M,d] containing the locations of the gaussina modes
            covs: Tensor of shape [M,d,d] containing the covariance matrices of the gaussina modes
            weights: Tensor of shape [M] containing the weights of the gaussian modes. Use equal weights if not specified
        """

        #get diensionality of the data set
        self.d = len(means[0])

        #Get the number of modes
        self.M = len(means)
        self.mode_list = []

        #Check weights
        if weights is None:
            self.weights = torch.ones(self.M) / self.M
        else:
            self.weights = weights

        if self.weights.sum() != 1.0: raise ValueError()

        #Initialize the normal modes
        for i in range(self.M):
            self.mode_list.append(D.MultivariateNormal(loc = means[i],covariance_matrix = covs[i]))

    def __call__(self,x:torch.tensor)->torch.tensor:
        """
        Evaluate the pdf of the model.

        parameters:
            x: Tensor of shape [N,d] containing the evaluation points

        returns:
            p: Tensor of shape [N] contaiing the pdf value for the evaluation points
        """

        p = torch.zeros(len(x))

        for i in range(self.M):
            p += self.mode_list[i].log_prob(x).exp() * self.weights[i]
        
        return p
    
    def log_prob(self,x)->torch.tensor:
        """
        Evaluate the log pdf of the model.

        parameters:
            x: Tensor of shape [N,d] containing the evaluation points

        returns:
            log_p: Tensor of shape [N] contaiing the log pdf value for the evaluation points
        """

        log_p = self.__call__(x).log()

        return log_p
    
    def sample(self,N:int)->torch.tensor:
        """
        Generate samples following the distribution

        parameters:
            N: Number of samples

        return:
            s: Tensor of shape [N,d] containing the generated samples
        """
        i = np.random.choice(a = self.M,size = (N,),p = self.weights)
        u,c = np.unique(i,return_counts=True)

        s = torch.zeros([0,self.d])

        for i in range(self.M):
            s_i = self.mode_list[u[i]].sample([c[i]])
            s = torch.cat((s,s_i),dim = 0)

        return s