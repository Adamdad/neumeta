import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def load_cka_tensor(file_path):
    # Load the tensor from a file
    tensor = torch.load(file_path)
    tensor.fill_diagonal_(1)
    return tensor

def plot_heatmap(tensor, name, vmin=None, vmax=None):
    # Convert tensor to numpy array for plotting
    data = tensor.numpy()
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

    # Plotting the heatmap
    plt.imshow(data, cmap='plasma',interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks(range(7), [64 - (i*8) for i in range(7)])
    plt.yticks(range(7), [64 - (i*8) for i in range(7)])
    # plt.title('Heatmap of the Tensor')
    plt.show()
    # name = '{}.pdf'.format(name)
    # print("Saving figure to {}".format(name))
    # plt.savefig(name)
    
def plot_cka():
    cka_file_path_1 = 'toy/similarity/cka_matrix_individual.pth'
    cka_file_path_2 = 'toy/similarity/cka_matrix.pth'
    cka_file_path_3 = 'toy/similarity/cka_matrix_kd.pth'
    index = torch.tensor(list(range(16, 65, 8))) - 16

    cka_tensor1 = load_cka_tensor(cka_file_path_1)[0]
    cka_tensor2 = load_cka_tensor(cka_file_path_2)[0][index]
    cka_tensor3 = load_cka_tensor(cka_file_path_3)[0]

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Set the font to Times New Roman
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    # matplotlib.rcParams['font.size'] = 12
    plt.figure(figsize=(10, 4))
    # Plotting
    plt.plot(cka_tensor1, label='Individual', color='#5471AB', linewidth=3)
    plt.plot(cka_tensor3, label='KD', color='#6AA66E', linewidth=3)
    plt.plot(cka_tensor2, label='NeuMeta',color='#CF5530', linewidth=3)
    plt.xticks(range(7), [64 - (i*8) for i in range(7)],fontsize=14)
    plt.yticks(fontsize=14)

    # Enhance plot aesthetics
    # plt.title('CKA Analysis')
    plt.xlabel('Model Dimension',fontsize=26)
    plt.ylabel('CKA Value',fontsize=28)
    plt.legend(fontsize=20, ncols=3)
    plt.tight_layout()

    # plt.show()
    plt.savefig('cifar10_cka_analysis.pdf', dpi=300)

def main():
    # Path to the tensor file (replace with your file path)
    cka_file_path = 'toy/similarity/cka_matrix_individual.pth'
    # kl_file_path = 'toy/similarity/kl_matrix_kd.pth'
    # kl_file_path = 'toy/similarity/kl_matrix_individual.pth'
    # kl_tensor = torch.load(kl_file_path)
    
    
    # kl_file_path = 'toy/similarity/kl_matrix.pth'
    # kl_tensor = torch.load(kl_file_path)
    # index = torch.tensor(list(range(16,65,8))) - 16
    # kl_tensor = kl_tensor[index][:,index]
    # Load the tensor
    cka_tensor = load_cka_tensor(cka_file_path)

    # Plot the heatmap
    plot_heatmap(cka_tensor, 'KD_CKA', vmin=0.6, vmax=1)
    # plot_heatmap(kl_tensor, 'Individual', 0, 0.7)

if __name__ == "__main__":
    # main()
    plot_cka()
