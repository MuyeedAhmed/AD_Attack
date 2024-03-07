import matplotlib.pyplot as plt
import numpy as np



def NDSourceExample(attack):
    np.random.seed(42) 
    x_coordinates = np.random.rand(12) * 10
    y_coordinates = np.random.rand(12) * 10  
    
    np.random.seed(3)
    all_indices = list(range(12))
    np.random.shuffle(all_indices)
    
    if attack == "Restart":
        x_indices = all_indices[:5]
        o_indices = all_indices[4:9]
        label2 = 'Points Selected in Run 1'
        label3 = 'Points Selected in Run 2'
    elif attack == "Resource":
        x_indices = all_indices[:5]
        o_indices = all_indices[4:6]
        label2 = 'Points Selected (max_samples=5)'
        label3 = 'Points Selected (max_samples=2)'
    
        
    # x_indices = [0, 3, 6, 9]
    x_x = [x_coordinates[i] for i in x_indices]
    x_y = [y_coordinates[i] for i in x_indices]
    
    # o_indices = [1, 4, 7, 10]
    o_x = [x_coordinates[i] for i in o_indices]
    o_y = [y_coordinates[i] for i in o_indices]
    
    
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [7, 4]
    plt.scatter(x_coordinates, y_coordinates, marker='.', label='Unpicked Points', color="black", s=100)
    plt.scatter(x_x, x_y, marker='v', label=label2, color="orange", s=150)
    plt.scatter(o_x, o_y, marker='^', label=label3, color="blue", s=150)
    
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    
    plt.legend(fontsize=12)
    
    plt.grid()
    plt.savefig("Fig/NDSource/"+attack+"_Cause.pdf", bbox_inches='tight')
    
    plt.show()

NDSourceExample("Restart")

NDSourceExample("Resource")
