import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def act_function(input):
    return np.sin(2*np.pi*input)

###Function that double checks partitions in respect to mse
def check_pt(prts, cut, mse_1):
    for i in prts:
        if i + mse_1 > cut and i - mse_1 < cut:
            return True
    return False

###Function to count the number of linear regions given the partitions unordered
def count_nr_of_linear_regions(ptrs):
    count = 1
    for i in ptrs:
        if i == 0 or i == 1 or i == 2:
            continue
        count = count + 1
    return count

#Function that checks whether an element is in an array, numpy
def entails(element, array):
    for i in range(0, array.size):
        if array[i] == element:
            return True
    return False

#Function that filters out all the cut_offs in range!
def in_range(input):
    in_range = []
    for i in input:
        if i > input_min and i < input_max:
            in_range.append(i)
    return in_range

#Function that computes papers co and ca and c count. First number is c, then c0 and then ca
def count_regions(input):
    co = 0
    for i in input:
        if i < input_max and i > input_min:
            co = co + 1
    counts = np.zeros(3)
    counts[0] = len(input)+1
    counts[1] = co
    counts[2] = len(input) - co
    return counts

nr_of_linear_regions = np.zeros(300)
for iterator in range(0, 300):
    ###Constants
    #filepath = "Models_1DSIN/LReLU_1D_D4_SINE_V5-model-01.hdf5"
    #filepath = "Models_1D/initReLU_1D_D1_V6.h5"
    filepath = "Models_init_1D/ReLU_1D_D1_V6_"+str(iterator)+".h5"
    input_min = 0.0
    input_max = 1.0
    depth = 1
    width = 16
    data_dim = 1000 #We take 100000 samples between 0 and 1 and compute the function output
    min_data = 0.0
    max_data = 1.0
    mse_percentage = 0.00
    alpha = 0.3
    ###load model and extract weights
    model = keras.models.load_model(filepath)
    weights = np.array(model.get_weights())
    nr_of_neurons = depth*width
    #model.summary()

    ###We compute an mse of a test set
    X_test = np.random.uniform(low= min_data, high = max_data, size = data_dim)
    Y_exact = act_function(X_test)
    Y_predict = model.predict(X_test)
    mse = np.square(np.subtract(Y_exact, Y_predict)).mean()
    mse_1 = mse_percentage*mse #If it is smaller than a percentage of the mse then it was meant as same point


    ###layer 1
    # We iterate over the first layers weights and compute the cuts.
    weight = weights[0]
    weight = weight[0]
    bias = weights[1]
    partitions = []
    ordered_partitions = []
    cut_offs = []
    for i in range(0, len(weight)):
        cut = -bias[i]/weight[i]
        cut_offs.append(cut)
        if cut <= 0 or cut >= 1 or weight[i] == 0 or check_pt(partitions, cut, mse_1):
            if cut <= 0:
                partitions.append(cut)
            elif cut >= 1:
                partitions.append(cut)
            else:
                partitions.append(cut) #will need to change this back. Not priority now
            continue
        partitions.append(cut)

    #print(partitions)   #This works perfectly, every first layer neuron will activate ones in 1D
    #print(count_nr_of_linear_regions(partitions))

    ordered_partitions = partitions.copy()
    layer_1 = partitions.copy()
    ordered_partitions.sort()
    neuron_partitions = np.zeros((16, len(ordered_partitions)+1)) #16 equals number of neurons
    biasses_partitions = np.zeros((16, len(ordered_partitions)+1))

    #If we have more layers we need to store the multiplicative terms and biasses too->easier scallable
    #We compute the activations as well 
    if depth > 1:
        for i in range(0, 16):
            for j in range(0, len(ordered_partitions)+1):
                if j == 0:
                    if weight[i] > 0:
                        neuron_partitions[i][j] = alpha*weight[i]
                        biasses_partitions[i][j] = alpha*bias[i]
                    else:
                        neuron_partitions[i][j] = weight[i]
                        biasses_partitions[i][j] = bias[i]
                    continue
                if partitions[i] <= ordered_partitions[j-1]:
                    if weight[i] > 0:
                        neuron_partitions[i][j] = weight[i]
                        biasses_partitions[i][j] = bias[i]  #store the right biasses, here overkill but for analogueness with following layers
                    else:
                        neuron_partitions[i][j] = alpha*weight[i]
                        biasses_partitions[i][j] = alpha*bias[i]
                else:
                    if weight[i] > 0:
                        neuron_partitions[i][j] = alpha*weight[i]
                        biasses_partitions[i][j] = alpha*bias[i]  #store the right biasses, here overkill but for analogueness with following layers
                    else:
                        neuron_partitions[i][j] = weight[i]
                        biasses_partitions[i][j] = bias[i]

    #print(neuron_partitions) #A lot of optimization possible!
    #print(weight)

    ### Layer 2 or more same scheme for now, for now not at all optimised in order to enhance readability!
    new_partitions = [] ###For layer 2
    old_ordered_partitions = ordered_partitions.copy()
    layer_3 = []
    layer_4 = []

    if depth > 1:
        for i in range(1, depth): #Loop over all layers now assume 2
            w = weights[2*i]
            b = weights[2*i+1]
            old_regions_weights = np.zeros((16, len(ordered_partitions)+1))
            old_region_biasses = np.zeros((16, len(ordered_partitions)+1))
            region_cut_offs = np.zeros((16, len(ordered_partitions)+1))

            #print(neuron_partitions)
            ### Compute the cut off for every neuron and add to new_partitions
            for j in range(0, len(b)): #loop over every neuron
                for pt in range(0, len(ordered_partitions)+1): #loop over possibly every partition.
                    a = 0.0 #this will give us the multiplicative term in each partition after following loop
                    b_n = 0.0 #Bias in each partition of each neuron
                    for neuron in range(0, len(b)): #loop to get output multiplicative of every neuron in every partition and x direction of weight matrix 
                        a0 = neuron_partitions[neuron][pt]   # The multiplicative term of that neuron output before in that partition
                        w_jneuron = w[neuron][j]
                        a = a + a0*w_jneuron
                        b0 = biasses_partitions[neuron][pt]
                        b_n = b_n + b0*w_jneuron
                    b_n = b_n + b[j]

                    #Store a and b for next layer
                    old_regions_weights[j][pt] = a
                    old_region_biasses[j][pt] = b_n

                    #Now, we have got a and b -> cut off is easy now!
                    cut_off = -b_n/a
                    
                    under = -1000
                    if pt > 0:
                        under = ordered_partitions[pt-1]

                    if pt < len(ordered_partitions):
                        above = ordered_partitions[pt]
                    if pt == len(ordered_partitions):
                        above = 10000

                    #if cut_off < ordered_partitions[pt] and cut_off > under:
                    if cut_off > under and cut_off < above:
                        if i == 1:
                            new_partitions.append(cut_off)
                        elif i == 2:
                            layer_3.append(cut_off)
                        elif i == 3:
                            layer_4.append(cut_off)
                        partitions.append(cut_off)
                        region_cut_offs[j][pt] = cut_off
            if depth > i+1:
                ###rearange everything for next layer! We use biasses partitions and neuron/weights partitions and partitions itself!
                ordered_partitions = partitions.copy()
                ordered_partitions.sort()

                neuron_partitions = np.zeros((16, len(ordered_partitions)+1))
                biasses_partitions = np.zeros((16, len(ordered_partitions)+1))
                for i in range(0, 16): #loop over all neurons in that layer
                    index_old = 0
                    index_part_old = 0
                    for j in range(0, len(ordered_partitions)+1): #loop over all the partitions, new and old!
                        if j == 0: #The outer left region, doesn't matter old or new one!
                                if old_regions_weights[i][0] > 0: 
                                    neuron_partitions[i][0] = alpha*old_regions_weights[i][0]
                                    biasses_partitions[i][0] = alpha*old_region_biasses[i][0]
                                else:
                                    neuron_partitions[i][0] = old_regions_weights[i][0]
                                    biasses_partitions[i][0] = old_region_biasses[i][0]
                                continue
                        if ordered_partitions[j-1] == region_cut_offs[i][index_old]: #It's the new cut itself!
                            if old_regions_weights[i][index_old] > 0:
                                neuron_partitions[i][j] = old_regions_weights[i][index_old]
                                biasses_partitions[i][j] = old_region_biasses[i][index_old]
                            else:
                                neuron_partitions[i][j] = alpha*old_regions_weights[i][index_old]
                                biasses_partitions[i][j] = alpha*old_region_biasses[i][index_old]
                            continue
                        if old_ordered_partitions[index_part_old] != ordered_partitions[j-1]: #New cut
                            if old_regions_weights[i][index_old]*ordered_partitions[j-1] + old_region_biasses[i][index_old] >= 0:
                                neuron_partitions[i][j] = old_regions_weights[i][index_old]
                                biasses_partitions[i][j] = old_region_biasses[i][index_old]  
                            else:
                                neuron_partitions[i][j] = alpha*old_regions_weights[i][index_old]
                                biasses_partitions[i][j] = alpha*old_region_biasses[i][index_old]
                            continue
                        else:                                                           #old cut 
                            index_old = index_old + 1
                            if old_regions_weights[i][index_old]*ordered_partitions[j-1] + old_region_biasses[i][index_old] >= 0:
                                neuron_partitions[i][j] = old_regions_weights[i][index_old]
                                biasses_partitions[i][j] = old_region_biasses[i][index_old]
                            else:
                                neuron_partitions[i][j] = alpha*old_regions_weights[i][index_old]
                                biasses_partitions[i][j] = alpha*old_region_biasses[i][index_old]
                            if index_part_old < len(old_ordered_partitions)-1:
                                index_part_old = index_part_old + 1
                old_ordered_partitions = ordered_partitions.copy()

    #print(new_partitions)          

    linear_regions = count_regions(partitions)
    nr_of_linear_regions[iterator] = linear_regions[1]

print(nr_of_linear_regions)
print(np.mean(nr_of_linear_regions))
print(np.var(nr_of_linear_regions))
