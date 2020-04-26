import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#True Global constants
semi_visualization = 0
emp_aver = np.zeros((3,3))

E_regionsV1 = np.zeros((3,3))
D_entrV1 = np.zeros((3,3))

E_regionsV2 = np.zeros((3,3))
D_entrV2 = np.zeros((3,3))

E_regionsV3 = np.zeros((3,3))
D_entrV3 = np.zeros((3,3))

E_regionsV4 = np.zeros((3,3))
D_entrV4 = np.zeros((3,3))

E_regionsV5 = np.zeros((3,3))
D_entrV5 = np.zeros((3,3))

E_regionsV6 = np.zeros((3,3))
D_entrV6 = np.zeros((3,3))

for iteration in range(0, 3):
    #Semi Global constants
    visualisation_histogramms = 0
    empirical_average = np.zeros(3)
    Eregions = np.zeros(3)
    DiffEntr = np.zeros(3)

    EregionsV2 = np.zeros(3)
    DiffEntrV2 = np.zeros(3)

    EregionsV3 = np.zeros(3)
    DiffEntrV3 = np.zeros(3)

    EregionsV4 = np.zeros(3)
    DiffEntrV4 = np.zeros(3)

    EregionsV5 = np.zeros(3)
    DiffEntrV5 = np.zeros(3)

    EregionsV6 = np.zeros(3)
    DiffEntrV6 = np.zeros(3)

    for ep in range(1, 4):
        ###We start with making the histograms of the network
        #constants
        bucket_width = 0.05
        number_of_buckets = 160 #integer
        center = 0
        width = 2
        nr_models = 100
        epoch = ep
        if iteration == 0:
            name_pre = "REModels_1D_D1_W2/RELU_1D_D1_V"
        elif iteration == 1: 
            name_pre = "REModels_1D_D1_W2_2/RELU_1D_D1_V"
        else: name_pre = "REModels_1D_D1_W2_3/RELU_1D_D1_V"

        name_end = "-model-0"+str(epoch)+".hdf5"
        histogram = np.zeros(number_of_buckets)
        hist_w1 = np.zeros(number_of_buckets)
        hist_w2 = np.zeros(number_of_buckets)
        hist_fs1 = np.zeros(number_of_buckets) #Try to seperate the two bults
        hist_fs2 = np.zeros(number_of_buckets)
        hist_tot = np.zeros((number_of_buckets, number_of_buckets))
        nr_b = number_of_buckets/2
        empirical_average_regions = 0
        #open the models and extract the weights.
        weights = np.zeros((nr_models, width))
        for i in range(0, nr_models):
            name = name_pre+str(i)+name_end
            model = keras.models.load_model(name)
            weight = model.get_weights()
            weights[i][:] = weight[0][0]
        print("extracted all the weights")

        for i in range(0, nr_models):
            bucket_nr1 = int(np.floor((weights[i][0]-center)/bucket_width+nr_b))
            bucket_nr2 = int(np.floor((weights[i][1]-center)/bucket_width+nr_b))
            if bucket_nr1 == bucket_nr2:
                empirical_average_regions = empirical_average_regions+1
            else:
                empirical_average_regions = empirical_average_regions+2
            if bucket_nr1 > bucket_nr2:
                swap = bucket_nr2
                bucket_nr2 = bucket_nr1
                bucket_nr1 = swap

            if bucket_nr1 < 0:
                hist_fs1[0] = hist_fs1[0] + 1
                if bucket_nr2 < 0:
                    hist_tot[0][0] = hist_tot[0][0] + 1
                elif bucket_nr2 > number_of_buckets-1:
                    hist_tot[0][number_of_buckets-1] = hist_tot[0][number_of_buckets-1]+1
                else: hist_tot[0][bucket_nr2] = hist_tot[0][bucket_nr2]+1
            elif bucket_nr1 > number_of_buckets-1:
                hist_fs1[number_of_buckets-1] = hist_fs1[number_of_buckets-1] + 1 
                if bucket_nr2 < 0: 
                    hist_tot[number_of_buckets-1][0] = hist_tot[number_of_buckets-1][0]
                elif bucket_nr2 > number_of_buckets-1:
                    hist_tot[number_of_buckets-1][number_of_buckets-1] = hist_tot[number_of_buckets-1][number_of_buckets-1]+1
                else: hist_tot[number_of_buckets-1][bucket_nr2] = hist_tot[number_of_buckets-1][bucket_nr2] + 1
            else: 
                hist_fs1[bucket_nr1] = hist_fs1[bucket_nr1] + 1
                hist_tot[bucket_nr1][bucket_nr2] = hist_tot[bucket_nr1][bucket_nr2] + 1

            if bucket_nr2 < 0:
                hist_fs2[0] = hist_fs2[0] + 1
            elif bucket_nr2 > number_of_buckets - 1:
                hist_fs2[number_of_buckets-1] = hist_fs2[number_of_buckets-1] + 1
            else: hist_fs2[bucket_nr2] = hist_fs2[bucket_nr2] + 1

            for j in range(0, width):
                bucket_nr = int(np.floor((weights[i][j]-center)/bucket_width+nr_b))
                if bucket_nr < 0:
                    histogram[0] = histogram[0] + 1
                    if j == 0:
                        hist_w1[0] = hist_w1[0] + 1
                    else:
                        hist_w2[0] = hist_w2[0] + 1
                elif bucket_nr > number_of_buckets-1:
                    histogram[number_of_buckets-1] = histogram[number_of_buckets-1] + 1
                    if j == 0:
                        hist_w1[number_of_buckets-1] = hist_w1[number_of_buckets-1] + 1
                    else:
                        hist_w2[number_of_buckets-1] = hist_w2[number_of_buckets-1] + 1
                else:
                    histogram[bucket_nr] = histogram[bucket_nr] + 1
                    if j == 0:
                        hist_w1[bucket_nr] = hist_w1[bucket_nr] + 1
                    else:
                        hist_w2[bucket_nr] = hist_w2[bucket_nr] + 1

        if visualisation_histogramms == 1:
            #We plot the histogramms: visualisation
            plt.figure(1)
            x_axis = (np.array(range(0,number_of_buckets))+0.5-number_of_buckets/2)*width + center
            plt.bar(x_axis, histogram, width = 0.5*width, linewidth = 0.5)
            plt.title("Histogram weights of -"+str(epoch))
            plt.xlabel("Weight values")
            plt.ylabel("the count counts")

            plt.figure(2)
            x_axis = (np.array(range(0,number_of_buckets))+0.5-number_of_buckets/2)*width + center
            plt.bar(x_axis, hist_w1, width = 0.5*width, linewidth = 0.5)
            plt.title("Histogram W1 of -"+str(epoch))
            plt.xlabel("Weight values")
            plt.ylabel("the count counts")

            plt.figure(3)
            x_axis = (np.array(range(0,number_of_buckets))+0.5-number_of_buckets/2)*width + center
            plt.bar(x_axis, hist_w2, width = 0.5*width, linewidth = 0.5)
            plt.title("Histogram W2 of -"+str(epoch))
            plt.xlabel("Weight values")
            plt.ylabel("the count counts")

            plt.figure(4)
            x_axis = (np.array(range(0,number_of_buckets))+0.5-number_of_buckets/2)*width + center
            plt.bar(x_axis, hist_fs1, width = 0.5*width, linewidth = 0.5)
            plt.title("Histogram fs1 of -"+str(epoch))
            plt.xlabel("Weight values")
            plt.ylabel("the count counts")

            plt.figure(5)
            x_axis = (np.array(range(0,number_of_buckets))+0.5-number_of_buckets/2)*width + center
            plt.bar(x_axis, hist_fs2, width = 0.5*width, linewidth = 0.5)
            plt.title("Histogram fs2 of -"+str(epoch))
            plt.xlabel("Weight values")
            plt.ylabel("the count counts")

            plt.show()

        ###Method 1: The histogram is F_w and we use Fs1/Fs2 to define F_w1_w2

        #We compute H_diff
        f_w = histogram/np.sum(histogram)
        f_s1 = hist_fs1/np.sum(hist_fs1)
        f_s2 = hist_fs2/np.sum(hist_fs2)
        Size = np.size(f_s1)
        f_w1_w2 = np.zeros((Size, Size))
        for i in range(0, Size):
            for j in range(0, Size):
                f_w1_w2[i][j] = f_s1[i]*f_s2[j]
        f_w1_w2 = (f_w1_w2 + np.transpose(f_w1_w2))/2

        H_W_V1 = 0
        H_W1_W2_V1 = 0
        for i in range(0, Size):
            if f_w[i] != 0:
                H_W_V1 = H_W_V1 + f_w[i]*np.log2(f_w[i])
            for j in range(0, Size):
                if f_w1_w2[i][j] != 0:
                    H_W1_W2_V1 = H_W1_W2_V1 + f_w1_w2[i][j]*np.log2(f_w1_w2[i][j])

        H_W_V1 = -2*H_W_V1
        H_W1_W2_V1 = -H_W1_W2_V1

        H_diffV1 = H_W_V1-H_W1_W2_V1

        #We compute expected number of regions
        RegionsV1 = 0
        for i in range(0, Size):
            RegionsV1 = RegionsV1 + 1 - (1-f_s1[i])*(1-f_s2[i])
        #Empirecal average number of regions
        empirical_average_regions = empirical_average_regions/nr_models
        empirical_average[ep-1] = empirical_average_regions
        DiffEntr[ep-1] = H_diffV1
        Eregions[ep-1] = RegionsV1

        ###Method 2 we use fs1 and fs2 but compute H_diff independent->0
        f_w1_w1 = np.zeros((Size, Size))
        f_w2_w2 = np.zeros((Size, Size))
        for i in range(0, Size):
            for j in range(0, Size):
                f_w1_w1[i][j] = f_s1[i]*f_s1[j]
                f_w2_w2[i][j] = f_s2[i]*f_s2[j]
        f_w1_w2_v2 = ((f_w1_w1+f_w2_w2)/2+f_w1_w2)/2
        H_W1_W2_V2 = 0
        for i in range(0, Size):
            for j in range(0, Size):
                if f_w1_w2_v2[i][j] != 0:
                    H_W1_W2_V2 = H_W1_W2_V2 + f_w1_w2_v2[i][j]*np.log2(f_w1_w2_v2[i][j])
        H_W1_W2_V2 = -H_W1_W2_V2
        H_diffV2 = H_W_V1 - H_W1_W2_V2
        RegionsV2 = 0
        for i in range(0, Size):
            for j in range(0, Size):
                if i != j:
                    RegionsV2 = RegionsV2+2*f_w1_w2_v2[i][j]
                else: RegionsV2 = RegionsV2+f_w1_w2_v2[i][j]
        EregionsV2[ep-1] = RegionsV2
        DiffEntrV2[ep-1] = H_diffV2

        ###Method 3 now we use f_w1 and f_w2 for H_W
        f_w1 = hist_w1/np.sum(hist_w1)
        f_w2 = hist_w2/np.sum(hist_w2)

        H_W1_W2_V3 = H_W1_W2_V1
        H_W1_V3 = 0
        H_W2_V3 = 0
        for i in range(0, Size):
            if f_w1[i] != 0:
                H_W1_V3 = H_W1_V3 - f_w1[i]*np.log2(f_w1[i])
            if f_w2[i] != 0:
                H_W2_V3 = H_W2_V3 - f_w2[i]*np.log2(f_w2[i])
        H_W_V3 = H_W1_V3 + H_W2_V3
        H_diffV3 = H_W_V3 - H_W1_W2_V3

        RegionsV3 = RegionsV1
        
        EregionsV3[ep-1] = RegionsV3
        DiffEntrV3[ep-1] = H_diffV3

        ###Method 4
        H_W1_W2_V4 = H_W1_W2_V2
        RegionsV4 = RegionsV2
        
        H_diffV4 = H_W_V3 - H_W1_W2_V4

        EregionsV4[ep-1] = RegionsV4
        DiffEntrV4[ep-1] = H_diffV4

        ###Method 5 Empirical but sum fs1 and fs2 for F_W1_W2
        H_W1_W2_V5 = 0
        f_w1_w2_v5 = hist_tot/(np.sum(hist_w1))
        RegionsV5 = 0
        for i in range(0, number_of_buckets):
            for j in range(0, number_of_buckets):
                if i != j:
                    RegionsV5 =RegionsV5 + 2*f_w1_w2_v5[i][j]
                else: RegionsV5 = RegionsV5 + f_w1_w2_v5[i][j]

                if f_w1_w2_v5[i][j] != 0:
                    H_W1_W2_V5 = H_W1_W2_V5 + f_w1_w2_v5[i][j]*np.log2(f_w1_w2_v5[i][j])
        H_W1_W2_V5 = -H_W1_W2_V5
        H_diffV5 = H_W_V1 - H_W1_W2_V5
        
        EregionsV5[ep-1] = RegionsV5
        DiffEntrV5[ep-1] = H_diffV5

        ###Method 6
        H_W1_W2_V6 = H_W1_W2_V5
        H_diffV6 = H_W_V3 - H_W1_W2_V6

        EregionsV6[ep-1] = RegionsV5
        DiffEntrV6[ep-1] = H_diffV6






        
    emp_aver[iteration][:] = empirical_average

    E_regionsV1[iteration][:] = Eregions
    D_entrV1[iteration][:] = DiffEntr

    E_regionsV2[iteration][:] = EregionsV2
    D_entrV2[iteration][:] = DiffEntrV2

    E_regionsV3[iteration][:] = EregionsV3
    D_entrV3[iteration][:] = DiffEntrV3

    E_regionsV4[iteration][:] = EregionsV4
    D_entrV4[iteration][:] = DiffEntrV4

    E_regionsV5[iteration][:] = EregionsV5
    D_entrV5[iteration][:] = DiffEntrV5

    E_regionsV6[iteration][:] = EregionsV6
    D_entrV6[iteration][:] = DiffEntrV6

    if semi_visualization == 1:
        ###Visualisation global results
        plt.figure(6)
        plt.plot(DiffEntr, Eregions, '*')
        plt.plot(DiffEntr, empirical_average, '+')
        print(empirical_average)
        plt.xlabel("Differential entropy")
        plt.ylabel("Expected number of regions")
        plt.title("method 1: +=empirical, *=from distributions")

        plt.figure(7)
        plt.plot(DiffEntrV2, EregionsV2, '*')
        plt.plot(DiffEntrV2, empirical_average, '+')
        plt.xlabel("Differential entropy")
        plt.ylabel("Expected number of regions")
        plt.title("method 2: +=empirical, *=from distributions")

        plt.figure(8)
        plt.plot(DiffEntrV3, EregionsV3, '*')
        plt.plot(DiffEntrV3, empirical_average, '+')
        plt.xlabel("Differential entropy")
        plt.ylabel("Expected number of regions")
        plt.title("method 3: +=empirical, *=from distributions")

        plt.figure(9)
        plt.plot(DiffEntrV4, EregionsV4, '*')
        plt.plot(DiffEntrV4, empirical_average, '+')
        plt.xlabel("Differential entropy")
        plt.ylabel("Expected number of regions")
        plt.title("method 4: +=empirical, *=from distributions")
        plt.show()

empAv = np.zeros(3)
empAv_var = np.zeros(3)
RV1 = np.zeros(3)
RV1_var = np.zeros(3)
RV2 = np.zeros(3)
RV2_var = np.zeros(3)
RV3 = np.zeros(3)
RV3_var = np.zeros(3)
RV4 = np.zeros(3)
RV4_var = np.zeros(3)
RV5 = np.zeros(3)
RV5_var = np.zeros(3)
RV6 = np.zeros(3)
RV6_var = np.zeros(3)
H_DV1 = np.zeros(3)
H_DV1_var = np.zeros(3)
H_DV2 = np.zeros(3)
H_DV2_var = np.zeros(3)
H_DV3 = np.zeros(3)
H_DV3_var = np.zeros(3)
H_DV4 = np.zeros(3)
H_DV4_var = np.zeros(3)
H_DV5 = np.zeros(3)
H_DV5_var = np.zeros(3)
H_DV6 = np.zeros(3)
H_DV6_var = np.zeros(3)

for i in range(0, 3):
    empAv[i] = np.mean(emp_aver[:][i])
    empAv_var[i] = np.var(emp_aver[:][i])

    RV1[i] = np.mean(E_regionsV1[:][i])
    RV1_var[i] = np.var(E_regionsV1[:][i])

    RV2[i] = np.mean(E_regionsV2[:][i])
    RV2_var[i] = np.var(E_regionsV2[:][i])

    RV3[i] = np.mean(E_regionsV3[:][i])
    RV3_var[i] = np.var(E_regionsV3[:][i])

    RV4[i] = np.mean(E_regionsV4[:][i])
    RV4_var[i] = np.var(E_regionsV4[:][i])

    RV5[i] = np.mean(E_regionsV5[:][i])
    RV5_var[i] = np.var(E_regionsV5[:][i])

    RV6[i] = np.mean(E_regionsV6[:][i])
    RV6_var[i] = np.var(E_regionsV6[:][i])

    H_DV1[i] = np.mean(D_entrV1[:][i])
    H_DV1_var[i] = np.var(D_entrV1[:][i])

    H_DV2[i] = np.mean(D_entrV2[:][i])
    H_DV2_var[i] = np.var(D_entrV2[:][i])

    H_DV3[i] = np.mean(D_entrV3[:][i])
    H_DV3_var[i] = np.var(D_entrV3[:][i])

    H_DV4[i] = np.mean(D_entrV4[:][i])
    H_DV4_var[i] = np.var(D_entrV4[:][i])

    H_DV5[i] = np.mean(D_entrV5[:][i])
    H_DV5_var[i] = np.var(D_entrV5[:][i])

    H_DV6[i] = np.mean(D_entrV6[:][i])
    H_DV6_var[i] = np.var(D_entrV6[:][i])

###Visualisation with errorbars based on means etc
plt.figure(10)
plt.errorbar(H_DV1, RV1, yerr= np.sqrt(RV1_var),xerr = np.sqrt(H_DV1_var), fmt='*')
plt.errorbar(H_DV1, empAv, yerr= np.sqrt(empAv_var), fmt='+')
plt.xlabel("Mutual Information")
plt.ylabel("Expected number of regions")
print(H_DV1_var)
plt.title("method 1: +=empirical, *=from distributions")


plt.figure(11)
plt.errorbar(H_DV2, RV2, yerr= np.sqrt(RV2_var),xerr = np.sqrt(H_DV2_var), fmt='*')
plt.errorbar(H_DV2, empAv, yerr= np.sqrt(empAv_var), fmt='+')
plt.xlabel("Mutual Information")
plt.ylabel("Expected number of regions")
plt.title("method 2: +=empirical, *=from distributions")

plt.figure(12)
plt.errorbar(H_DV3, RV3, yerr= np.sqrt(RV3_var),xerr = np.sqrt(H_DV3_var), fmt='*')
plt.errorbar(H_DV3, empAv, yerr= np.sqrt(empAv_var), fmt='+')
plt.xlabel("Mutual Information")
plt.ylabel("Expected number of regions")
plt.title("method 3: +=empirical, *=from distributions")

plt.figure(13)
plt.errorbar(H_DV4, RV4, yerr= np.sqrt(RV4_var), xerr = np.sqrt(H_DV4_var), fmt='*')
plt.errorbar(H_DV4, empAv, yerr= np.sqrt(empAv_var), fmt='+')
plt.xlabel("Mutual Information")
plt.ylabel("Expected number of regions")
plt.title("method 4: +=empirical, *=from distributions")

plt.figure(14)
plt.errorbar(H_DV5, RV5, yerr= np.sqrt(RV5_var),xerr = np.sqrt(H_DV5_var),  fmt='*')
plt.errorbar(H_DV5, empAv, yerr= np.sqrt(empAv_var), fmt='+')
plt.xlabel("Mutual Information")
plt.ylabel("Expected number of regions")
plt.title("method 5: +=empirical, *=from distributions")

plt.figure(15)
plt.errorbar(H_DV6, RV6, yerr= np.sqrt(RV6_var),xerr = np.sqrt(H_DV6_var), fmt='*')
plt.errorbar(H_DV6, empAv, yerr= np.sqrt(empAv_var), fmt='+')
plt.xlabel("Mutual Information")
plt.ylabel("Expected number of regions")
plt.title("method 6: +=empirical, *=from distributions")
print(H_DV6_var)
plt.show()