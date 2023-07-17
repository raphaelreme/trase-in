#importing packages
import numpy as np
import scipy as sp
import skimage.io as iio
import skvideo

#set this path to the FFmpegTool\bin location on your machine (download FFmpeg if not already installed)
#make sure to restart the kernel after setting the path
skvideo.setFFmpegPath("/usr/bin/")

import skvideo.io as io
import pandas as pd
import matplotlib.pyplot as plt
from caiman.source_extraction.cnmf import deconvolution as deconv
from scipy.spatial.distance import cdist, pdist, euclidean
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import cv2

#Defining Functions - SHyGI Single-cell Hydra GCaMP Imaging package

#Load data from specified paths
def Read_Data(csv_path, vid_path, red_vid_path):
    positions = (pd.read_csv(csv_path,usecols=['TrackID','t','x','y'])).values
    vid = io.vread(vid_path)
    red_vid = io.vread(red_vid_path)
    return positions, vid, red_vid

#Load data from specified folder of .tif files - Higher resolution but longer processing time
def Read_Data_TIFseq(csv_path, vid_path, red_vid_path):
    positions = (pd.read_csv(csv_path,usecols=['TrackID','t','x','y'])).values
    vid = iio.ImageCollection(vid_path + '/*.tif')
    red_vid = iio.ImageCollection(red_vid_path + '/*.tif')
    return positions, vid, red_vid

def Read_Data_TIFseq_byo(npy_path, vid_path, red_vid_path):
    positions = np.load(npy_path, allow_pickle=True)
    vid = iio.ImageCollection(vid_path + '/*.tif')
    red_vid = iio.ImageCollection(red_vid_path + '/*.tif')
    return positions, vid, red_vid

#ROI function
def ROIextractor(frame,points,dim):
    #why are these NOT already integers?? is it not pixel values?
    x = int(points[0])
    y = int(points[1])
    row1 = x - dim
    row2 = x + dim
    column1 = y - dim
    column2 = y + dim
    ROI = frame[column1:column2+1,row1:row2+1]
    return ROI

#Filter Nematocytes
#dead neuron and nematocyte removal function - removed tracks in the lowest set percentile of standard deviation
#flaw is that it will remove neurons if there are no nematocytes recorded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def filt_nematocytes(intensities,percentile_threshold,positions):
    std_devs = []
    filtered_neurons = np.zeros((len(intensities),len(intensities[0])))

    for i in range(len(intensities)):
        std_dev = np.std(intensities[i])
        std_devs.append(std_dev)

    #lowest 20th percentile to find nematocytes
    std_range = (np.max(std_devs) - np.min(std_devs))
    percentile = (std_range/100)
    threshold = np.min(std_devs)+(percentile*percentile_threshold)

    for i in range(len(intensities)):
        if std_devs[i] >= threshold:
            filtered_neurons[i]=intensities[i]

    filtered_neurons2 = []
    positions2 = []
    for i in range(len(filtered_neurons)):
        if np.ndarray.tolist(filtered_neurons[i]) != np.ndarray.tolist(np.zeros(len(filtered_neurons[i]))):
            filtered_neurons2.append(filtered_neurons[i])
            positions2.append(positions[i])

    return filtered_neurons2, positions2

#Filters non-neuronal cells based on how well signals follow a gaussian distribution
def Gaussian_noise_filter(intensities, alpha, positions):
    new_data = []
    positions_new = []
    filtered_tracks = []
    for i in range(len(intensities)):
        a,b = sp.stats.normaltest(intensities[i])
        if b >= alpha:
            #follows normal dist
            filtered_tracks.append(intensities[i])
        elif b < alpha:
            #does not follow normal dist
            new_data.append(intensities[i])
            positions_new.append(positions[i])
    return new_data, positions_new, filtered_tracks

#smoothing step
def smoother(intensities, window):
    N = window
    filtered_data = []
    for i in range(len(intensities)):
        filtered_data.append(np.convolve(intensities[i], np.ones((N,))/N, mode='valid'))
    return filtered_data

#deltaF/F
#following idea from the c.elegans paper on whole brain imaging
def df_f(intensity):
    df_f = []
    for i in range(len(intensity)):
        df_f_neuron = []

        #lowest 20th percentile used as base f
        range_f = np.max(intensity[i]) - np.min(intensity[i])
        percentile = range_f/100
        percentile_boundary = 20
        f_bkg = np.min(intensity[i])+(percentile*percentile_boundary)

        #f = min(intensities[i])
        #f = np.mean(intensities[i][0:10])
        f = f_bkg
        for j in range(len(intensity[0])):
            df_f_frame = (intensity[i][j] - f)/f
            df_f_neuron.append(df_f_frame)
        df_f.append(df_f_neuron)
    return df_f

#deltaR/R
def find_dR_R(intensities_green,intensities_red):
    dR_R = []
    for i in range(len(intensities_red)):
        dR_R_neuron = []

        #create R array
        R = np.zeros(len(intensities_red[i]))
        for k in range(len(intensities_red[i])):
            R[k] = intensities_green[i][k]/intensities_red[i][k]

        #lowest 20th percentile used as base r (from C. Elegans Paper)
        range_R = np.max(R) - np.min(R)
        percentile = range_R/100
        percentile_boundary = 20
        R_bkg = np.min(R)+(percentile*percentile_boundary)

        for j in range(len(intensities_red[i])):
            dR_R_frame = ((R[j] - R_bkg)/R_bkg)
            dR_R_neuron.append(dR_R_frame)
        dR_R.append(dR_R_neuron)
    return dR_R

# #Deconvolution using ICA - code modified from 'Predicting natural behavior from whole-brain neural dynamics' by Scholz et al
# def decorrelateNeuronsICA(R, G, tolerance):
#     """use PCA to remove covariance in Green and Red signals."""
#     R = np.asanyarray(R)
#     G = np.asanyarray(G)
#     Ynew = []
#     ica = FastICA(n_components = 2, tol = tolerance)
#     for li in range(len(R)):
#         Y = np.vstack([R[li], G[li]]).T
#         sclar2= StandardScaler(copy=True, with_mean=True, with_std=True)
#         Y = sclar2.fit_transform(Y)
#         S = ica.fit_transform(Y)
#         # order components by max correlation with red signal
#         v = [np.corrcoef(s,G[li])[0,1] for s in S.T]
#         idn = np.argmin(np.abs(v))
#         # check if signal needs to be inverted
#         sign = np.sign(np.corrcoef(S[:,idn],R[li])[0,1])
#         signal = sign*(S[:,idn])
#         Ynew.append(signal)
#     return np.array(Ynew)#, np.mean(var, axis=0), Rs, Gs

def ICAdecorr(G, R, tolerance, repeats):
    #edited function from Scholz et al to account for randomness of ICA by repeating multiple times and selecting best outcome
    R = np.asanyarray(R)
    G = np.asanyarray(G)
    Ynew = []
    for li in range(len(R)):
        possible_outcomes = []
        for k in range(repeats):
            ica = FastICA(n_components = 2, tol = tolerance)
            Y = np.vstack([G[li], R[li]]).T
            sclar2= StandardScaler(copy=True, with_mean=True, with_std=True)
            Y = sclar2.fit_transform(Y)
            S = ica.fit_transform(Y)
            # order components by max correlation with red signal
            v = [np.corrcoef(s,R[li])[0,1] for s in S.T]
            idn = np.argmin(np.abs(v))
            # check if signal needs to be inverted
            sign = np.sign(np.corrcoef(S[:,idn],G[li])[0,1])
            signal = sign*(S[:,idn])
            possible_outcomes.append(signal)
        #best_outcome = possible outcome least correlated with the red signal (including anticorrelation)
        correlations = []
        for j in range(len(possible_outcomes)):
            correlations.append(np.corrcoef(possible_outcomes[j], R[li])[0,1])
        min_corr_index = np.argmin(np.abs(correlations))
        best_outcome = possible_outcomes[min_corr_index]
        Ynew.append(best_outcome)
    return np.array(Ynew)

#Normalise data between 0 and 1
def norm_Data(data):
    data_out = []
    for i in range(len(data)):
        data_out.append( (data[i] - min(data))/(max(data) - min(data)) )
    return data_out

#Normalise all trakcs in array between 0 and 1
def norm_all_data(data):
    data_norm = []
    for i in range(len(data)):
        data_norm.append(norm_Data(data[i]))
    return data_norm

#reshaping the .csv to be analysed
def reshaper(positions):
    number_tracks = int(positions[len(positions)-1,0])
    position_reshaped = []
    for i in range(number_tracks):
        positions_track = []
        for j in range(len(positions)):
            if positions[j,0] == i:
                positions_track.append(positions[j,2:4])
        position_reshaped.append(positions_track)
    posit = np.asanyarray(position_reshaped)
    return posit

#Possibly faster untested reshaper function
def fastreshaper(positions):
    number_tracks = int(positions[len(positions)-1,0])
    number_frames = int(max(positions[:,1])+1)
    position_reshaped = np.zeros([number_tracks, number_frames, 2])
    for i in range(number_tracks):
        positions_track = np.zeros([number_frames, 2])
        frame_count = 0
        for j in range(len(positions)):
            if positions[j,0] == i:
                positions_track[frame_count] = positions[j,2:4]
                frame_count += 1
        position_reshaped[i] = positions_track
    posit = np.asanyarray(position_reshaped)
    return posit

#removing incomplete tracks
def remove_incomplete_tracks(posit, num_frames):
    posit_corrected = []
    for i in range(len(posit)):
        if len(posit[i]) == num_frames:
            posit_corrected.append(posit[i])
    return posit_corrected

#extract ROI throughout the video and record intensities
def Extract_Fluorescence(position_corrected, video, dimention):
    dim = dimention
    vid = video
    posit_corrected = position_corrected
    num_frames = len(posit_corrected[0])
    intensities = []
    #needs to update position array as some fluorescence tracks may be dropped here
    position_updated = []
    for track in range(len(posit_corrected)):
        intensity = []
        break_count = 0
        for frame in range(num_frames):
            area = ROIextractor(vid[frame],posit_corrected[track][frame],dim)
            near_edge = np.isnan(np.mean(area))
            if near_edge != True:
                intensity.append(np.mean(area))
            else:
                dim_count = 1
                while np.isnan(np.mean(area)) == True and dim-dim_count > dim/2:
                    area = ROIextractor(vid[frame],posit_corrected[track][frame],dim-dim_count)
                    dim_count += 1
                if dim-dim_count > dim/2:
                    intensity.append(np.mean(area))
                else:
                    break_count = 1
                    break
        if break_count == 0:
            intensities.append(intensity)
            position_updated.append(position_corrected[track])
        else:
            continue
    return intensities, position_updated

#show all tracking info on neuron
def full_eval(neuron, signal, eval_frame, dim, posit_corrected, vid):
    #intensity over time plot of the neuron
    plt.plot(signal[neuron])
    plt.show()

    #neuron in ROI
    plt.imshow(ROIextractor(vid[eval_frame],posit_corrected[neuron][eval_frame],dim))
    plt.show()

    #position of neuron on animal at set frame
    plt.imshow(vid[eval_frame])
    plt.scatter(posit_corrected[neuron][eval_frame][0],posit_corrected[neuron][eval_frame][1], edgecolors = 'r',facecolors='none')
    plt.show()

def plot_all(input_array):
    for i in range(len(input_array)):
        plt.figure(i)
        plt.plot(input_array[i])
        plt.title(i)
    plt.show()

#heatmap plotting
def plot_heatmap(dataset, title, scale):
    fig, axs = plt.subplots(1,1)
    heatmap_neurons = plt.imshow(dataset, aspect = 'auto', origin = 'lower')
    plt.xlabel('Frame')
    plt.ylabel('Neuron')
    plt.title(title)
    cbar = plt.colorbar(heatmap_neurons, ax=axs, orientation='vertical', fraction=.1)
    cbar.set_label(scale, rotation = '-90', labelpad = 15)
    cbar.minorticks_on()
    plt.show()

#detrends single neuron's trace - used to evaluate ideal polynomial degree
#polynomial of degree 17 works well as starting point
def detrend(data, polynomial_degree):
    x_vals = np.arange(len(data))
    coeffs = np.polyfit(x_vals, data, polynomial_degree)
    polynomial = np.polyval(coeffs, x_vals)
    new_sequence = data - polynomial
    return new_sequence

#detrendes all data in array
def detrend_all(input_array, polynomial_degree):
    detrended = []
    poly_deg = polynomial_degree
    for i in range(len(input_array)):
        detrended.append(detrend(input_array[i], poly_deg))
    return detrended

#superimpose ROI onto video frame
def Super_impose(video, frame_to_view, positions, Title):
    vid = video
    plt.imshow(vid[frame_to_view])
    for track in range(len(positions)):
        plt.scatter(positions[track][frame_to_view][0],positions[track][frame_to_view][1], edgecolors = 'r',facecolors='none')
    plt.title(Title)
    plt.show()

#Super impose locations of neurons in specific cluster onto video
def Super_impose_cluster(video, frame_to_view, posit_corrected, clusters, cluster_to_view, Title):
    vid = video
    plt.imshow(vid[frame_to_view])
    cluster_index = []

    for i in range(len(clusters)):
        if clusters[i] == cluster_to_view:
            cluster_index.append(i)

    for track in cluster_index:
        plt.scatter(posit_corrected[track][frame_to_view][0],posit_corrected[track][frame_to_view][1], edgecolors = 'r',facecolors='none')
    plt.title(Title)
    plt.show()

#Uses FOOPSI algrorithm from CAIMAN package to decompose calcium signal and estimate neural spikes
def FOOPSI_all(detrended_data):
    detrended = detrended_data
    spikes_signal_dR = np.zeros((len(detrended),len(detrended[1])))
    ca_foopsi_traces = np.zeros((len(detrended),len(detrended[1])))
    for i in range(len(detrended)):
        ca_foopsi,cb,b1,g,c1,spikes_foopsi,lam = deconv.constrained_foopsi(np.asanyarray(detrended[i]),p=2)
        spikes_signal_dR[i] = spikes_foopsi
        ca_foopsi_traces[i] = ca_foopsi
    return ca_foopsi_traces, spikes_signal_dR

#Uses FOOPSI spike information to generate an array to be plotted as a raster plot
def Find_Raster(foopsi_spikes, threshold):
    spikes_signal_dR = foopsi_spikes
    spike_thresh_dR = threshold
    raster_array_dR = np.zeros((len(spikes_signal_dR),len(spikes_signal_dR[1])))
    for i in range(len(spikes_signal_dR)):
        for j in range(len(spikes_signal_dR[i])):
            if max(spikes_signal_dR[i]) > 0:
                if spikes_signal_dR[i][j] >= spike_thresh_dR: #*np.mean(spikes_signal_dR[i]):
                    raster_array_dR[i][j] = j
    return raster_array_dR

def Find_Raster_adaptive(foopsi_spikes, ratio):
    spikes_signal_dR = foopsi_spikes
    raster_array_dR = np.zeros((len(spikes_signal_dR),len(spikes_signal_dR[1])))
    for i in range(len(spikes_signal_dR)):
        spike_thresh_dR = np.max(spikes_signal_dR[i])*ratio
        for j in range(len(spikes_signal_dR[i])):
            if max(spikes_signal_dR[i]) > 0:
                if spikes_signal_dR[i][j] >= spike_thresh_dR: #*np.mean(spikes_signal_dR[i]):
                    raster_array_dR[i][j] = j
    return raster_array_dR


def Find_Raster_adaptive2(foopsi_spikes, ratio):
    spikes_signal_dR = foopsi_spikes
    raster_array_dR = np.zeros((len(spikes_signal_dR),len(spikes_signal_dR[1])))
    for i in range(len(spikes_signal_dR)):
        stdev = np.std(spikes_signal_dR[i])
        spike_thresh_dR = np.mean(spikes_signal_dR[i])-3*stdev
        for j in range(len(spikes_signal_dR[i])):
            if max(spikes_signal_dR[i]) > 0:
                if spikes_signal_dR[i][j] >= spike_thresh_dR: #*np.mean(spikes_signal_dR[i]):
                    raster_array_dR[i][j] = j
    return raster_array_dR

#extract clusters from dendogram
#borrowed from wed (datanongrata.com)
def give_cluster_assigns(df, numclust, transpose):
    if transpose==True:
        data_dist = pdist(df.transpose())
        data_link = shc.linkage(data_dist, method='ward')
        cluster_assigns=pd.Series(shc.fcluster(data_link, numclust, criterion='maxclust', monocrit=None), index=df.columns)
    else:
        data_dist = pdist(df)
        data_link = shc.linkage(data_dist, metric = 'correlation', method='ward')
        cluster_assigns=pd.Series(shc.fcluster(data_link, numclust, criterion='maxclust', monocrit=None), index=df.index)
    return cluster_assigns

#show all tracking info on neuron
def single_neuron_investigation(neuron, signal, Foopsi_ca, raster_array_dR, vid, eval_frame, dim, posit_corrected):
    print('Neuron: ', neuron)

    #intensity over time plot of the neuron
    plt.plot(signal[neuron])
    plt.title('Raw Intensity Signal')
    plt.show()

    #Denoised Ca signal plot
    plt.plot(Foopsi_ca[neuron])
    plt.title('Denoised Ca Signal')
    plt.show()

    #Raster plot of neuron
    plt.eventplot(raster_array_dR[neuron],linelengths = 0.6)
    plt.xlim((1,len(raster_array_dR[neuron])))
    plt.title('Raster Plot of Neuron')
    plt.show()

    #neuron in ROI
    plt.imshow(ROIextractor(vid[eval_frame],posit_corrected[neuron][eval_frame],dim))
    plt.title('Neuron ROI')
    plt.show()

    #position of neuron on animal at set frame
    plt.imshow(vid[eval_frame])
    plt.scatter(posit_corrected[neuron][eval_frame][0],posit_corrected[neuron][eval_frame][1], edgecolors = 'r',facecolors='none')
    plt.title('Neuron Position on Hydra')
    plt.show()

#create a 2D gaussian kernel
def gkern(kernlen, nsig):
    'from stack overflow'
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(sp.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

#extraction of intensity from single neuron within ROI
def SingleCellIntensity(neuron, video, positions, dimentionROI, Circle_radius, distance_threshold, display_on = False):
    display = []
    mask_circle_radius = Circle_radius - 0
    neuron_points = []
    intensities = []
    positions_corrected = []
    backframes = 10
    past_thresh = 3
    for frame in range(len(video)):
#         print('Frame: ', frame)
        #Load raw image and copy to avoid affecting the original video
        raw_image = ROIextractor(video[frame],positions[neuron][frame], dimentionROI)
        image = raw_image.copy()

        #correction for ROI leaving field
        dim_image = np.min(image.shape[0:1])
        if dim_image < 5:
            dim_image = 5
        image = image[0:dim_image, 0:dim_image]

        #Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Copy Image pre processing to use for finding final intensity value from
        image_out = image.copy()
        image_display = image.copy()
        #Create Gaussian Kernel to make points close to the centre appear brighter than points on the edges
        kernel_gauss = gkern(image.shape[0], 0.7)
        image = kernel_gauss*image
        #find Centre Point of the image to use as starting point for finding neuron
        centrept = [int(len(image[0])/2), int(len(image[1])/2)]

        if frame == 0:
            #find first 3 highest pixels highest pixel
            highpt_prev = centrept
            highpts = []
            distances = []
            for k in range(4):
                image = cv2.GaussianBlur(image, (5,5), 0)
                low,high,lowpt,highpt = cv2.minMaxLoc(image)
                highpts.append(highpt)

                #add brighness coefficient to couteract possibility of darker point near centre
                bright_coeff = (1/high)*0.5

                cv2.circle(image, highpt, Circle_radius, 0, -1)

                distances.append(bright_coeff*euclidean(highpts[k], highpt_prev))

            #take point closest to centre
            mindist = np.argmin(distances)
            neuronpt = highpts[mindist]

        elif frame > 0:
            image = cv2.GaussianBlur(image, (5,5), 0)
            low,high,lowpt,highpt = cv2.minMaxLoc(image)
            distance = euclidean(highpt, highpt_prev)

            if frame < backframes:
                if distance <= distance_threshold:
                    neuronpt = highpt

                elif distance > distance_threshold:
                    for k in range(3):
                        cv2.circle(image, highpt, Circle_radius, 0, -1)
                        image = cv2.GaussianBlur(image, (5,5), 0)
                        low,high,lowpt,highpt = cv2.minMaxLoc(image)
                        distance = euclidean(highpt, highpt_prev)
                        if distance <= distance_threshold:
                            neuronpt = highpt
                            break

            else:
                distance_past = euclidean(highpt, neuron_points[frame-backframes])

                if distance <= distance_threshold and distance_past <= past_thresh:
                    neuronpt = highpt

                if distance > distance_threshold:
#                     print('thresh1')
                    for k in range(3):
                        cv2.circle(image, highpt, Circle_radius-1, 0, -1)
                        image = cv2.GaussianBlur(image, (5,5), 0)
                        low,high,lowpt,highpt = cv2.minMaxLoc(image)
                        distance = euclidean(highpt, highpt_prev)
                        if distance <= distance_threshold:
                            neuronpt = highpt
                            distance_past = euclidean(highpt, neuron_points[frame-backframes])
                            break

                if distance_past > past_thresh: # and distance <= distance_threshold:
#                     print('past')
                    neuronpt = neuron_points[frame-2]

        #save_values for next iteration
        highpt_prev = neuronpt
        neuron_points.append(neuronpt)
#         print('npt ',neuronpt)

        #extract fluorescence within selected circle
        circle_mask = np.zeros((image_out.shape[0],image_out.shape[1]),dtype = np.uint8)
        cv2.circle(circle_mask, neuronpt, mask_circle_radius, 255, -1)

        if display_on == True:
            image_display = cv2.GaussianBlur(image_display, (5,5), 0)
            cv2.circle(image_display, neuronpt, Circle_radius, 255)
            plt.imshow(image_display)
            plt.show()
            display.append(image_display)

        image_out = cv2.GaussianBlur(image_out, (5,5), 0)
        mean = np.max(cv2.mean(image_out, mask=circle_mask))
#         print('mean: ', mean)
#         print('max: ', np.max(image_out))
        intensities.append(mean)
        positions_corrected.append(positions[neuron][frame])
    if display_on == True:
        return intensities, positions_corrected, neuron_points, display
    else:
        return  intensities, positions_corrected, neuron_points

#red single cell ROI
def Red_SingleCellIntensity(neuron, points, vid_red, dim, positions, Circle_radius):
    images_out = []
    intensities = []
    mask_circle_radius = Circle_radius - 0
    for frame in range(len(points)):
        raw_image_red = ROIextractor(vid_red[frame],positions[neuron][frame], dim)
        image_red = raw_image_red.copy()

        #correction for ROI leaving field
        dim_image = np.min(image_red.shape[0:1])
        image_red = image_red[0:dim_image, 0:dim_image]

         #extract fluorescence within selected circle
        circle_mask = np.zeros((image_red.shape[0],image_red.shape[1]),dtype = np.uint8)
        cv2.circle(circle_mask, points[frame], mask_circle_radius, 255, -1)

        image_display = cv2.GaussianBlur(image_red, (5,5), 0)
        cv2.circle(image_display, points[frame], Circle_radius, 255)
#         plt.imshow(image_display)
#         images_out.append(image_display)
#         plt.show()

        image_out = cv2.GaussianBlur(image_red, (5,5), 0)
        mean = np.max(cv2.mean(image_out, mask=circle_mask))
#         plt.imshow(image_out)
#         plt.show()
#         print('mean: ', mean)
#         print('max: ', np.max(image_out))
        intensities.append(mean)
    return intensities#, images_out

#intensities from all individual neurons
def SingleCellIntensities(video, positions, dimentionROI, Circle_radius, distance_threshold):

    '''Must Run the Extract_Fluorescence function before the SingleCellIntensities function as you need
    to use the updated 'posit_corrected' output from Extract_Fluorescence as the positions for
    SingleCellIntensities as it has no feature to correct this itself'''

    intensities = []
    position_updated = []
    all_points = []
    for track in range(len(positions)):
        try:
            intensity, posits_corr,points = SingleCellIntensity(track, video, positions, dimentionROI, Circle_radius, distance_threshold)
            intensities.append(intensity)
            position_updated.append(positions[track])
            all_points.append(points)
        except Exception as e:
            print(e)
            print('Issue in track ID: ', track)
    return intensities, position_updated, all_points

#intensities from all individual red channels
def SingleCellIntensities_Red(video_red, all_points, positions, dimentionROI, Circle_radius):
    intensities_red = []
    for track in range(len(positions)):
        intensity = Red_SingleCellIntensity(track, all_points[track], video_red, dimentionROI, positions, Circle_radius)
        intensities_red.append(intensity)
    return intensities_red
