#!/usr/bin/env python

import argparse
import numpy as np

from scipy.signal import butter, filtfilt

def get_args():
    parser = argparse.ArgumentParser(description='Remove wiggles in FTIR data')
    parser.add_argument("csvfile", help="csv file with wavenumber, intensity")
    parser.add_argument("outfile", help="csv output file")
    parser.add_argument("-p", action="store_true",
                        help="generate plot file")
    parser.add_argument("-w",  type=float, default=35,
                        help="Bandstop filter Half width in wavenumber [ 35 ]")
    parser.add_argument("-c",  type=float, default=125,
                        help="Bandstop filter central value in wavenumber [ 125 ]")
    parser.add_argument("-o", type=int, default=2,
                        help="Order of the filter [2]")
    return parser.parse_args()

def butter_bandrej(cutoffs, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoffs = np.array(cutoffs) / nyq
    b, a = butter(order, normal_cutoffs, btype='bandstop', analog=False)
    return b, a

def butter_bandrej_filtfilt(data, cutoffs, fs, order=2):
    b, a = butter_bandrej(cutoffs, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_data(k0, A0, Ar, v2, cut0, cuts, outfile):
    import os
    import matplotlib.pyplot as plt
    pdf = os.path.splitext(outfile)[0]+'.pdf'
    fontsize = 6
    #plt.figure(figsize=(9,7))
    plt.subplot(2,2,1)
    j = np.argmin(np.abs(v[:,0] - 4000))
    plt.plot(k0,A0)
    plt.plot(k0,Ar)
    plt.xlabel('Wavenumber [1/cm]')
    plt.ylabel('Intensity')
    plt.legend(['Data','Filtered'], fontsize=fontsize)

    plt.subplot(2,2,2)
    plt.plot(k0,A0)
    plt.plot(k0,Ar)

    i0 = np.argmin(np.abs(k0 - 4000))
    i1 = np.argmin(np.abs(k0 - 6000))
    vmin,vmax = np.min(A0[i0:i1]), np.max(A0[i0:i1])

    plt.xlim(6000,4000)
    plt.ylim(vmin,vmax)
    plt.xlabel('Wavenumber [1/cm]')
    plt.ylabel('Intensity');

    k,A,As = v2[:,0],v2[:,1],v2[:,2]
    Af = np.fft.fft(A)
    Asf = np.fft.fft(As)
    f = np.fft.fftfreq(len(A), np.min(np.diff(k)))
    plt.subplot(2,2,3)
    plt.loglog(f[:len(A)//2], np.abs(Af[:len(A)//2]))
    plt.loglog(f[:len(A)//2], np.abs(Asf[:len(A)//2]))
    plt.axvline(1/cut0,color='gray')
    plt.xlabel('Wavelength [cm]')
    plt.ylabel('Amplitude')

    plt.subplot(2,2,4)
    plt.semilogx(f[:len(A)//2], np.abs(Af[:len(A)//2]))
    plt.semilogx(f[:len(A)//2], np.abs(Asf[:len(A)//2]))

    plt.axvline(1./cut0,color='red')
    for cut in cuts:
        plt.axvline(1./cut,color='gray')
    freqs = [2e-3, 1e-1]
    plt.xlim(freqs)
    i0 = np.argmin(np.abs(f[:len(A)//2] - freqs[0]))
    i1 = np.argmin(np.abs(f[:len(A)//2] - freqs[1]))
    vmin = np.min(np.abs(Af[i0:i1]))
    vmax = np.max(np.abs(Af[i0:i1]))
    plt.ylim(vmin,vmax)
    plt.ylabel('Amplitude')
    plt.xlabel('Wavelength [cm]');

    plt.legend(['Spectrum','Spectrum Filtered','Center Freq.','Filter band'],
               fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(pdf)

if __name__ == '__main__':
    args = get_args()

    # Load the data
    v = np.loadtxt(args.csvfile, delimiter=',')

    # Determine the spacing of the data in Wavenumber
    dk = np.diff(v[:,0])

    # Resample the data to the smallest wavenumber
    dk0 = np.min(dk)
    k0, k1 = np.min(v[:,0]), np.max(v[:,0])
    k = np.arange(k0, k1, dk0)
    A = np.interp(k, v[:,0],v[:,1])

    cut0  = args.c  # Filter Center, Wavenumber 
    width = args.w  # Filter Width,  Wavenumber
    order = args.o  # Filter Order
    cuts  = np.array([width,-width]) + cut0

    # Filter the data
    As = butter_bandrej_filtfilt(A, 1./cuts, 1./dk0, order=order)

    # Resample the data back onto the original data
    Asr = np.interp(v[:,0], k, As)

    # Compose a header for the data file
    hdr = "Wavenumber, Intensity, Intensity Filtered at %.2f cm width %.2f order %d" % (args.c, args.w, args.o)
    # Append the filtered, resampled data to the original data
    v2 = np.hstack([v, Asr.reshape((-1,1))])
    # Output the data (with a header)
    np.savetxt(args.outfile, v2, delimiter=',', header=hdr)

    # Generate a plot if requested
    if args.p:
        plot_data(v[:,0], v[:,1], Asr, np.array([k,A,As]).T,
                  cut0, cuts, args.outfile)

