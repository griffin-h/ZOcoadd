import numpy as np

def stack_images(images, psfs, variances, flux_zps):
    '''Coadd a list of images using the Zackay & Ofek (2017) algorithm'''
    stacked_fft = np.zeros(images[0].shape)
    for flux_zp, psf, variance, image in zip(flux_zps, psfs, variances, images):
        psf_fft = np.fft.fft2(psf)
        image_fft = np.fft.fft2(image)
        stacked_fft += flux_zp / variance * psf_fft * image_fft
    stacked = np.fft.ifft2(stacked_fft)
    return stacked
