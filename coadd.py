import numpy as np
from astropy.io import fits
from argparse import ArgumentParser

def stack_images(images, psfs, variances, flux_zps):
    '''Coadd a list of images using the Zackay & Ofek (2017) algorithm'''
    stacked_fft = np.zeros(images[0].shape)
    for flux_zp, psf, variance, image in zip(flux_zps, psfs, variances, images):
        psf_fft = np.fft.fft2(psf)
        image_fft = np.fft.fft2(image)
        stacked_fft += flux_zp / variance * psf_fft * image_fft
    stacked = np.fft.ifft2(stacked_fft)
    return stacked

def normalize_stacked_image(stacked, psfs, variances, flux_zps, flux_units=False):
    '''Normalize a coadded image to have units of standard deviations (default) or flux'''
    norm_factor = np.zeros(stacked.shape)
    for flux_zp, psf, variance in zip(flux_zps, variances, psfs):
        norm_factor += flux_zp**2 / variance * psf @ psf
    if flux_units:
        stacked_norm = stacked / norm_factor
    else:
        stacked_norm = stacked / norm_factor**0.5
    return stacked_norm

if __name__ == '__main__':
    parser = ArgumentParser(description='Coadd a list of images using the Zackay & Ofek (2017) algorithm')
    parser.add_argument('images', nargs='+', help='filenames to stack')
    parser.add_argument('--psf', nargs='+', help='filenames of PSF images')
    parser.add_argument('--zp-keyword', help='header keyword containing zero point')
    parser.add_argument('--sky-keyword', help='header keyword containing standard deviation of sky')
    parser.add_argument('--flux-units', action='store_true', help='return stacked image in flux units, rather than standard deviations')
    parser.add_argument('--output', help='output filename')
    args = parser.parse_args()
    
    images = [fits.getdata(fn) for fn in args.images]
    headers = [fits.getheader(fn) for fn in args.images]
    flux_zps = [hdr[args.zp_keyword] for hdr in headers]
    variances = [hdr[args.sky_keyword]**2 for hdr in headers]
    psfs = [fits.getdata(fn) for fn in args.psf]
    
    stacked = stack_images(images, psfs, variances, flux_zps)
    stacked_norm = normalize_stacked_image(stacked, psfs, variances, flux_zps, args.flux_units)
    
    fits.writeto(args.output, stacked_norm)
