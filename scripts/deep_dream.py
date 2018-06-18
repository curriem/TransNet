import keras
import time
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import argparse
import commands
import scipy.optimize as opt


parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('fits_path', type=str)
parser.add_argument('model_path', type=str)
parser.add_argument('scale_arg', type=int)
parser.add_argument('target_score', type=float)
parser.add_argument('run', type=int)
args = parser.parse_args()
fits_path = args.fits_path
model_path = args.model_path
scale_arg = args.scale_arg
target_score = args.target_score
run = args.run


def load_fits(fits_path):
    fits_im = pyfits.open(fits_path)
    fits_im = fits_im[0].data
    print fits_im.shape
    return fits_im


def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model


def add_jitter(im, scale):
    jitter = scale * np.random.randn(1, 32, 32, 3)
    im += jitter

    im[0, :, :, 0] = normalize_im(im[0, :, :, 0])
    im[0, :, :, 1] = normalize_im(im[0, :, :, 1])
    im[0, :, :, 2] = normalize_im(im[0, :, :, 2])

    return im


def normalize_im(im):
    im -= np.min(im)
    im /= np.max(im)
    return im


def save_im(im, score, save_path):
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.title('score = %.04f' % score)
    plt.savefig(save_path)
    plt.close()


def fun(stack):
    stack = stack.reshape((1, 32, 32, 3))
    stack[0, :, :, 0] = normalize_im(stack[0, :, :, 0])
    stack[0, :, :, 1] = normalize_im(stack[0, :, :, 1])
    stack[0, :, :, 2] = normalize_im(stack[0, :, :, 2])

    score = model.predict(stack)[0][0]
    return score


if __name__ == '__main__':

    start_time = time.time()

    scales = [1, 0.1, 0.01, 0.001]
    scale = scales[scale_arg]

    #commands.getoutput('rm ../deep_dream/*pdf')
    if fits_path.split('.')[-1] == 'npy':
        stack = np.load(fits_path)
    elif fits_path.split('.')[-1] == 'fits':
        fits_im = load_fits(fits_path)

        stack = np.stack((fits_im[:, :, 0],
                          fits_im[:, :, 1],
                          0.5*(fits_im[:, :, 0] 
                               + fits_im[:, :, 1])),
                         axis=-1)


        stack = np.array([stack])

    else:
        assert False

    model = load_model(model_path)
    print stack.shape
    save_im(stack[0, :, :, 0],
            0.06,
            '../deep_dream/test1.pdf')
    plt.figure()
    plt.imshow(stack[0, :, :, :])
    plt.savefig('color_sn.pdf')
    plt.show()
    stack = stack.flatten()
    res = opt.minimize(fun=fun, x0=stack, method='L-BFGS-B',
                       options={'eps':1e-2, 'disp':True,'gtol':1e-6,
                                'maxfun':25000})
    res_stack = res.x
    res_stack = res_stack.reshape((1, 32, 32, 3))
    save_im(res_stack[0, :, :, 0],
            1-res.fun,
            '../deep_dream/test3.pdf')

    plt.figure()
    plt.imshow(res_stack[0, :, :, :])
    plt.savefig('color_art.pdf')
    plt.show()

    #np.save('../deep_dream/dream_sn.npy', res_stack)
    '''
    old_sn_score = model.predict(old_stack)[0][0]
    print old_sn_score
    new_sn_score = 0
    save_im(old_stack[0, :, :, 0],
            old_sn_score,
            '../deep_dream/orig_im.pdf')
    loop_counter = 0
    iteration_num = []
    improvement_score = []
    while new_sn_score < target_score and loop_counter < 1000000:
        new_stack = add_jitter(old_stack, scale)
        new_sn_score = model.predict(new_stack)[0][0]
        if new_sn_score > old_sn_score:
            print 'took %i iterations to improve to %.04f' % (loop_counter,
                                                              new_sn_score)
            print '--- %s seconds ---' % (time.time() - start_time)
            iteration_num.append(loop_counter)
            improvement_score.append(new_sn_score)
            #save_im(new_stack[0, :, :, 0],
            #        new_sn_score,
            #        '../deep_dream/improved_im%i.pdf' % loop_counter)
            old_stack = new_stack
            old_sn_score = new_sn_score

        loop_counter += 1

    plt.figure()
    plt.plot(iteration_num, improvement_score)
    plt.xlabel('iteration number')
    plt.ylabel('improved score')
    plt.savefig('../deep_dream/progress_plot_%i_run%i.pdf' % (scale_arg,
                                                              run))
    plt.close()
    print 'new sn score is', old_sn_score
    save_im(old_stack[0, :, :, 0],
            old_sn_score,
            '../deep_dream/final_im_%i_run%i.pdf' % (scale_arg,
                                                 run))
                                                 '''
