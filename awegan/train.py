import time
import numpy as np

try:
    from .options import TrainOptions
    from .datasets import create_dataset
    from .models import create_model
    from .util.visualizer import Visualizer
except ImportError:
    from options import TrainOptions
    from datasets import create_dataset
    from models import create_model
    from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset
    print('the number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model with model parameters in `opt`
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.start_epoch, opt.start_epoch + opt.n_epochs + 1):
        epoch_since = time.time()
        iter_data_since = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()    # update learning rates in the beginning of every epoch
        for i, batch in enumerate(dataset):
            iter_since = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_since - iter_data_since

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(batch)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()    # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:
                save_result = False
                model.compute_visuals()
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_since) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

    print('end of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.start_epoch + opt.n_epochs, time.time() - epoch_since))
