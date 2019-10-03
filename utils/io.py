import logging
import os
import warnings

import matplotlib
import numpy as np
import torch

import utils

matplotlib.use('agg')  # this needs to be before the next line
import matplotlib.pyplot as plt
import datasets


def _vis_results_fn(np_steps, distilled_images_per_class_per_step, dataset_info, arch, dpi,
                    vis_dir=None, vis_name_fmt='visuals_step{step:03d}',
                    cmap=None, supertitle=True, subtitle=True, fontsize=None,
                    reuse_axes=True):
    if vis_dir is None:
        logging.warning('Not saving because vis_dir is not given')
    else:
        vis_name_fmt += '.png'
        utils.mkdir(vis_dir)

    dataset, nc, input_size, mean, std, label_names = dataset_info

    N = len(np_steps[0][0])
    nrows = max(2, distilled_images_per_class_per_step)
    grid = (nrows, np.ceil(N / float(nrows)).astype(int))
    plt.rcParams["figure.figsize"] = (grid[1] * 1.5 + 1, nrows * 1.5 + 1)

    plt.close('all')
    fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1])
    axes = axes.flatten()
    if supertitle:
        fmts = [
            'Dataset: {dataset}',
            'Arch: {arch}',
        ]
        if len(np_steps) > 1:
            fmts.append('Step: {{step}}')
        if np_steps[0][-1] is not None:
            fmts.append('LR: {{lr:.4f}}')
        supertitle_fmt = ', '.join(fmts).format(dataset=dataset, arch=arch)

    plt_images = []
    first_run = True
    for i, (data, labels, lr) in enumerate(np_steps):
        for n, (img, label, axis) in enumerate(zip(data, labels, axes)):
            if nc == 1:
                img = img[..., 0]
            img = (img * std + mean).clip(0, 1)
            if first_run:
                plt_images.append(axis.imshow(img, interpolation='nearest', cmap=cmap))
            else:
                plt_images[n].set_data(img)
            if first_run:
                axis.axis('off')
                if subtitle:
                    axis.set_title('Label {}'.format(label_names[label]), fontsize=fontsize)
        if supertitle:
            if lr is not None:
                lr = lr.sum().item()
            plt.suptitle(supertitle_fmt.format(step=i, lr=lr), fontsize=fontsize)
            if first_run:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0, rect=[0, 0, 1, 0.95])
        fig.canvas.draw()
        if vis_dir is not None:
            plt.savefig(os.path.join(vis_dir, vis_name_fmt.format(step=i)), dpi=dpi)
        if reuse_axes:
            first_run = False
        else:
            fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1])
            axes = axes.flatten()
            plt.show()


def vis_results(state, steps, *args, immediate=False, **kwargs):
    if not state.get_output_flag():
        logging.warning('Skip visualize results because output_flag is False')
        return

    if isinstance(steps[0][0], torch.Tensor):
        steps = to_np(steps)

    _, _, nc, input_size, _, (mean, std), label_names = datasets.get_info(state)
    dataset_vis_info = (state.dataset, nc, input_size, np.array(mean), np.array(std), label_names)

    vis_args = (steps, state.distilled_images_per_class_per_step, dataset_vis_info, state.arch, state.image_dpi) + args

    if not immediate:
        state.vis_queue.enqueue(_vis_results_fn, *vis_args, **kwargs)
    else:
        _vis_results_fn(*vis_args, **kwargs)


def to_np(steps):
    if isinstance(steps[0][0], np.ndarray):  # noop if already ndarray
        return steps
    np_steps = []
    for data, label, lr in steps:
        np_data = data.detach().permute(0, 2, 3, 1).to('cpu').numpy()
        np_label = label.detach().to('cpu').numpy()
        if lr is not None:
            lr = lr.detach().cpu().numpy()
        np_steps.append((np_data, np_label, lr))
    return np_steps


def to_torch(np_steps, device):
    _t = np_steps[0][0]
    if isinstance(_t, torch.Tensor) and _t.device == device:  # noop if already tensor at correct device
        return np_steps
    steps = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step in np_steps:
            steps.append(tuple(torch.as_tensor(t, device=device) for t in step))
    return steps


def save_results(state, steps, visualize=True, subfolder=''):
    if not state.get_output_flag():
        logging.warning('Skip saving results because output_flag is False')
        return

    expr_dir = os.path.join(state.get_save_directory(), subfolder)
    utils.mkdir(expr_dir)
    save_data_path = os.path.join(expr_dir, 'results.pth')

    steps = [(d.detach().cpu(), l.detach().cpu(), lr) for (d, l, lr) in steps]
    if visualize:
        vis_results(state, steps, expr_dir)

    torch.save(steps, save_data_path)
    logging.info('Results saved to {}'.format(save_data_path))


def load_results(state, save_data_path=None, device=None):
    if save_data_path is None:
        expr_dir = state.get_load_directory()
        save_data_path = os.path.join(expr_dir, 'results.pth')
    device = device or state.device
    return to_torch(torch.load(save_data_path, map_location=device), device)


def save_test_results(state, results):
    assert state.phase != 'train'
    if not state.get_output_flag():
        logging.warning('Skip saving test results because output_flag is False')
        return

    test_dir = state.get_save_directory()
    utils.mkdir(test_dir)
    result_file = os.path.join(test_dir, 'results.pth')
    torch.save(results, os.path.join(test_dir, 'results.pth'))
    logging.info('Test results saved as {}'.format(result_file))
