import torch
import logging
import numpy as np


def get_baseline_label_for_one_step(state):
    label = torch.tensor(list(range(state.num_classes)), device=state.device)
    if state.mode == 'distill_attack':
        label[state.attack_class] = state.target_class
    label = label.repeat(state.distilled_images_per_class_per_step, 1)  # [[0, 1, 2, ...], [0, 1, 2, ...], ...]
    return label.t().reshape(-1)  # [0, 0, ..., 1, 1, ...]


def random_train(state):
    data_list = [[] for _ in range(state.num_classes)]
    needed = state.distill_steps * state.distilled_images_per_class_per_step
    counts = np.zeros((state.num_classes))
    for datas, labels in state.train_loader:
        for data, label in zip(datas, labels):
            label_id = label.item()
            if counts[label_id] < needed:
                counts[label_id] += 1
                data_list[label_id].append(data.to(state.device))
                if np.sum(counts) == needed * state.num_classes:
                    break
    steps = []
    label = get_baseline_label_for_one_step(state)
    for i in range(0, needed, state.distilled_images_per_class_per_step):
        data = sum((cd[i:(i + state.distilled_images_per_class_per_step)] for cd in data_list), [])
        data = torch.stack(data, 0)
        steps.append((data, label))
    return [s for _ in range(state.distill_epochs) for s in steps]


def average_train(state):
    sum_images = torch.zeros(
        state.num_classes, state.nc, state.input_size, state.input_size,
        device=state.device, dtype=torch.double)
    counts = torch.zeros(state.num_classes, dtype=torch.long)
    for data, label in state.train_loader:
        for i, (d, l) in enumerate(zip(data, label)):
            sum_images[l].add_(d.to(sum_images))
            counts[l] += 1
    mean_imgs = sum_images / counts[:, None, None, None].to(state.device, torch.double)
    mean_imgs = mean_imgs.to(torch.float)
    mean_imgs = mean_imgs.repeat(state.distilled_images_per_class_per_step, 1, 1, 1, 1)
    mean_imgs = mean_imgs.transpose(0, 1).flatten(end_dim=1)
    label = get_baseline_label_for_one_step(state)
    return [(mean_imgs, label) for _ in range(state.distill_epochs) for _ in range(state.distill_steps)]


def kmeans_train(state, p=2):
    k = state.distilled_images_per_class_per_step * state.distill_steps

    if k == 1:
        return average_train(state)

    cls_data = [[] for _ in range(state.num_classes)]

    for data, label in state.train_loader:
        for d, l in zip(data, label):
            cls_data[l.item()].append(d.flatten())
    cls_data = [torch.stack(coll, 0).to(state.device) for coll in cls_data]

    # kmeans++
    cls_centers = []
    for c in range(state.num_classes):
        c_center = torch.empty(k, state.nc * state.input_size * state.input_size, device=state.device)
        c_data = cls_data[c]
        # first is uniform
        c_center[0] = c_data[torch.randint(len(c_data), ()).item()]
        for i in range(1, k):
            assert p == 2
            dists_sq = (c_data[:, None, :] - c_center[:i]).pow(2).sum(dim=2)  # D x I
            weights = dists_sq.min(dim=1).values
            # A-res
            r = torch.rand_like(weights).pow(1 / weights)
            c_center[i] = c_data[r.argmax().item()]
        cls_centers.append(c_center)

    cls_centers = torch.stack(cls_centers, dim=0)
    cls_assign = [torch.full((coll.size(0),), -1, dtype=torch.long, device=state.device) for coll in cls_data]

    def iterate(n=1024):
        nonlocal cls_centers
        changed = torch.tensor(0, dtype=torch.long, device=state.device)
        cls_totals = torch.zeros_like(cls_centers)
        cls_counts = cls_totals.new_zeros(state.num_classes, k, dtype=torch.long)
        for c in range(state.num_classes):
            c_center = cls_centers[c]
            c_total = cls_totals[c]
            c_count = cls_counts[c]
            for d, a in zip(cls_data[c].split(n, dim=0), cls_assign[c].split(n, dim=0)):
                new_a = torch.norm(
                    d[:, None, :] - c_center,
                    dim=2, p=p,
                ).argmin(dim=1)
                c_total.index_add_(0, new_a, d)
                c_count.index_add_(0, new_a, c_count.new_ones(d.size(0)))
                changed += (a != new_a).sum()
                a.copy_(new_a)
            # keep empty clusters unchanged
            empty = (c_count == 0)
            nempty = empty.sum().item()
            if nempty > 0:
                logging.warning("{} empty cluster(s) found for class of index {} (kept unchanged)".format(nempty, c))
                c_count[empty] = 1
                c_total[empty] = c_center[empty]
        cls_centers = cls_totals / cls_counts.unsqueeze(2).to(cls_totals)
        return changed.item()

    logging.info('Compute {}-means with {}-norm ...'.format(k, p))
    changed = 1
    i = 0
    while changed > 0:
        changed = iterate()
        i += 1
        logging.info('\tIteration {:>3d}: {:>6d} samples changed cluster label'.format(i, changed))

    logging.info('done')

    label = get_baseline_label_for_one_step(state)

    per_step_imgs = cls_centers.view(
        state.num_classes, state.distill_steps, state.distilled_images_per_class_per_step, state.nc,
        state.input_size, state.input_size).transpose(0, 1).flatten(1, 2).unbind(0)

    return [(imgs, label) for _ in range(state.distill_epochs) for imgs in per_step_imgs]
