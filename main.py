from __future__ import print_function

import functools
import heapq
import logging
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import networks
import train_distilled_image
import utils
from base_options import options
from basics import evaluate_models, evaluate_steps, format_stepwise_results
from networks.utils import print_network
from utils.io import load_results, save_test_results


def train(state, model, epoch, optimizer):
    model.train()
    for it, (data, target) in enumerate(state.train_loader):
        data, target = data.to(state.device, non_blocking=True), target.to(state.device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if state.log_interval > 0 and it % state.log_interval == 0:
            log_str = 'Epoch: {:4d} ({:2.0f}%)\tTrain Loss: {: >7.4f}'.format(
                epoch, 100. * it / len(state.train_loader), loss.item())
            if it == 0 or (state.log_interval > 0 and it % state.log_interval == 0):
                acc, loss = evaluate_models(state, [model])
                log_str += '\tTest Acc: {: >5.2f}%\tTest Loss: {: >7.4f}'.format(acc.item() * 100, loss.item())
                model.train()
            logging.info(log_str)


def main(state):
    logging.info('mode: {}, phase: {}'.format(state.mode, state.phase))

    if state.mode == 'train':
        model_dir = state.get_model_dir()
        utils.mkdir(model_dir)
        start_idx = cur_idx = state.world_rank * state.local_n_nets
        end_idx = start_idx + state.local_n_nets
        if state.train_nets_type == 'loaded':
            logging.info('Loading checkpoints [{} ... {}) from {}'.format(
                start_idx, end_idx, model_dir))
        else:
            logging.info('Save checkpoints [{} ... {}) to {}'.format(
                start_idx, end_idx, model_dir))
        queue_size = 10  # heuristics
        while cur_idx < end_idx:
            next_cur_idx = min(end_idx, cur_idx + queue_size)
            models = networks.get_networks(state, N=(next_cur_idx - cur_idx))

            for n, model in enumerate(models, start=cur_idx):
                if n == start_idx:
                    print_network(model)
                logging.info('Train network {:04d}'.format(n))
                if state.train_nets_type == 'loaded':
                    model_path = os.path.join(model_dir, 'net_{:04d}'.format(n))
                    model.load_state_dict(torch.load(model_path, map_location=state.device))
                    logging.info('Loaded from {}'.format(model_path))

                optimizer = optim.Adam(model.parameters(), lr=state.lr, betas=(0.5, 0.999))
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=state.decay_epochs, gamma=state.decay_factor)
                for epoch in range(state.epochs):
                    scheduler.step()
                    train(state, model, epoch, optimizer)
                model_path = os.path.join(model_dir, 'net_{:04d}'.format(n))
                if state.train_nets_type == 'loaded':
                    model_path += '_{}'.format(state.dataset)
                torch.save(model.state_dict(), model_path)
            acc, loss = evaluate_models(state, models, test_all=True)
            desc = 'Test networks in [{} ... {})'.format(cur_idx, next_cur_idx)
            logging.info('{}:\tTest Acc: {: >5.2f}%\tTest Loss: {: >7.4f}'.format(desc, acc.mean() * 100, loss.mean()))
            cur_idx = next_cur_idx

    elif state.mode in ['distill_basic', 'distill_attack', 'distill_adapt']:
        # train models
        def load_train_models():
            if state.train_nets_type == 'unknown_init':
                model, = networks.get_networks(state, N=1)
                return [model for _ in range(state.local_n_nets)]
            elif state.train_nets_type == 'known_init':
                return networks.get_networks(state, N=state.local_n_nets)
            elif state.train_nets_type == 'loaded':
                models = networks.get_networks(state, N=state.local_n_nets)
                with state.pretend(phase='train'):  # in case test_nets_type == same_as_train
                    model_dir = state.get_model_dir()
                start_idx = state.world_rank * state.local_n_nets
                for n, model in enumerate(models, start_idx):
                    model_path = os.path.join(model_dir, 'net_{:04d}'.format(n))
                    model.load_state_dict(torch.load(model_path, map_location=state.device))
                logging.info('Loaded checkpoints [{} ... {}) from {}'.format(
                    start_idx, start_idx + state.local_n_nets, model_dir))
                return models
            else:
                raise ValueError("train_nets_type: {}".format(state.train_nets_type))

        # only construct when in training mode or test_nets_type == same_as_train
        if state.phase == 'train' or state.test_nets_type == 'same_as_train':
            state.models = load_train_models()

        # test models
        if state.test_nets_type == 'unknown_init':
            test_model, = networks.get_networks(state, N=1)
            state.test_models = [test_model for _ in range(state.local_test_n_nets)]
        elif state.test_nets_type == 'same_as_train':
            assert state.test_n_nets == state.n_nets, \
                "test_nets_type=same_as_train, expect test_n_nets=n_nets"
            state.test_models = state.models
        elif state.test_nets_type == 'loaded':
            state.test_models = networks.get_networks(state, N=state.local_test_n_nets)
            with state.pretend(phase='test'):
                model_dir = state.get_model_dir()   # get test models
            start_idx = state.world_rank * state.local_test_n_nets
            for n, test_model in enumerate(state.test_models, start_idx):
                model_path = os.path.join(model_dir, 'net_{:04d}'.format(n))
                test_model.load_state_dict(torch.load(model_path, map_location=state.device))
            logging.info('Loaded held-out checkpoints [{} ... {}) from {}'.format(
                start_idx, start_idx + state.local_test_n_nets, model_dir))

        if state.phase == 'train':
            logging.info('Train {} steps iterated for {} epochs'.format(state.distill_steps, state.distill_epochs))
            steps = train_distilled_image.distill(state, state.models)
            evaluate_steps(state, steps,
                           'distilled with {} steps and {} epochs'.format(state.distill_steps, state.distill_epochs),
                           test_all=True)
        elif state.phase == 'test':
            logging.info('')
            logging.info((
                'Test:\n'
                '\ttest_distilled_images:\t{}\n'
                '\ttest_distilled_lrs:\t{}\n'
                '\ttest_distill_epochs:\t{}\n'
                '\ttest_optmize_n_runs:\t{}\n'
                '\ttest_optmize_n_nets:\t{}\n'
                '\t{} time(s)'
            ).format(
                state.test_distilled_images,
                ' '.join(state.test_distilled_lrs),
                state.test_distill_epochs,
                state.test_optimize_n_runs,
                state.test_optimize_n_nets,
                state.test_n_runs))
            logging.info('')

            loaded_steps = load_results(state, device=state.device)  # loaded

            if state.test_distilled_images == 'loaded':
                unique_data_label = [s[:-1] for s in loaded_steps[:state.distill_steps]]

                def get_data_label(state):
                    return [x for _ in range(state.distill_epochs) for x in unique_data_label]

            elif state.test_distilled_images == 'random_train':
                get_data_label = utils.baselines.random_train
            elif state.test_distilled_images == 'average_train':
                avg_images = None

                def get_data_label(state):
                    nonlocal avg_images
                    if avg_images is None:
                        avg_images = utils.baselines.average_train(state)
                    return avg_images

            elif state.test_distilled_images == 'kmeans_train':
                get_data_label = utils.baselines.kmeans_train
            else:
                raise NotImplementedError('test_distilled_images: {}'.format(state.test_distilled_images))

            # get lrs
            # allow for passing multiple options
            lr_meth = state.test_distilled_lrs[0]

            if lr_meth == 'nearest_neighbor':
                assert state.mode == 'distill_basic', 'nearest_neighbor test only supports distill_basic'
                assert state.test_distill_epochs is None, 'nearest_neighbor test expects unset test_distill_epochs'
                assert state.test_optimize_n_runs is None, 'nearest_neighbor test expects unset test_optimize_n_runs'

                k = int(state.test_distilled_lrs[1])
                p = float(state.test_distilled_lrs[2])

                class TestRunner(object):
                    def __init__(self, state):
                        self.state = state

                    def run(self, test_idx, test_at_steps=None):
                        assert test_at_steps is None

                        logging.info(
                            'Test #{} nearest neighbor classification with k={} and {}-norm'.format(test_idx, k, p))

                        state = self.state

                        with state.pretend(distill_epochs=1):
                            ref_data_label = tuple(get_data_label(state))

                        ref_flat_data = torch.cat([d for d, _ in ref_data_label], 0).flatten(1)
                        ref_label = torch.cat([l for _, l in ref_data_label], 0)

                        assert k <= ref_label.size(0), (
                            'k={} is greater than the number of data {}. '
                            'Set k to the latter').format(k, ref_label.size(0))

                        total = np.array(0, dtype=np.int64)
                        corrects = np.array(0, dtype=np.int64)
                        for data, target in state.test_loader:
                            data = data.to(state.device, non_blocking=True)
                            target = target.to(state.device, non_blocking=True)
                            dists = torch.norm(
                                data.flatten(1)[:, None, ...] - ref_flat_data,
                                dim=2, p=p
                            )
                            if k == 1:
                                argmin_dist = dists.argmin(dim=1)
                                pred = ref_label[argmin_dist]
                                del argmin_dist
                            else:
                                _, argmink_dist = torch.topk(dists, k, dim=1, largest=False, sorted=False)
                                labels = ref_label[argmink_dist]
                                counts = [torch.bincount(l, minlength=state.num_classes) for l in labels]
                                counts = torch.stack(counts, 0)
                                pred = counts.argmax(dim=1)
                                del argmink_dist, labels, counts
                            corrects += (pred == target).sum().item()
                            total += data.size(0)

                        at_steps = torch.ones(1, dtype=torch.long, device=state.device)
                        acc = torch.as_tensor(corrects / total, device=state.device).view(1, 1)   # STEP x MODEL
                        loss = torch.full_like(acc, utils.nan)  # STEP x MODEL
                        return (at_steps, acc, loss)

                    def num_steps(self):
                        return 1

            else:
                if lr_meth == 'loaded':
                    assert state.test_distill_epochs is None

                    def get_lrs(state):
                        return tuple(s[-1] for s in loaded_steps)

                elif lr_meth == 'fix':
                    val = float(state.test_distilled_lrs[1])

                    def get_lrs(state):
                        n_steps = state.distill_steps * state.distill_epochs
                        return torch.full((n_steps,), val, device=state.device).unbind()

                else:
                    raise NotImplementedError('test_distilled_lrs first: {}'.format(lr_meth))

                if state.test_optimize_n_runs is None:
                    class StepCollection(object):
                        def __init__(self, state):
                            self.state = state

                        def __getitem__(self, test_idx):
                            steps = []
                            for (data, label), lr in zip(get_data_label(self.state), get_lrs(self.state)):
                                steps.append((data, label, lr))
                            return steps
                else:
                    assert state.test_optimize_n_runs >= state.test_n_runs

                    class StepCollection(object):
                        @functools.total_ordering
                        class Step(object):
                            def __init__(self, step, acc):
                                self.step = step
                                self.acc = acc

                            def __lt__(self, other):
                                return self.acc < other.acc

                            def __eq__(self, other):
                                return self.acc == other.acc

                        def __init__(self, state):
                            self.state = state
                            self.good_steps = []  # min heap
                            logging.info('Start optimizing evaluated steps...')
                            for run_idx in range(state.test_optimize_n_runs):
                                if state.test_nets_type == 'unknown_init':
                                    subtest_nets = [state.test_models[0] for _ in range(state.test_optimize_n_nets)]
                                else:
                                    with state.pretend(local_n_nets=state.test_optimize_n_nets):
                                        with utils.logging.disable(logging.INFO):
                                            subtest_nets = load_train_models()
                                with state.pretend(test_models=subtest_nets, test_loader=state.train_loader):
                                    steps = []
                                    for (data, label), lr in zip(get_data_label(self.state), get_lrs(self.state)):
                                        steps.append((data, label, lr))
                                    res = evaluate_steps(state, steps, '', '', test_all=False,
                                                         test_at_steps=[len(steps)], log_results=False)
                                    acc = self.acc(res)
                                    elem = StepCollection.Step(steps, acc)
                                    if len(self.good_steps) < state.test_n_runs:
                                        heapq.heappush(self.good_steps, elem)
                                    else:
                                        heapq.heappushpop(self.good_steps, elem)
                                    logging.info((
                                        '\tOptimize run {:> 3}:\tAcc on training set {: >5.2f}%'
                                        '\tBoundary Acc {: >5.2f}%'
                                    ).format(run_idx, acc * 100, self.good_steps[0].acc * 100))
                            logging.info('done')

                        def acc(self, res):
                            state = self.state
                            if state.mode != 'distill_attack':
                                return res[1].mean().item()
                            else:
                                return res[1][:, 1].mean().item()

                        def __getitem__(self, test_idx):
                            return self.good_steps[test_idx].step

                class TestRunner(object):  # noqa F811
                    def __init__(self, state):
                        self.state = state
                        if state.test_distill_epochs is None:
                            self.test_distill_epochs = state.distill_epochs
                        else:
                            self.test_distill_epochs = state.test_distill_epochs
                        with state.pretend(distill_epochs=self.test_distill_epochs):
                            self.stepss = StepCollection(state)

                    def run(self, test_idx, test_at_steps=None):
                        with self.state.pretend(distill_epochs=self.test_distill_epochs):
                            steps = self.stepss[test_idx]  # before seeding!
                            with self.seed(self.state.seed + 1 + test_idx):
                                return evaluate_steps(
                                    self.state, steps,
                                    'Test #{}'.format(test_idx), '({}) images & ({}) lrs'.format(
                                        self.state.test_distilled_images, ' '.join(state.test_distilled_lrs)
                                    ), test_all=True, test_at_steps=test_at_steps)

                    @contextmanager
                    def seed(self, seed):
                        cpu_rng = torch.get_rng_state()
                        cuda_rng = torch.cuda.get_rng_state(self.state.device)
                        torch.random.default_generator.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        yield
                        torch.set_rng_state(cpu_rng)
                        torch.cuda.set_rng_state(cuda_rng, self.state.device)

                    def num_steps(self):
                        return self.state.distill_steps * self.test_distill_epochs

            # run tests
            test_runner = TestRunner(state)
            cache_init_res = state.test_nets_type != 'unknown_init'
            ress = []
            for idx in range(state.test_n_runs):
                if cache_init_res and idx > 0:
                    test_at_steps = [-1]
                else:
                    test_at_steps = None
                res = test_runner.run(idx, test_at_steps)
                if cache_init_res:
                    if idx == 0:
                        assert res[0][0].item() == 0
                    else:
                        cached = ress[0]
                        res = (cached[0],
                               torch.cat([cached[1][:1], res[1]], 0),
                               torch.cat([cached[2][:1], res[2]], 0))
                ress.append(res)
            # See NOTE [ Evaluation Result Format ] for output format
            if state.test_n_runs == 1:
                results = ress[0]
            else:
                results = (
                    ress[0][0],                          # at_steps
                    torch.cat([v[1] for v in ress], 1),  # accs
                    torch.cat([v[2] for v in ress], 1),  # losses
                )
            logging.info('')
            # Use dummy learning rates to print summary
            steps = [(None, None, np.array(utils.nan)) for _ in range(test_runner.num_steps())]
            test_desc = '({}) images & ({}) lrs'.format(state.test_distilled_images, ' '.join(state.test_distilled_lrs))
            logging.info(format_stepwise_results(state, steps, 'Summary with ' + test_desc, results))
            save_test_results(state, results)
            logging.info('')
        else:
            raise ValueError('phase: {}'.format(state.phase))

    else:
        raise NotImplementedError('unknown mode: {}'.format(state.mode))


if __name__ == '__main__':
    try:
        main(options.get_state())
    except Exception:
        logging.exception("Fatal error:")
        raise
