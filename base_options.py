import argparse
import logging
import math
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import yaml

import datasets
import utils


class State(object):
    class UniqueNamespace(argparse.Namespace):
        def __init__(self, requires_unique=True):
            self.__requires_unique = requires_unique
            self.__set_value = {}

        def requires_unique(self):
            return self.__requires_unique

        def mark_set(self, name, value):
            if self.__requires_unique and name in self.__set_value:
                raise argparse.ArgumentTypeError(
                    "'{}' appears several times: {}, {}.".format(
                        name, self.__set_value[name], value))
            self.__set_value[name] = value

    __inited = False

    def __init__(self, opt=None):
        if opt is None:
            self.opt = UniqueNamespace()
        else:
            if isinstance(opt, argparse.Namespace):
                opt = vars(opt)
            self.opt = argparse.Namespace(**opt)
        self.extras = {}
        self.__inited = True
        self._output_flag = True

    def __setattr__(self, k, v):
        if not self.__inited:
            return super(State, self).__setattr__(k, v)
        else:
            self.extras[k] = v

    def __getattr__(self, k):
        if k in self.extras:
            return self.extras[k]
        elif k in self.opt:
            return getattr(self.opt, k)
        raise AttributeError(k)

    def copy(self):
        return argparse.Namespace(**self.merge())

    def get_output_flag(self):
        return self._output_flag

    @contextmanager
    def pretend(self, **kwargs):
        saved = {}
        for key, val in kwargs.items():
            if key in self.extras:
                saved[key] = self.extras[key]
            setattr(self, key, val)
        yield
        for key, val in kwargs.items():
            self.pop(key)
            if key in saved:
                self.extras[key] = saved[key]

    def set_output_flag(self, val):
        self._output_flag = val

    def pop(self, k, default=None):
        return self.extras.pop(k, default)

    def clear(self):
        self.extras.clear()

    # returns a single dict containing both opt and extras
    def merge(self, public_only=False):
        vs = vars(self.opt).copy()
        vs.update(self.extras)
        if public_only:
            for k in tuple(vs.keys()):
                if k.startswith('_'):
                    vs.pop(k)
        return vs

    def get_base_directory(self):
        vs = self.merge()
        opt = argparse.Namespace(**vs)
        if opt.expr_name_format is not None:
            assert len(self.expr_name_format) > 0
            dirs = [fmt.format(**vs) for fmt in opt.expr_name_format]
        else:
            if opt.train_nets_type != 'loaded':
                train_nets_str = '{},{}'.format(opt.init, opt.init_param)
            else:
                train_nets_str = 'loaded,{}'.format(opt.n_nets)

            name = 'arch({},{})_distillLR{}_E({},{},{})_lr{}_B{}x{}x{}'.format(
                opt.arch, train_nets_str, str(opt.distill_lr),
                opt.epochs, opt.decay_epochs, str(opt.decay_factor), str(opt.lr),
                opt.distilled_images_per_class_per_step, opt.distill_steps, opt.distill_epochs)
            if opt.sample_n_nets > 1:
                name += '_{}nets'.format(opt.sample_n_nets)
            name += '_train({})'.format(opt.train_nets_type)
            if opt.dropout:
                name += '_dropout'
            dirs = [opt.mode, opt.dataset, name]
        return os.path.join(opt.results_dir, *dirs)

    def get_load_directory(self):
        return self.get_base_directory()

    def get_save_directory(self):
        base_dir = self.get_base_directory()
        if self.phase != 'train':
            base_dir = os.path.join(base_dir, 'test')
            subdir = self.get_test_subdirectory()
            if subdir is not None and subdir != '':
                base_dir = os.path.join(base_dir, subdir)
        return base_dir

    def get_test_subdirectory(self):
        if self.test_name_format is not None:
            assert len(self.test_name_format) > 0
            vs = self.merge()
            return self.test_name_format.format(**vs)
        else:
            return 'nRun{}_nNet{}_nEpoch{}_image_{}_lr_{}{}'.format(
                self.test_n_runs, self.test_n_nets, self.test_distill_epochs,
                self.test_distilled_images, self.test_distilled_lrs[0],
                '' if len(self.test_distilled_lrs) == 1 else '({})'.format('_'.join(self.test_distilled_lrs[1:])))

    def get_model_dir(self):
        vs = vars(self.opt).copy()
        vs.update(self.extras)
        opt = argparse.Namespace(**vs)
        model_dir = opt.model_dir
        arch = opt.arch
        if opt.mode == 'distill_adapt':
            dataset = opt.source_dataset
        else:
            dataset = opt.dataset
        if self.model_subdir_format is not None and self.model_subdir_format != '':
            subdir = self.model_subdir_format.format(**vs)
        else:
            subdir = os.path.join('{:s}_{:s}_{:s}_{}'.format(dataset, arch, opt.init, opt.init_param))
        return os.path.join(model_dir, subdir, opt.phase)


class BaseOptions(object):
    def __init__(self):
        # argparse utils

        def comp(type, op, ref):
            op = getattr(type, '__{}__'.format(op))

            def check(value):
                ivalue = type(value)
                if not op(ivalue, ref):
                    raise argparse.ArgumentTypeError("expected value {} {}, but got {}".format(op, ref, value))
                return ivalue

            return check

        def int_gt(i):
            return comp(int, 'gt', i)

        def float_gt(i):
            return comp(float, 'gt', i)

        pos_int = int_gt(0)
        nonneg_int = int_gt(-1)
        pos_float = float_gt(0)

        def get_unique_action_cls(actual_action_cls):
            class UniqueSetAttrAction(argparse.Action):
                def __init__(self, *args, **kwargs):
                    self.subaction = actual_action_cls(*args, **kwargs)

                def __call__(self, parser, namespace, values, option_string=None):
                    if isinstance(namespace, State.UniqueNamespace):
                        requires_unique = namespace.requires_unique()
                    else:
                        requires_unique = False
                    if requires_unique:
                        namespace.mark_set(self.subaction.dest, values)
                    self.subaction(parser, namespace, values, option_string)

                def __getattr__(self, name):
                    return getattr(self.subaction, name)

            return UniqueSetAttrAction

        self.parser = parser = argparse.ArgumentParser(description='PyTorch Dataset Distillation')

        action_registry = parser._registries['action']
        for name, action_cls in action_registry.items():
            action_registry[name] = get_unique_action_cls(action_cls)

        parser.add_argument('--batch_size', type=pos_int, default=1024,
                            help='input batch size for training (default: 1024)')
        parser.add_argument('--test_batch_size', type=pos_int, default=1024,
                            help='input batch size for testing (default: 1024)')
        parser.add_argument('--test_niter', type=pos_int, default=1,
                            help='max number of batches to test (default: 1)')
        parser.add_argument('--epochs', type=pos_int, default=400, metavar='N',
                            help='number of total epochs to train (default: 400)')
        parser.add_argument('--decay_epochs', type=pos_int, default=40, metavar='N',
                            help='period of weight decay (default: 40)')
        parser.add_argument('--decay_factor', type=pos_float, default=0.5, metavar='N',
                            help='weight decay multiplicative factor (default: 0.1)')
        parser.add_argument('--lr', type=pos_float, default=0.01, metavar='LR',
                            help='learning rate used to actually learn stuff (default: 0.01)')
        parser.add_argument('--init', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal|zero|default]')
        parser.add_argument('--init_param', type=float, default=1.,
                            help='network initialization param: gain, std, etc.')
        parser.add_argument('--base_seed', type=int, default=1, metavar='S',
                            help='base random seed (default: 1)')
        parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--checkpoint_interval', type=int, default=10, metavar='N',
                            help='checkpoint interval (epoch)')
        parser.add_argument('--dataset', type=str, default='MNIST',
                            help='dataset: MNIST | Cifar10 | PASCAL_VOC | CUB200')
        parser.add_argument('--source_dataset', type=str, default=None,
                            help='dataset: MNIST | Cifar10 | PASCAL_VOC | CUB200')
        parser.add_argument('--dataset_root', type=str, default=None,
                            help='dataset root')
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='results directory')
        parser.add_argument('--arch', type=str, default='LeNet',
                            help='architecture: LeNet | AlexNet | etc.')
        parser.add_argument('--mode', type=str, default='distill_basic',
                            help='mode: train | distill_basic | distill_attack | distill_adapt ')
        parser.add_argument('--distill_lr', type=float, default=0.02,
                            help='learning rate to perform GD with distilled images PER STEP (default: 0.02)')
        parser.add_argument('--model_dir', type=str, default='./models/',
                            help='directory storing trained models')
        parser.add_argument('--model_subdir_format', type=str, default=None,
                            help='directory storing trained models')
        parser.add_argument('--train_nets_type', type=str, default='unknown_init',
                            help='[ unknown_init | known_init | loaded ]')  # add things like P(reset) = 0.7?
        parser.add_argument('--test_nets_type', type=str, default='unknown_init',
                            help='[ unknown_init | same_as_train | loaded ]')
        parser.add_argument('--dropout', action='store_true',
                            help='if set, use dropout')
        parser.add_argument('--distilled_images_per_class_per_step', type=pos_int, default=1,
                            help='use #batch_size distilled images for each class in each step')
        parser.add_argument('--distill_steps', type=pos_int, default=10,
                            help='Iterative distillation, use #num_steps * #batch_size * #classes distilled images. '
                                 'See also --distill_epochs. The total number '
                                 'of steps is distill_steps * distill_epochs.')
        parser.add_argument('--distill_epochs', type=pos_int, default=3,
                            help='how many times to repeat all steps 1, 2, 3, 1, 2, 3, ...')
        parser.add_argument('--n_nets', type=int, default=1,
                            help='# random nets')
        parser.add_argument('--sample_n_nets', type=pos_int, default=None,
                            help='sample # nets for each iteration. Default: equal to n_nets')
        parser.add_argument('--device_id', type=comp(int, 'ge', -1), default=0, help='device id, -1 is cpu')
        parser.add_argument('--image_dpi', type=pos_int, default=80,
                            help='dpi for visual image generation')
        parser.add_argument('--attack_class', type=nonneg_int, default=0,
                            help='when mode is distill_attack, the objective is to predict this class as target_class')
        parser.add_argument('--target_class', type=nonneg_int, default=1,
                            help='when mode is distill_attack, the objective is to predict forget class as this class')
        parser.add_argument('--expr_name_format', nargs='+', default=None, type=str,
                            help='expriment save dir name format. multiple values means nested folders')
        parser.add_argument('--phase', type=str, default='train',
                            help='phase')
        parser.add_argument('--test_distill_epochs', nargs='?', type=pos_int, default=None,
                            help='IN TEST, how many times to repeat all steps 1, 2, 3, 1, 2, 3, ...'
                                 'Defaults to distill_epochs.')
        parser.add_argument('--test_n_runs', type=pos_int, default=1,
                            help='do num test (no training), each test generates new distilled image, label, and lr')
        parser.add_argument('--test_n_nets', type=pos_int, default=1,
                            help='# reset model in test to get average performance, useful with unknown init')
        parser.add_argument('--test_distilled_images', default='loaded', type=str,
                            help='which distilled images to test [ loaded | random_train | kmeans_train ]')
        parser.add_argument('--test_distilled_lrs', default=['loaded'], nargs='+', type=str,
                            help='which distilled lrs to test [ loaded | fix [lr] | nearest_neighbor [k] [p] ]')
        parser.add_argument('--test_optimize_n_runs', default=None, type=pos_int,
                            help='if set, evaluate test_optimize_n_runs sets of test images, label and lr on '
                                 'test_optimize_n_nets training networks, and pick the best test_n_runs sets.'
                                 'Default: None.')
        parser.add_argument('--test_optimize_n_nets', default=20, type=pos_int,
                            help='number of networks used to optimize data. See doc for test_optimize_n_runs.')
        parser.add_argument('--num_workers', type=nonneg_int, default=8,
                            help='number of data loader workers')
        parser.add_argument('--no_log', action='store_true',
                            help='if set, will not log into file')
        parser.add_argument('--log_level', type=str, default='INFO',
                            help='logging level, e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL')
        parser.add_argument('--test_name_format', nargs='+', type=str, default=None,
                            help='test save subdir name format. multiple values means nested folders')
        parser.add_argument('--world_size', nargs='?', type=comp(int, 'ge', 1), default=1,
                            help='if > 1, word size used for distributed training in reverse mode with NCCL. '
                                 'This will read an environ variable representing the process RANK, and several '
                                 'others needed to initialize the process group, which can '
                                 'be either MASTER_PORT & MASTER_ADDR, or INIT_FILE. '
                                 'Then it stores the values in state as "distributed_master_addr", '
                                 '"distributed_master_port", etc. Only rank 0 process writes checkpoints. ')

    def get_dummy_state(self, *cmdargs, yaml_file=None, **opt_pairs):
        if yaml_file is None:
            # Use default Namespace (not UniqueNamespace) because dummy state may
            # want to overwrite things using `cmdargs`
            opt = self.parser.parse_args(args=list(cmdargs), namespace=argparse.Namespace())
        else:
            with open(yaml_file, 'r') as f:
                opt = yaml.load(f)
        state = State(opt)
        valid_keys = set(state.merge().keys())
        for k in opt_pairs:
            # TODO: check against argparse instead
            assert k in valid_keys, "'{}' is not a valid key".format(k)
        state.extras.update(opt_pairs)
        return self.set_state(state, dummy=True)

    def get_state(self):
        if hasattr(self, 'state'):
            return self.state

        logging.getLogger().setLevel(logging.DEBUG)
        self.opt, unknowns = self.parser.parse_known_args(namespace=State.UniqueNamespace())
        assert len(unknowns) == 0, 'Unexpected args: {}'.format(unknowns)
        self.state = State(self.opt)
        return self.set_state(self.state)

    def set_state(self, state, dummy=False):
        if state.opt.sample_n_nets is None:
            state.opt.sample_n_nets = state.opt.n_nets

        base_dir = state.get_base_directory()
        save_dir = state.get_save_directory()

        state.opt.start_time = time.strftime(r"%Y-%m-%d %H:%M:%S")

        # Usually only rank 0 can write to file (except logging, training many
        # nets, etc.) so let's set that flag before everything
        state.opt.distributed = state.world_size > 1
        if state.distributed:
            # read from os.environ
            def set_val_from_environ(key, save_obj, ty=str, fmt="distributed_{}"):
                if key not in os.environ:
                    raise ValueError("expected environment variable {} to be set when using distributed".format(key))
                setattr(save_obj, fmt.format(key.lower()), ty(os.environ[key]))

            set_val_from_environ("RANK", state, int, "world_rank")

            state.opt.distributed_file_init = 'INIT_FILE' in os.environ
            if state.opt.distributed_file_init:
                def absolute_path(val):
                    return os.path.abspath(os.path.expanduser(str(val)))

                set_val_from_environ("INIT_FILE", state.opt, ty=absolute_path)
            else:
                os.environ['WORLD_SIZE'] = str(state.world_size)
                set_val_from_environ("MASTER_ADDR", state.opt)
                set_val_from_environ("MASTER_PORT", state.opt, int)

            state.set_output_flag(state.world_rank == 0)
        else:
            state.world_rank = 0
            state.set_output_flag(not dummy)

        if not dummy:
            utils.mkdir(save_dir)

            # First thing: set logging config:
            if not state.opt.no_log:
                log_filename = 'output'
                if state.distributed:
                    log_filename += '_rank{:02}'.format(state.world_rank)
                log_filename += '.log'
                state.opt.log_file = os.path.join(save_dir, log_filename)
            else:
                state.opt.log_file = None

            state.opt.log_level = state.opt.log_level.upper()

            if state.distributed:
                logging_prefix = 'rank {:02d} / {:02d} - '.format(state.world_rank, state.world_size)
            else:
                logging_prefix = ''
            utils.logging.configure(state.opt.log_file, getattr(logging, state.opt.log_level),
                                    prefix=logging_prefix)

            logging.info("=" * 40 + " " + state.opt.start_time + " " + "=" * 40)
            logging.info('Base directory is {}'.format(base_dir))

            if state.phase == 'test' and not os.path.isdir(base_dir):
                logging.warning("Base directory doesn't exist")

        _, state.opt.dataset_root, state.opt.nc, state.opt.input_size, state.opt.num_classes, \
            state.opt.dataset_normalization, state.opt.dataset_labels = datasets.get_info(state)

        # Write yaml
        yaml_str = yaml.dump(state.merge(public_only=True), default_flow_style=False, indent=4)
        logging.info("Options:\n\t" + yaml_str.replace("\n", "\n\t"))

        if state.get_output_flag():
            yaml_name = os.path.join(save_dir, 'opt.yaml')
            if os.path.isfile(yaml_name):
                old_opt_dir = os.path.join(save_dir, 'old_opts')
                utils.mkdir(old_opt_dir)
                with open(yaml_name, 'r') as f:
                    # ignore unknown ctors
                    yaml.add_multi_constructor('', lambda loader, suffix, node: None)
                    old_yaml = yaml.load(f)  # this is a dict
                old_yaml_time = old_yaml.get('start_time', 'unknown_time')
                for c in ':-':
                    old_yaml_time = old_yaml_time.replace(c, '_')
                old_yaml_time = old_yaml_time.replace(' ', '__')
                old_opt_new_name = os.path.join(old_opt_dir, 'opt_{}.yaml'.format(old_yaml_time))
                try:
                    os.rename(yaml_name, old_opt_new_name)
                    logging.warning('{} already exists, moved to {}'.format(yaml_name, old_opt_new_name))
                except FileNotFoundError:
                    logging.warning((
                        '{} already exists, tried to move to {}, but failed, '
                        'possibly due to other process having already done it'
                    ).format(yaml_name, old_opt_new_name))
                    pass

            with open(yaml_name, 'w') as f:
                f.write(yaml_str)

        # FROM HERE, we have saved options into yaml,
        #            can start assigning objects to opt, and
        #            modify the values for process-specific things
        def assert_divided_by_world_size(key, strict=True):
            val = getattr(state, key)
            if strict:
                assert val % state.world_size == 0, \
                    "expected {}={} to be divisible by the world size={}".format(key, val, state.world_size)
                val = val // state.world_size
            else:
                val = math.ceil(val / state.world_size)
            setattr(state, 'local_{}'.format(key), val)

        assert_divided_by_world_size('n_nets')

        if state.mode != 'train':
            assert_divided_by_world_size('test_n_nets')
            assert_divided_by_world_size('sample_n_nets')

        if state.device_id < 0:
            state.opt.device = torch.device("cpu")
        else:
            torch.cuda.set_device(state.device_id)
            state.opt.device = torch.device("cuda:{}".format(state.device_id))

        if not dummy:
            if state.device.type == 'cuda' and torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = True

            seed = state.base_seed
            if state.distributed:
                seed += state.world_rank
                logging.info("In distributed mode, use arg.seed + rank as seed: {}".format(seed))
            state.opt.seed = seed

            # torch.manual_seed will seed ALL GPUs.
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if not dummy and state.distributed:
            logging.info('Initializing distributed process group...')

            if state.distributed_file_init:
                dist.init_process_group("NCCL",
                                        init_method="file://{}".format(state.distributed_init_file),
                                        rank=state.world_rank,
                                        world_size=state.world_size)
            else:
                dist.init_process_group("NCCL", init_method="env://")

            utils.distributed.barrier()
            logging.info('done!')

            # Check command args consistency across ranks
            # Use a raw parsed dict because we assigned a bunch of things already
            # so this doesn't include things like seed (which can be rank-specific),
            # but includes base_seed.
            opt_dict = vars(self.parser.parse_args())
            opt_dict.pop('device_id')  # don't compare this
            bytes = yaml.dump(opt_dict, encoding='utf-8')
            bytes_storage = torch.ByteStorage.from_buffer(bytes)
            opt_tensor = torch.tensor((), dtype=torch.uint8).set_(bytes_storage).to(state.opt.device)
            for other, ts in enumerate(utils.distributed.all_gather_coalesced([opt_tensor])):
                other_t = ts[0]
                if not torch.equal(other_t, opt_tensor):
                    other_str = bytearray(other_t.cpu().storage().tolist()).decode(encoding="utf-8")
                    this_str = bytes.decode(encoding="utf-8")
                    raise ValueError(
                        "Rank {} opt is different from rank {}:\n".format(state.world_rank, other) +
                        utils.diff_str(this_str, other_str))

        # in case of downloading, to avoid race, let rank 0 download.
        if state.world_rank == 0:
            train_dataset = datasets.get_dataset(state, 'train')
            test_dataset = datasets.get_dataset(state, 'test')

        if not dummy and state.distributed:
            utils.distributed.barrier()

        if state.world_rank != 0:
            train_dataset = datasets.get_dataset(state, 'train')
            test_dataset = datasets.get_dataset(state, 'test')

        state.opt.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=state.batch_size,
            num_workers=state.num_workers, pin_memory=True, shuffle=True)

        state.opt.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=state.test_batch_size,
            num_workers=state.num_workers, pin_memory=True, shuffle=True)

        if not dummy:
            logging.info('train dataset size:\t{}'.format(len(train_dataset)))
            logging.info('test dataset size: \t{}'.format(len(test_dataset)))
            logging.info('datasets built!')

            state.vis_queue = utils.multiprocessing.FixSizeProcessQueue(2)
        return state


options = BaseOptions()
