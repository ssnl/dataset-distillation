import contextlib
import functools
import inspect
import pprint
import unittest
import warnings

import numpy as np
import torch

import networks
from base_options import options
from train_distilled_image import Trainer


def unittest_verbosity():
    """Return the verbosity setting of the currently running unittest
    program, or 0 if none is running.
    """
    frame = inspect.currentframe()
    while frame:
        self = frame.f_locals.get('self')
        if isinstance(self, unittest.TestProgram):
            return self.verbosity
        frame = frame.f_back
    return 0


def suppress_wranings(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    return wrapped


def format_intlist(intlist):
    return ", ".join("{:>2d}".format(x) for x in intlist)


class TestDistilledImageTrainer(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)

    @staticmethod
    def _test_params_invariance(self, state):
        models = networks.get_networks(state, 1)
        trainer = Trainer(state, models)
        model = trainer.models[0]

        ref_w = model.get_param(clone=True)

        rdata, rlabel = next(iter(state.train_loader))
        rdata = rdata.to(state.device, non_blocking=True)
        rlabel = rlabel.to(state.device, non_blocking=True)

        model.train()

        steps = trainer.get_steps()

        l, saved = trainer.forward(model, rdata, rlabel, steps)
        self.assertTrue(torch.equal(ref_w, model.get_param()))
        trainer.backward(model, rdata, rlabel, steps, saved)
        self.assertTrue(torch.equal(ref_w, model.get_param()))

    def test_params_invariance(self):
        state = options.get_dummy_state(dataset='Cifar10', arch='AlexCifarNet',
                                        distill_steps=10, distill_epochs=2)
        self._test_params_invariance(self, state)
        state = options.get_dummy_state(dataset='PASCAL_VOC', arch='AlexNet',
                                        distill_steps=2, distill_epochs=2)
        self._test_params_invariance(self, state)

    @staticmethod
    def _test_backward(self, state, eps=2e-8, atol=1e-5, rtol=1e-3, max_num_per_param=5):
        @contextlib.contextmanager
        def double_prec():
            saved_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.double)
            yield
            torch.set_default_dtype(saved_dtype)

        with double_prec():
            models = [m.to(torch.double) for m in networks.get_networks(state, 1)]
            trainer = Trainer(state, models)

            model = trainer.models[0]

            rdata, rlabel = next(iter(state.train_loader))
            rdata = rdata.to(state.device, torch.double, non_blocking=True)
            rlabel = rlabel.to(state.device, non_blocking=True)
            steps = trainer.get_steps()

            l, saved = trainer.forward(model, rdata, rlabel, steps)
            grad_info = trainer.backward(model, rdata, rlabel, steps, saved)
            trainer.accumulate_grad([grad_info])

            with torch.no_grad():
                for p_idx, p in enumerate(trainer.params):
                    pdata = p.data
                    N = p.numel()
                    for flat_i in np.random.choice(N, min(N, max_num_per_param), replace=False):
                        i = []
                        for s in reversed(p.size()):
                            i.insert(0, flat_i % s)
                            flat_i //= s
                        i = tuple(i)
                        ag = p.grad[i].item()
                        orig = pdata[i].item()
                        pdata[i] -= eps
                        steps = trainer.get_steps()
                        lm, _ = trainer.forward(model, rdata, rlabel, steps)
                        pdata[i] += eps * 2
                        steps = trainer.get_steps()
                        lp, _ = trainer.forward(model, rdata, rlabel, steps)
                        ng = (lp - lm).item() / (2 * eps)
                        pdata[i] = orig
                        rel_err = abs(ag - ng) / (atol + rtol * abs(ng))
                        info_msg = "testing param {} with shape [{}] at ({}):\trel_err={:.4f}\t" \
                                   "analytical={:+.6f}\tnumerical={:+.6f}".format(
                                       p_idx, format_intlist(p.size()),
                                       format_intlist(i), rel_err, ag, ng)
                        if unittest_verbosity() > 0:
                            print(info_msg)
                        self.assertTrue(rel_err <= 1, "gradcheck failed when " + info_msg)

    @suppress_wranings
    def test_backward(self):
        for ds, arch in (('MNIST', 'LeNet'), ('Cifar10', 'AlexCifarNet')):
            args = dict(
                dataset=ds, arch=arch,
                distill_steps=4, distill_epochs=2, distill_lr=0.02
            )
            with self.subTest(**args):
                if unittest_verbosity() > 0:
                    print("\nRunning subtest: \n{}".format(pprint.pformat(args)))
                state = options.get_dummy_state(**args)
                self._test_backward(self, state)


if __name__ == "__main__":
    unittest.main()
