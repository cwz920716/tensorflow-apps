import numpy as np
from codegen import *

class Device(object):
    def __init__(self, dev_id):
        # dev_id *must* be a string like '/cpu:0', '/gpu:0', '/gpu:1', etc.
        self.dev_id = str(dev_id)

    def __repr__(self):
        return self.dev_id

class DeviceBlock(object):
    def __init__(self, device):
        self.device = device
        self._bb = []

    def append(self, inst):
        self._bb.append(inst)

    def __len__(self):
        return len(self._bb)

    def __getitem__(self, position):
        return self._bb[position]

    def codegen(self):
        dev_string = StringLiteral(str(self.device))
        tf_device = Symbol('tf.device')
        w = PyWith(PyCall(tf_device, [dev_string]), self._bb)
        return w.codegen()

class AbstractOpPlacer(object):
    def scanAndAssign(self, bb):
        pass

class SingleDevicePlacer(AbstractOpPlacer):
    def __init__(self, device):
        self.device = device

    def scanAndAssign(self, bb):
        # a linear scan device placement algorithm
        out_bb = BasicBlock()
        tf_buffer = None
        for inst in bb:
            if isinstance(inst, TFOperation):
                if tf_buffer is None:
                    tf_buffer = DeviceBlock(self.device)
                tf_buffer.append(inst)
            else:
                if not (tf_buffer is None):
                    out_bb.append(tf_buffer)
                    tf_buffer = None
                out_bb.append(inst)
        if not (tf_buffer is None):
            out_bb.append(inst)
        return out_bb
