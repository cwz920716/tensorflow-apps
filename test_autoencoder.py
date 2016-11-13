import sys
from codegen import *
from devices import *

def model_build():
    n_input = Symbol('n_input')
    n_hidden = Symbol('n_hidden')
    transfer_function = Symbol('transfer_function')
    optimizer = Symbol('optimizer')
    self_ = Symbol('self')
    bb = BasicBlock()
    inst = PyAssign(PyGetMember(self_, n_input), n_input)
    bb.append(inst)
    inst = PyAssign(PyGetMember(self_, n_hidden), n_hidden)
    bb.append(inst)
    transfer = Symbol('transfer')
    inst = PyAssign(PyGetMember(self_, transfer), transfer_function)
    bb.append(inst)

    # variables
    weights = Symbol('weights')
    all_weights = Symbol('all_weights')
    _initialize_weights = Symbol('_initialize_weights')
    dict_ = Symbol('dict')
    inst = PyAssign(all_weights, PyCall(dict_, []))
    bb.append(inst)
    
    # variables
    tf_variable = Symbol('tf.Variable')
    w1 = StringLiteral('w1')
    b1 = StringLiteral('b1')
    w2 = StringLiteral('w2')
    b2 = StringLiteral('b2')
    autoencoder_Utils_xavier_init = Symbol('autoencoder.Utils.xavier_init')
    inst = TFOperation(tf_variable, [
                       TFOperation(autoencoder_Utils_xavier_init, [
                                   PyGetMember(self_, n_input),
                                   PyGetMember(self_, n_hidden)
                                   ])
                       ], name=PyGetItem(all_weights, w1))
    bb.append(inst)
    tf_zeros = Symbol('tf.zeros')
    tf_float32 = Symbol('tf.float32')
    dtype = Symbol('dtype')
    inst = TFOperation(tf_variable, [
                       TFOperation(tf_zeros, [
                                   ListLiteral([PyGetMember(self_, n_hidden)]),
                                   PyKeywordArg(dtype, tf_float32)
                                   ])
                       ], name=PyGetItem(all_weights, b1))
    bb.append(inst)
    inst = TFOperation(tf_variable, [
                       TFOperation(tf_zeros, [
                                   ListLiteral([PyGetMember(self_, n_hidden), PyGetMember(self_, n_input)]),
                                   PyKeywordArg(dtype, tf_float32)
                                   ])
                       ], name=PyGetItem(all_weights, w2))
    bb.append(inst)
    inst = TFOperation(tf_variable, [
                       TFOperation(tf_zeros, [
                                   ListLiteral([PyGetMember(self_, n_input)]),
                                   PyKeywordArg(dtype, tf_float32)
                                   ])
                       ], name=PyGetItem(all_weights, b2))
    bb.append(inst)
    inst = PyAssign(PyGetMember(self_, weights), all_weights)
    bb.append(inst)

    # model
    x = Symbol('x')
    tf_placeholder = Symbol('tf.placeholder')
    None_ = Symbol('None')
    inst = TFOperation(tf_placeholder, [tf_float32, ListLiteral([None_, PyGetMember(self_, n_input)])], name=PyGetMember(self_, x))
    bb.append(inst)
    hidden = Symbol('hidden')
    tf_add = Symbol('tf.add')
    tf_matmul = Symbol('tf.matmul')
    inst = TFOperation(PyGetMember(self_, transfer), [
                       TFOperation(tf_add, [
                                   TFOperation(tf_matmul, [
                                               PyGetMember(self_, x), 
                                               PyGetItem(PyGetMember(self_, weights), w1)
                                               ]), 
                                   PyGetItem(PyGetMember(self_, weights), b1)
                                   ])
                       ], name=PyGetMember(self_, hidden))
    bb.append(inst)
    reconstruction = Symbol('reconstruction')
    inst = TFOperation(tf_add, [
                       TFOperation(tf_matmul, [
                                   PyGetMember(self_, hidden),
                                   PyGetItem(PyGetMember(self_, weights), w2)
                                   ]),
                       PyGetItem(PyGetMember(self_, weights), b2)
                       ], name=PyGetMember(self_, reconstruction))
    bb.append(inst)

    # cost
    cost = Symbol('cost')
    tf_reduce_sum = Symbol('tf.reduce_sum')
    tf_pow = Symbol('tf.pow')
    tf_sub = Symbol('tf.sub')
    const2_0 = Symbol('2.0')
    const0_5 = Symbol('0.5')
    inst = TFOperation('*', [ 
                       const0_5,
                       TFOperation(tf_reduce_sum, [
                                   TFOperation(tf_pow, [
                                               TFOperation(tf_sub, [
                                                           PyGetMember(self_, reconstruction),
                                                           PyGetMember(self_, x)
                                                           ]),
                                               const2_0
                                               ])
                                   ])
                       ], name=PyGetMember(self_, cost), is_arith_overload=True)
    bb.append(inst)
    minimize = Symbol('minimize')
    inst = TFOperation(PyGetMember(optimizer, minimize), [
                       PyGetMember(self_, cost)
                       ], name=PyGetMember(self_, optimizer))
    bb.append(inst)

    # init
    init = Symbol('init')
    tf_initialize_all_variables = Symbol('tf.initialize_all_variables')
    inst = TFOperation(tf_initialize_all_variables, [], name=init)
    bb.append(inst)
    sess = Symbol('sess')
    inst = TFObject(PyGetMember(self_, sess))
    bb.append(inst)
    run = Symbol('run')
    inst = PyCall(PyGetMember(PyGetMember(self_, sess), run), [init]) 
    bb.append(inst)

    return bb

print '# Usage: test_*.py [dev_type_string] [dev_id_string]'
dev_string='/gpu:0'
argc = len(sys.argv)
if argc > 2:
    dev_string = '/' + sys.argv[1] + ':' + sys.argv[2]
    print '# dev_string=', dev_string

bb = model_build()
dev = Device(dev_string)
op_placer = SingleDevicePlacer(dev)
bb = op_placer.scanAndAssign(bb)
print bb.codegen()
