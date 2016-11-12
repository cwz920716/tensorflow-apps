import numpy as np

class Symbol(object):
    def __init__(self, sym):
        self.sym = sym

    def codegen(self):
        return self.sym

class Op(object):
    def __init__(self, op):
        self.op = op
        self.args = []

class TFOperation(Op):
    def __init__(self, op, args, name=None, is_arith_overload=False, device=None):
        self.op = op
        self.args = args
        self.name = name
        self.is_arith_overload = is_arith_overload
        self.device = device

    def codegen(self):
        if self.is_arith_overload:
            return self.arith_codegen()
        return self.default_codegen()

    def arith_codegen(self):
        code = ''
        if self.name != None:
            code = code + self.name.codegen() + ' = '
        return code + self.args[0].codegen() + ' ' + self.op + ' ' + self.args[1].codegen()

    def default_codegen(self):
        code = ''
        if self.name != None:
            code = code + self.name.codegen() + ' = '
        code = code + self.op.codegen() + '('
        if len(self.args) > 0:
            code = code + self.args[0].codegen()
            args_ = self.args[1:]
            for arg in args_:
                code = code + ', ' + arg.codegen()
        code = code + ')'
        return code 

class TFObject(Op):
    def __init__(self, name, constructor='tf.Session'):
        self.name = name
        self.constructor = 'tf.Session'

    def codegen(self):
        code = self.name.codegen() + ' = ' + self.constructor + '()'
        return code 

class StringLiteral(object):
    def __init__(self, literal):
        self.literal = literal

    def codegen(self):
        return '\'' + self.literal + '\''

class ListLiteral(object):
    def __init__(self, list_literal):
        self.list_literal = list_literal

    def codegen(self):
        code = '['
        if len(self.list_literal) > 0:
            code = code + self.list_literal[0].codegen()
            lists = self.list_literal[1:]
            for i in lists:
                code = code + ', ' + i.codegen()
        code = code + ']'
        return code

class PyBinaryOp(Op):
    def __init__(self, op, arg1, arg2):
        self.op = op
        self.args = [arg1, arg2]

class PyCall(Op):
    def __init__(self, fun, args):
        self.fun = fun
        self.args = args

    def codegen(self):
        code = self.fun.codegen() + '('
        if len(self.args) > 0:
            code = code + self.args[0].codegen()
            args_ = self.args[1:]
            for arg in args_:
                code = code + ', ' + arg.codegen()
        code = code + ')'
        return code 

class PyAssign(Op):
    def __init__(self, lvalue, rvalue):
        self.op = '='
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.args = [lvalue, rvalue]

    def codegen(self):
        return self.lvalue.codegen() + ' = ' + self.rvalue.codegen()

class PyGetMember(Op):
    def __init__(self, obj, member):
        self.op = '.'
        self.obj = obj
        self.member = member
        self.args = [obj, member]

    def codegen(self):
        return self.obj.codegen() + '.' + self.member.codegen()

class PyGetItem(Op):
    def __init__(self, array, index):
        self.op = '[]'
        self.array = array
        self.index = index
        self.args = [array, index]

    def codegen(self):
        return self.array.codegen() + '[' + self.index.codegen() + ']'

def codegen_bb(bb):
    code = '#generate from basic block\n'
    for i in bb:
        code = code + i.codegen() + '\n'
    return code

def model_build():
    n_input = Symbol('n_input')
    n_hidden = Symbol('n_hidden')
    transfer_function = Symbol('transfer_function')
    optimizer = Symbol('optimizer')
    self_ = Symbol('self')
    bb = []
    inst = PyAssign(PyGetMember(self_, n_input), n_input)
    bb.append(inst)
    inst = PyAssign(PyGetMember(self_, n_hidden), n_hidden)
    bb.append(inst)
    transfer = Symbol('transfer')
    inst = PyAssign(PyGetMember(self_, transfer), transfer_function)
    bb.append(inst)
    weights = Symbol('weights')
    _initialize_weights = Symbol('_initialize_weights')
    inst = PyAssign(PyGetMember(self_, weights), PyCall(PyGetMember(self_, _initialize_weights), []))
    bb.append(inst)

    # model
    x = Symbol('x')
    tf_float32 = Symbol('tf.float32')
    placeholder = Symbol('tf.placeholder')
    None_ = Symbol('None')
    inst = TFOperation(placeholder, [tf_float32, ListLiteral([None_, PyGetMember(self_, n_input)])], name=PyGetMember(self_, x))
    bb.append(inst)
    hidden = Symbol('hidden')
    tf_add = Symbol('tf.add')
    tf_matmul = Symbol('tf.matmul')
    inst = TFOperation(PyGetMember(self_, transfer), [
                       TFOperation(tf_add, [
                                   TFOperation(tf_matmul, [
                                               PyGetMember(self_, x), 
                                               PyGetItem(PyGetMember(self_, weights), StringLiteral('w1'))
                                               ]), 
                                   PyGetItem(PyGetMember(self_, weights), StringLiteral('b1'))
                                   ])
                       ], name=PyGetMember(self_, hidden))
    bb.append(inst)
    reconstruction = Symbol('reconstruction')
    inst = TFOperation(tf_add, [
                       TFOperation(tf_matmul, [
                                   PyGetMember(self_, hidden),
                                   PyGetItem(PyGetMember(self_, weights), StringLiteral('w2'))
                                   ]),
                       PyGetItem(PyGetMember(self_, weights), StringLiteral('b2'))
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

    # test
    code = codegen_bb(bb)
    print code

model_build()
