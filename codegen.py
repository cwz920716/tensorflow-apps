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
    def __init__(self, name, args, constructor=Symbol('tf.Session')):
        self.name = name
        self.args = args
        self.constructor = constructor

    def codegen(self):
        code = self.name.codegen() + ' = ' + self.constructor.codegen() + '('
        if len(self.args) > 0:
            code = code + self.args[0].codegen()
            args_ = self.args[1:]
            for arg in args_:
                code = code + ', ' + arg.codegen()
        code = code + ')'
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

class PyKeywordArg(Op):
    def __init__(self, lvalue, rvalue):
        self.op = '='
        # lvalue *must* be a symbol
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.args = [lvalue, rvalue]

    def codegen(self):
        return self.lvalue.codegen() + '=' + self.rvalue.codegen()

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

class PyWith(Op):
    def __init__(self, with_expr, block, as_expr=None, indent='    '):
        self.with_expr = with_expr
        self.block = block
        self.as_expr = as_expr
        self.indent = indent

    def codegen(self):
        code = 'with ' + self.with_expr.codegen() + ':\n'
        for inst in self.block:
            code = code + self.indent + inst.codegen() + '\n'
        return code

class BasicBlock(object):
    def __init__(self, indent=''):
        self._bb = []
        self.indent = ''

    def append(self, inst):
        self._bb.append(inst)

    def __len__(self):
        return len(self._bb)

    def __getitem__(self, position):
        return self._bb[position]

    def codegen(self):
        code = '# generate from basic block\n'
        for i in self._bb:
            code = code + self.indent + i.codegen() + '\n'
        return code
