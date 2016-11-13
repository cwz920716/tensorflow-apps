import re

class MacroPass(object):
    def visit(self, text):
        pass

class IncludePass(MacroPass):
    def __init__(self):
        self.doc_string = 'Include Pass: handling #include(...) as if C/C++ INCLUDE macros.'
        return

    def visit(self, text):
        cmd = text
        start_of_cmd = len(cmd) - len(cmd.lstrip())
        padding_ws = ''
        if start_of_cmd > 0:
            padding_ws = cmd[0:start_of_cmd]
            cmd = cmd[start_of_cmd:]
        cmd = cmd.rstrip()
        if len(cmd) <= 10 or (not (cmd[0:9] == '#include(' and cmd[-1] == ')')):
            return text
        path = cmd[9:-1] + '.py'
        ret_text = ''
        with open(path) as f:
            for line in f:
                ret_text = ret_text + padding_ws + line
        return ret_text


def preprocess(fname):
    code = ''
    ip = IncludePass()
    with open(fname) as f:
        for line in f:
            code = code + ip.visit(line)
    return code

import sys
print preprocess(sys.argv[1] + '.pym')
