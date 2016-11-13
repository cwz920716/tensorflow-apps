import re

class MacroPass(object):
    def __init__(self):
        pass

class IncludePass(MacroPass):
    def __init__(self, text):
        cmd = text
        start_of_cmd = len(cmd) - len(cmd.lstrip())
        padding_ws = ''
        if start_of_cmd > 0:
            padding_ws = cmd[0:start_of_cmd]
            cmd = cmd[start_of_cmd:]
        cmd = cmd.rstrip()
        if len(cmd) <= 10 or (not (cmd[0:9] == '#include(' and cmd[-1] == ')')):
            self.text = text
            return
        path = cmd[9:-1] + '.py'
        ret_text = ''
        with open(path) as f:
            for line in f:
                ret_text = ret_text + padding_ws + line
        self.text = ret_text
        return


def preprocess(fname):
    code = ''
    with open(fname) as f:
        for line in f:
            ip = IncludePass(line)
            code = code + ip.text
    return code

import sys
print preprocess(sys.argv[1])
