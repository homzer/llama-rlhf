import collections
import re
import traceback

ExecOutputs = collections.namedtuple("ExecOutputs", [
    'error_line_id', "error_type", "result"
])


class CodeExecutor:
    def __init__(self):
        self.globals = {
            '__builtins__': {
                'list': list,
                'dict': dict,
                'str': str,
                'set': set,
                'print': print,
                'int': int,
                'range': range,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
                'len': len,
                'abs': abs,
                'zip': zip,
                'float': float,
                'bool': bool,
                'sorted': sorted,
                'isinstance': isinstance,
                'reversed': reversed,
                'map': map,
                'enumerate': enumerate,
                'ord': ord,
                'filter': filter,
                'bin': bin,
                'chr': chr,
                'all': all,
                'any': any,
                'tuple': tuple,
                '__import__': __import__
            }}
        self.pattern = r'File "<string>", line (\d+)'  # for extracting error line id
        self.unsafe_modules = [
            "while",
            "sys",
            "thread",
            "threading",
            "multiprocessing",
            "multiprocess",
            'subprocess',
            'os',
            'socket',
            'builtins',
            'importlib',
            'marshal',
            'ctypes'
        ]

    def safety(self, code: str) -> ExecOutputs:
        """ examine the code to judge whether is safe to run or not """
        error_line_id = None
        error_type = None
        matches = re.search(rf'import\s+({"|".join(self.unsafe_modules)})[^\n\w]*\n+', code)
        if matches:
            # search for line id
            match = matches.group(1)
            error_type = f"<class 'NotSafeToRunError'> Importing '{match}' module is prohibited"
            for i, line in enumerate(code.split('\n')):
                if re.search(rf'import\s+{match}', line):
                    error_line_id = i
                    break
        if 'while' in code:  # TODO: infinite loop detecting.
            for i, line in enumerate(code.split('\n')):
                error_type = f"<class 'NotSafeToRunError'> Using the 'while' statement is prohibited"
                if re.search(r'\s*while\s+', line):
                    error_line_id = i
                    break
        return ExecOutputs(
            error_line_id=error_line_id,
            error_type=error_type,
            result={}
        )

    def forward(self, code: str) -> ExecOutputs:
        """
        Given a code string, execute it.
        Return `True` if it is executed successfully;
        Or return the error information otherwise.
        """
        trace_backing = None
        error_line = None
        error_type = None
        local_variables = {}
        safety = self.safety(code)
        if safety.error_type is not None:
            return safety
        else:
            try:
                exec(code, self.globals, local_variables)
            except Exception as e:
                error_type = f"{str(type(e))} {str(e)}"
                trace_backing = traceback.format_exc()

            # extract error line id
            if trace_backing is not None:
                match = re.search(self.pattern, trace_backing)
                error_line = int(match.group(1)) if match else None

        return ExecOutputs(
            error_line_id=error_line,
            error_type=error_type,
            result=local_variables
        )
