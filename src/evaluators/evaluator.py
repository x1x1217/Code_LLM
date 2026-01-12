'''evaluate the generated code'''

import os
import tempfile, subprocess

class KodCodeEvaluator:
    
    def __init__(self, timeout: int=5):
        self.timeout = timeout
        
    def evaluate(self, solution_code: str, test_code: str) -> bool:
        
        with tempfile.TemporaryDirectory() as tmpdir:
            solution_path = os.path.join(tmpdir, 'solution.py')
            test_path =  os.path.join(tmpdir, 'test_solution.py')
            
            with open(solution_path, 'w', encoding='utf-8') as f:
                f.write(solution_code)
            
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            try:
                result = subprocess.run(
                    ["pytest", "test_solution.py", "-q", "--disable-warnings", "--maxfail=1"],
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout
                )
                # print("STDOUT:", result.stdout)
                # print("STDERR:", result.stderr)
                
                return result.returncode == 0
            
            except subprocess.TimeoutExpired:
                return False
            except Exception as e:
                return False
