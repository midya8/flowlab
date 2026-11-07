import json
import os
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def test_train_score_sample(self):
        with tempfile.TemporaryDirectory() as d:
            data,model=os.path.join(d,'points.json'),os.path.join(d,'model.json')
            with open(data,'w') as f:json.dump([[0.,0.],[1.,1.]],f)
            def run(*args):return subprocess.run([sys.executable,'-m','flowlab',*args],capture_output=True,text=True)
            p=run('train',data,model,'--updates','2');self.assertEqual(p.returncode,0,p.stderr)
            p=run('score',model,data);self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(len(json.loads(p.stdout)['log_density']),2)
            p=run('sample',model,'--count','4');self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(len(json.loads(p.stdout)['points']),4)

    def test_invalid_model_clean_error(self):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            json.dump({},f);f.flush()
            p=subprocess.run([sys.executable,'-m','flowlab','sample',f.name],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)

    def test_gradient_outputs_and_row_limit(self):
        from flowlab.core import Flow,artifact
        with tempfile.TemporaryDirectory() as d:
            model,data=os.path.join(d,'model.json'),os.path.join(d,'points.json')
            with open(model,'w') as f:json.dump(artifact(Flow()),f)
            for rows,code in [([[.2,.4]],0),([[0.,0.]]*256,0),([[0.,0.]]*257,2),([],2)]:
                with open(data,'w') as f:json.dump(rows,f)
                p=subprocess.run([sys.executable,'-m','flowlab','gradient',model,data],capture_output=True,text=True)
                self.assertEqual(p.returncode,code,p.stderr)
                if code==0:
                    result=json.loads(p.stdout)['rows'];self.assertEqual(len(result),len(rows))
                    self.assertEqual(result[0],Flow().input_score(rows[0]))
                else:self.assertNotIn('Traceback',p.stderr)
