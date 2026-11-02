"""Run: py -3.13 test_notebook.py. Executes definition cells only, never training."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PATH = Path(__file__).with_name('signature_forgery_detection.ipynb')
NS = {'__name__': 'notebook_definitions'}
if PATH.exists():
    for cell in json.loads(PATH.read_text(encoding='utf-8'))['cells']:
        if 'definitions' in cell.get('metadata', {}).get('tags', []):
            exec(''.join(cell['source']), NS)

class NotebookChecks(unittest.TestCase):
    def test_metric_orientation(self):
        self.assertIn('evaluate', NS, 'Notebook evaluation definitions must exist')
        r = NS['evaluate']([0,0,1,1], [.1,.8,.6,.9], .5)
        self.assertAlmostEqual(r['accuracy'], .75)
        self.assertAlmostEqual(r['FAR'], .5)
        self.assertAlmostEqual(r['FRR'], 0)
        self.assertAlmostEqual(r['forged_recall'], .5)
        self.assertAlmostEqual(r['HTER'], .25)
    def test_threshold_ties_and_endpoints(self):
        self.assertIn('select_threshold', NS)
        y = np.array([0,0,1,1])
        t = NS['select_threshold'](y, np.array([.1,.2,.8,.9]))
        self.assertEqual(NS['evaluate'](y,[.1,.2,.8,.9],t)['accuracy'],1)
        self.assertTrue(np.isfinite(NS['select_threshold'](y,np.ones(4))))
        self.assertEqual(NS['evaluate'](y,np.ones(4),2)['FRR'],1)
        self.assertEqual(NS['evaluate'](y,np.ones(4),0)['FAR'],1)
        with self.assertRaises(ValueError): NS['select_threshold']([1,1],[0,1])
    def test_enrollment_query_separation(self):
        self.assertIn('enrollment_protocol', NS)
        rows=[dict(path=f'{w}-{label}-{i}',writer=str(w),label=label,sha256=f'{w}-{label}-{i}')
              for w in range(20) for label in [0,1] for i in range(8)]
        a=NS['enrollment_protocol'](pd.DataFrame(rows), seed=0,n_refs=3)
        b=NS['enrollment_protocol'](pd.DataFrame(rows), seed=0,n_refs=3)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue((a.groupby('writer')['split'].nunique()==1).all())
        self.assertEqual(len(a[a.role=='enrollment']),60)
        self.assertFalse(set(a[a.role=='enrollment'].path)&set(a[a.role=='query'].path))
        rows[1]['sha256']=rows[0]['sha256']
        with self.assertRaises(ValueError): NS['enrollment_protocol'](pd.DataFrame(rows))
    def test_reference_aggregation(self):
        self.assertIn('aggregate_scores', NS)
        pairs=pd.DataFrame(dict(query=['b','a','b','a'],writer=['1']*4,label=[0,1,0,1]))
        out=NS['aggregate_scores'](pairs,[1,2,3,4])
        self.assertEqual(out.set_index('query').score.to_dict(),{'a':3.,'b':2.})
    def test_preprocessing_and_features(self):
        self.assertIn('preprocess',NS)
        with self.assertRaises(ValueError): NS['preprocess'](np.full((100,200),255,np.uint8))
        im=np.full((100,200),255,np.uint8); im[45:50,20:180]=0
        bw=NS['preprocess'](im)
        self.assertEqual(bw.shape,(64,128))
        self.assertTrue(np.isfinite(NS['geometric_features'](bw)).all())
    def test_writer_bootstrap_reproducible(self):
        self.assertIn('writer_bootstrap',NS)
        frame=pd.DataFrame(dict(writer=['a','a','b','b'],label=[0,1,0,1],score=[0,1,0,1]))
        a=NS['writer_bootstrap'](frame,.5,n_boot=20,seed=1)
        b=NS['writer_bootstrap'](frame,.5,n_boot=20,seed=1)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue((a['lower_95']==a['upper_95']).all())
    def test_cnn_symmetry_and_batches(self):
        self.assertIn('SiameseNet',NS)
        torch=NS['torch']; torch.manual_seed(0)
        model=NS['SiameseNet']().eval()
        x=torch.rand(5,1,64,128); pairs=np.array([[0,1],[1,0],[2,3],[3,4]])
        scores=NS['cnn_scores'](model,x,pairs,batch_size=2,device='cpu')
        self.assertEqual(scores.shape,(4,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores[0]),float(scores[1]),places=6)

if __name__=='__main__': unittest.main(verbosity=2)
