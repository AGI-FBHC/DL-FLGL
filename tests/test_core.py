import unittest
import numpy as np
import torch
from dataclasses import replace
from sklearn.preprocessing import MinMaxScaler
from dl_flgl import Config, DLFLGL
from dl_flgl.clustering import impurity, rule_geometry
from dl_flgl.consequent import correlation_psd, update_relation, soft_targets, smooth_loss, smooth_gradient, optimize
from dl_flgl.data import demo_data
from dl_flgl.metrics import evaluate

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_impurity_hand_calculation(self):
        u = np.array([[1., 0.], [1., 0.], [0., 1.]])
        y = np.array([[1., 0.], [1., 1.], [0., 1.]])
        self.assertAlmostEqual(impurity(u, y), (2/3)*(1-(2/3)**2-(1/3)**2), places=10)

    def test_gaussian_width(self):
        x = np.array([[0.], [2.]])
        u = np.array([[.8], [.2]])
        v, delta2 = rule_geometry(x, u, 2., .1)
        center = (.2**2 * 2) / (.8**2 + .2**2)
        variance = (.8**2 * center**2 + .2**2 * (2-center)**2) / (.8**2+.2**2)
        np.testing.assert_allclose(v, [[center]])
        np.testing.assert_allclose(delta2, [[.01 * variance]])

    def test_relation_orientation_and_clipping(self):
        z = torch.tensor([[2., .2], [-1., .8]], dtype=torch.float64)
        y = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
        s = update_relation(z, y)
        expected = torch.tensor([[1/1.2, 0.], [.2/1.2, 1.]], dtype=torch.float64)
        torch.testing.assert_close(s, expected)
        torch.testing.assert_close(soft_targets(y, s), expected.T)
        self.assertTrue(torch.all(soft_targets(torch.ones_like(y), s) <= 1))
        self.assertTrue(torch.equal(update_relation(z, torch.zeros_like(y)), torch.zeros_like(s)))

    def test_psd_with_constant_labels(self):
        y = torch.tensor([[1.,0.,1.],[0.,1.,1.],[1.,0.,1.]], dtype=torch.float64)
        r = correlation_psd(y)
        torch.testing.assert_close(r, r.T)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(r).min()), -1e-10)
        self.assertTrue(torch.isfinite(r).all())

    def test_correlation_matches_local_reference(self):
        import warnings
        for y in (torch.tensor([[1.,0.,1.],[0.,1.,1.],[1.,0.,1.]], dtype=torch.float64),
                  torch.tensor([[1.,0.,1.]], dtype=torch.float64),
                  torch.tensor([[1.],[0.],[1.]], dtype=torch.float64)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                corr = torch.nan_to_num(torch.corrcoef(y.T).reshape(y.shape[1], y.shape[1]), nan=0., posinf=0., neginf=0.)
                raw = 1 - corr
                raw = (raw + raw.T) / 2
                values, vectors = torch.linalg.eigh(raw)
                expected = (vectors * values.clamp_min(0)) @ vectors.T
                torch.testing.assert_close(correlation_psd(y), expected)

    def test_relation_uses_local_epsilon(self):
        z = torch.tensor([[1e-10, 2e-10]], dtype=torch.float64)
        y = torch.tensor([[1., 0.]], dtype=torch.float64)
        a = z.clamp(0, 1).T @ y
        torch.testing.assert_close(update_relation(z, y), a / (a.sum(0, keepdim=True) + 1e-8))

    def test_gradient_against_autograd(self):
        generator = torch.Generator().manual_seed(17)
        phi = torch.randn(7, 5, generator=generator, dtype=torch.float64)
        p = torch.randn(5, 3, generator=generator, dtype=torch.float64, requires_grad=True)
        y = torch.randint(0, 2, (7,3), generator=generator).double()
        target = torch.rand(7,3, generator=generator, dtype=torch.float64)
        r = correlation_psd(y)
        for alpha in (0., .37, 1.):
            c = Config(alpha=alpha, beta1=.7, beta2=.3)
            automatic, = torch.autograd.grad(smooth_loss(p, phi, y, target, r, c), p)
            torch.testing.assert_close(smooth_gradient(p, phi, y, target, r, c), automatic)

    def test_fista_against_closed_form(self):
        phi = torch.eye(3, dtype=torch.float64)
        y = torch.tensor([[1.,0.],[0.,1.],[1.,1.]], dtype=torch.float64)
        target = .4 * y
        c = Config(alpha=.4, beta1=0., beta2=.2, beta3=.1, maxIter=2000, minimumLossMargin=1e-9)
        actual, history = optimize(phi, y, target, torch.zeros(2,2,dtype=torch.float64), c, torch.zeros_like(y))
        b = c.alpha*y + (1-c.alpha)*target
        expected = (b-c.beta3).clamp_min(0)/(1+2*c.beta2)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
        self.assertTrue(history['converged'])

    def test_end_to_end_reproducibility(self):
        x,y = demo_data(11)
        scaler = MinMaxScaler().fit(x[:70])
        c = Config(B=4, B_prime=2, C=2, num_epochs=2, maxIter=50)
        a = DLFLGL(c).fit(scaler.transform(x[:70]), y[:70])
        b = DLFLGL(c).fit(scaler.transform(x[:70]), y[:70])
        np.testing.assert_allclose(a.decision_function(scaler.transform(x[70:])), b.decision_function(scaler.transform(x[70:])))
        strengths = a.antecedent_.firing_strength(scaler.transform(x[70:]))
        np.testing.assert_allclose(strengths.sum(axis=1), 1)
        mu = a.predict_membership(scaler.transform(x[70:]))
        self.assertTrue(((mu >= 0) & (mu <= 1)).all())
        self.assertEqual(a.P_.shape, (4*9,4))

    def test_metrics_known_case(self):
        result = evaluate(np.array([[1,0,1]]), np.array([[.9,.1,.8]]))
        self.assertEqual(result, {'AP':1., 'HL':0., 'OE':0., 'RL':0., 'CV':1/3})
        self.assertIsNone(evaluate(np.zeros((1,2)), np.zeros((1,2)))['AP'])

    def test_invalid_config(self):
        for kwargs in ({'B_prime':10}, {'m':1}, {'beta1':-1}, {'maxIter':0}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

if __name__ == '__main__':
    unittest.main()
