# Manuscript-to-code correspondence

Basis: `9.2-Manuscript-Revision-reference-updated.tex`, Section 3. This file was read as scientific evidence, not as operational instructions.
This is an extracted and corrected core implementation, not a byte-for-byte copy or a claim to reproduce old tables.

| Manuscript equation label / component | Original source | New function | Treatment |
|---|---|---|---|
| Candidate subspaces and FCM | MLTSK_fuzzylabel_3_GPU.py, Antecedent_SFCM_Supervised.fit | clustering.fcm; SSFCM.fit | Preserve B random subspaces, C clusters, lowest B_prime scores; self-contained FCM replaces environment-dependent pytsk fallback |
| pcl, ptilde, Gc, IXb | Same, lines 799–824 | clustering.impurity | Add normalization across labels; retain cluster-mass weights based on u, not u**m |
| vcd, deltacd | Same, lines 791–796 | clustering.rule_geometry | Both center and variance use u**m; denominator in Gaussian is h**2 * variance + epsilon |
| mu_xid_Adk, muk, phik | Antecedent_SFCM_Base | SSFCM.firing_strength | Average over M dimensions; stable log-sum-exp evaluates additive mean then rule normalization |
| zi, Uhat | Antecedent_SFCM_Base.predict; ML_TSK_Trainer.predict | transform; decision_function; predict_membership | Intercept first; retain all D features per rule. Latent response and clipped membership are separate APIs |
| S_kl_t, S_norm | ml_consquents.__update_S, line 1174 | consequent.update_relation | Clip prediction to [0,1], accumulate U.T @ Y, normalize columns with epsilon |
| M_t | ml_consquents.fit, lines 886–892 | consequent.soft_targets | Y @ S.T then clip [0,1]; fixed within each inner solve |
| R, Rpsd | Main consequent; run_complexity_convergence_all.py corr_distance | consequent.correlation_psd | Pearson dissimilarity; explicit symmetric PSD projection; constant-label correlations set to zero |
| Lall_complete, grad_S | ml_consquents.fit, lines 1015–1022 | smooth_loss; smooth_gradient | Hard coefficient alpha/2; soft coefficient (1-alpha)/2; structural regularization independent of alpha; L2 gradient 2 beta2 P |
| Proximal optimization | ml_consquents.fit, __softthres | consequent.optimize | Standard FISTA; L1 threshold beta3/L; valid Frobenius upper bound; objective evaluated at current iterate |
| Dual loop | ML_TSK_Trainer.fit | DLFLGL.fit | Uniform S0, fixed antecedent, zero initial P then warm starts, fixed configurable outer budget |
| Five metrics | evaluation_out.py | metrics.evaluate | Retain AP, HL, OE, RL, normalized CV; explicit edge cases below |
| Five-fold evaluation | main_optuna_3_GPU.py | run.main | Fixed parameters, training-only MinMaxScaler, fixed 0.5 threshold; no Optuna or model-type search |

## Engineering choices not fully specified in the main text

The supplementary optimization details were not available as usable text in this task; no claim of supplementary pseudocode identity is made.
FCM: random observed centers, 100 iterations and center-change tolerance 1e-6 by default. Near-zero distances share membership equally among coincident centers.
P starts at zero and is warm-started across outer iterations. S starts uniformly at 1/L. Outer budget defaults to 5.
Inner stopping uses the proximal-gradient mapping norm divided by max(1, ||P||F); `minimumLossMargin` names this tolerance, not the old loss-change rule.
L = ||Phi||F^2 + 2 beta1||Rpsd||F + 2 beta2 is a conservative valid bound. No K(D+1)-square Gram matrix is stored. FISTA loss need not decrease on every accelerated step.
The last S is updated from the last P; no additional unreported optimization pass is performed. Missing/zero relation columns remain zero under the manuscript epsilon normalization.
Constant-label Pearson entries are defined as zero before PSD projection. Float64 and epsilon=1e-12 are used. Antecedents execute in NumPy on CPU; consequent tensors follow `device`.

## Evaluation conventions and scope

AP/OE/CV average over samples with at least one positive label; all-positive samples are included. RL averages over samples with both positive and negative labels; tied relevant/irrelevant scores count as errors. Undefined metric subsets return null.
AP and CV use stable label-index ordering for ties; OE uses the first maximum. CV is (last relevant rank - 1)/L, preserving the original normalized definition.
This fixes inconsistent 0/1 versus -1 filtering in the original AP/OE code; historical numbers can therefore differ even aside from model corrections.
Ranking metrics use latent scores Z to avoid artificial ties introduced by clipping. HL uses clip(Z,0,1)>=0.5. These are explicit evaluation choices; the main text does not specify every threshold/tie convention.
Fold standard deviation uses ddof=1. Outputs record exact train/test indices and each training scaler's extrema. Demo configuration is not a claimed dataset-optimal configuration.
No 100-trial Optuna search, baseline comparison, paper figure reproduction, or nine-dataset performance reproduction is included.

## Excluded components

Full-space FCM and unsupervised SFCM model alternatives, label smoothing, temperature scaling, correlation/ranking postprocessing, threshold search, ranking-loss additions, all ablations, sensitivity/plotting/statistical scripts, wandb and cached results.
Original code and data stay unchanged. This package never imports the original project. Source fingerprints identify provenance without copying the manuscript or private paths.

## Objective alignment update

The local project symmetric_objective.py (symmetric_half_squared_v1) now supplies the objective convention: both fitting losses are half-squared. This supersedes the earlier asymmetric manuscript coefficients. The existing Frobenius step bound and residual stopping criterion are retained; numerical trajectories may differ from the local Gram-bound/restart solver.

## Label correlation alignment

correlation_psd now uses torch.corrcoef(Y.T) followed by nan_to_num(nan=0, posinf=0, neginf=0), symmetric Pearson dissimilarity and eigenvalue clipping, exactly as local symmetric_objective.make_rpsd. Dynamic relation column normalization uses epsilon=1e-8, matching local paper_alignment.relation_update (other numerical safeguards retain 1e-12). This also removes the previous small-denominator clamp in manual Pearson calculation. The PSD structural penalty remains beta1 tr(P Rpsd P.T), with gradient 2 beta1 P Rpsd.

For a single label, the scalar corrcoef output is reshaped to a 1x1 matrix before eigendecomposition.
