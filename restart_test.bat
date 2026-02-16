set deleteConceptActivations=0	
set dataset=cub
set model=ViT-B16
set backbone=resnet18_cub
set n_iters_interpretability_cutoff=1
set interpretability_cutoff=0.45
set conceptFile=filtered
set extraSettings=--n_iters 5000 --clip_cutoff 0.26 --interpretability_cutoff %interpretability_cutoff% --lam 0.0006 --n_iters_interpretability_cutoff %n_iters_interpretability_cutoff%

IF %deleteConceptActivations%==1 (
	del saved_activations\%dataset%_%conceptFile%_%model%.pt
)
call conda activate BSThesisNewtorch
set PYTHONUTF8=1
python train_cbm.py --dataset %dataset% --backbone %backbone% --feature_layer features.final_pool --concept_set data/concept_sets/%dataset%_%conceptFile%.txt %extraSettings%
pause