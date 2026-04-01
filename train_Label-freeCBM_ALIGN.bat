set deleteConceptActivations=0	
set dataset=cub
set clip_name=ALIGN
set backbone=resnet18_cub
set n_iters_interpretability_cutoff=1
set interpretability_cutoff=0.15
set lambda=0.0002
set conceptFile=random_words_200
set extraSettings=--batch_size 124 --n_iters 5000 --print --clip_cutoff 0.1 --interpretability_cutoff %interpretability_cutoff% --lam %lambda% --n_iters_interpretability_cutoff %n_iters_interpretability_cutoff%

IF %deleteConceptActivations%==1 (
	del saved_activations\%dataset%_%conceptFile%_%clip_name%.pt
)
call conda activate BSThesisNewtorch
set PYTHONUTF8=1
python train_cbm.py --dataset %dataset% --backbone %backbone% --clip_name %clip_name% --feature_layer features.final_pool --concept_set data/concept_sets/%dataset%_%conceptFile%.txt %extraSettings%
pause