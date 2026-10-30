# DIAMOND Pixel Evaluation

Evaluation of predicted highway frames against observed frames.

## Language

**One-step pixel evaluation**:
Comparison of a predicted next frame with its observed target, using real preceding frames as context.
_Avoid_: Autoregressive rollout evaluation

**Unmasked pixel MAE**:
Mean absolute prediction error across all image pixels and color channels.
_Avoid_: TTC error, pixel MSE

**Masked pixel MAE**:
Absolute image prediction error weighted by a target-frame spatial mask and normalized by its weight. It is an evaluation metric, separate from the training objective.
_Avoid_: Masked training loss

**Five-fold cross-validation**:
Five independent training runs, each holding out a different fifth of the episodes for evaluation. As in KineMamba, the train and test splits are pooled before folding, so no episode stays outside CV.
_Avoid_: Five partitions evaluated using one fixed checkpoint
