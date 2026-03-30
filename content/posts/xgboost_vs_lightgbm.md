+++
title = 'What do we learn by making Xgboost behave like LightGBM?'
date = 2026-03-22T19:46:17+01:00
draft = true
+++

Recently, in an ML interview, I was tasked to develop model for a fraud detection case when customers are flagged between 0, 1 and unknown where these labels depict whether they have commit a fradulous transaction or not. 

In that interview with a famous nordic bank, one of the models I used was a gradient boosting model, and specifically the LightGBM implementation. 

During the interview, the interviewer asked a lot of questions about the hyper-parameters of the LightGBM and I froze because I realized I mixed up LightGBM and XGboost with one another. 

There are a lot of tutorials about the difference between the two on a theoretical level. One of the main differences is how they build the tree of course. Xgboost builds the tree with `depth-wise` grow policy versus LightGBM uses `leaf-wise` grow policy.

LightGBM's growth policy greedy and local. It aims to optimize: "Splitting on which single node leads to largest performance drop in objective?". On the other hand, XGboost works differently. Instead of chasing the best leaf to split on, it aims to expand the tree uniformly.

Even if this is the main difference between the two, their official packages have even further practical difference. 

In this gist, [https://gist.github.com/amir-rahnama/494ec346aca2eee6f9f17c6ec161eb99](Gradient Boosting Parameter Mappings), I mapped out different parameters used in different implementations of Gradient Boosting Trees. If you look at the file, you see that they do not support the same set of parameters. Something I should have known before the interview, and I didn't. 

But I wanted to test this mapping, or in other ways, test whether I can get them to behave the same. Both Xgboost and LightGBM have a set of similar parameters naturally. Let us try to set these parameters to a specific value first. Let us chose two trees with `n_estimators` to be a small number to be able to visualize the trees later on.

```python

SHARED = dict(
    n_estimators     = 2,
    learning_rate    = 0.3,
    reg_lambda       = 1.0,
    reg_alpha        = 0.0,
    subsample        = 1.0,
    colsample_bytree = 1.0,
    min_child_weight = 1,
)
```

Note that `reg_alpha` represents the L1 regularization and `reg_lambda` is the L2 regularization. Since trees do not necessarily have $|W|$ (weight vectors), these regularization parameters have a bit of different effect and interpretaiton. 


Now, in order to have almost equally trees, all you need to do is fix the `N_LEAVES` to 8 to make the tree with 2^3 leaves. Then, XGBoost  with grow_policy=`lossguide` and max_leaves=N_LEAVES and no `max_depth` (max_depth=0) will build the tree the same way as LightGBM with num_leaves=`N_LEAVES` and `max_depth=-1`. 


```python 
model_xgb = xgb.XGBClassifier(
    **SHARED,
    grow_policy  = 'lossguide',  # ← leaf-wise growth, same strategy as LightGBM
    max_leaves   = N_LEAVES,     # ← equivalent to LightGBM's num_leaves
    max_depth    = 0,            # 0 = no depth cap in lossguide mode
    random_state = SEED,
    verbosity    = 0,
)

model_lgb = lgb.LGBMClassifier(
    **SHARED,
    num_leaves        = N_LEAVES,  # primary complexity control
    max_depth         = -1,        # -1 = no depth cap
    min_child_samples = 1,
    subsample_freq    = 0,
    random_state      = SEED,
    verbose           = -1,
)
```

The only real two parameters that make Xgboost to behave like LightGBM are then essentially: `max_depth` and `grow_policy`. In the case of Xgboost, these two parameters have an interaction in place. 

In Xgboost, you can either select `grow_policy` to be `lossguide` or `depthwise`. In `loss_guide` the criterion with split is always driven by loss change. This makes the tree to grow on certain leaf nodes further away from the root. On the other hand, `depthwise` aims to split at nodes closer to the root, hence making the tree balanced. 

Let's see first how much these two trees will look the same on `make` moon dataset.

INSERT IMAGE tree_comparison_1

Are these changes that important to the decision boundary or accuracy. Xgboost's  test accuracy is around 91.00% while LightGBM test accuracy is 89.00%. 


# Complexity is kept equivalent:
#   XGBoost  max_depth=3   → fully symmetric tree has 2^3 = 8 leaves
#   LightGBM num_leaves=8  → same budget, but spent leaf-wise (greedily)
#
# LightGBM's max_depth=-1 (no cap) lets it go deeper on a single branch
# if that yields higher gain — the key structural difference vs depthwise.


In the visualizations below, we see three measures. First, the percentage of features used to split at each depth. The gain by splitting those features at each depth and lastly the balance ratio in each depth. As we can see, the majority of similarities between the two trees happen at the first two levels of depth. Then the trees behave differently. 

These internal variations however does not affect the decision boundary as you can see below.

part 2 

Okay so so far, we saw that using a combination of `N_LEAVES`, `max_depth` and `grow_policy`, Xgboost behaves somewhat like the LightGBM. 

What will we happen if we keep the rest of shared parameters similar, but 


```python
model_xgb = xgb.XGBClassifier(
    **SHARED,
    grow_policy  = 'depthwise',  # ← leaf-wise growth, same strategy as LightGBM
    max_leaves   = N_LEAVES,     # ← equivalent to LightGBM's num_leaves
    random_state = SEED,
    verbosity    = 0,
)
model_lgb = lgb.LGBMClassifier(
    **SHARED,
    num_leaves        = N_LEAVES,  # primary complexity control
    max_depth         = 3,        # -1 = no depth cap
    min_child_samples = 1,
    subsample_freq    = 0,
    random_state      = SEED,
    verbose           = -1,
)
```

#   - Depthwise splits every node at a given depth before going deeper.
#     It spends budget evenly across all regions, even unimportant ones.
#   - Leaf-wise always chases the single highest-gain leaf next.
#     With multiple hard sub-clusters, it will drill deep into the most
#     confusing region while barely touching easier ones.
