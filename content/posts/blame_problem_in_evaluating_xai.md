+++
title = 'Toward Unbiased Evaluation of Local Explanations: How to Tackle the Blame Problem'
date = 2023-12-14T12:00:47+01:00
draft = false
+++

Note: This blog post summarizes my research paper, "The Blame Problem in Evaluating Local Explanations and How to Tackle It". The article is available on [Arxiv] (https://arxiv.org/abs/2310.03466) and was accepted in ECAI XAI^3 Workshop. The workshop website can be accessed [here](https://xai3ecai2023.github.io/). 


### TLDR;
The evaluation measures of local explanations suffer from a problem called blame problem. The blame problem is when we blame the explanation wrongfully for their unfaithful explanations when the fault can be traced back to the black-box model as the oracle. The blame problem can be avoided when ground truth feature importance scores are directly extracted from the model we aim to explain. 


### Introduction
Explainable Artificial Intelligence (xAI) is the study or understanding complex machine learning models. The complexity is, of course with respect to two main factors: the size of parameters and models optimization algorithms. Before xAI, interpretability was only achieved using interpretable models. The problem is that using interpretable models does not guarantee high accuracy due to their limited capacity. Simply put, Explainable AI promises that we can achieve interpretability without losing the ability to keep using black-box models.

Local explanation techniques are among the most popular tools in Explainable Artificial Intelligence (xAI) toolkits. They provide information about the black-box model's prediction of a single instance in the dataset. Since the introduction of LIME [1], more and more explanation techniques have been introduced. Studies such as [2, 3] have shown that these explanation techniques fail at times. However, the majority of these evaluations rely on intuition, not rigor. 

We now have a flood of new explanation techniques that are introduced. Only in one study [2], the authors listed more than 29 explanation techniques:

![List of Explanation Techniques](/list_of_explanations.png)


And the not-surprising fact is that they largely disagree with one another. Local explanations even disagree when explaining the prediction of Random Forest model for the test instance number ten from the Iris dataset with respect to its predicted class:

![List of Explanation Techniques](/lime_vs_kernel_shap.png)

The million-dollar question is: "Which one of these explanations is more faithful to the model?". Even though researchers have yet to really answer this question, more and more explanation techniques are being introduced. I suspect that the reason is that,

> In the absence of reliable evalaution measures, the bar is very low for introducing new explanation techniques. As a result, it is also hard to see whether new explanation techniques are significantly better than their predecessors.  

What is the challenge in evaluating local explanations? The most optimal and rigorous way to evaluate the local explanations is to measure their similarity to the ground truth feature importance score. These ground truth scores need to be obtained **directly** from the black-box model. But this is not possible. Had we had access to such ground truth feature importance scores, we would not need local explanations to begin with. So, it is important to know that there is a limit to what we can evaluate.

Faced with the challenge of directly extracting ground truth feature importance scores from black-box models, the majority of evaluation measures rely on indirect methods, e.g., using a black-box model as the oracle, contamination of the explained models, or even using human subjects. Let us first provide a taxonomy of the evaluation techniques used for local model-agnostic explanations: 

* Robustness Measures
* Model Randomization
* Human-grounded Evaluation 
* Using Ground Truth
    * from Synthetic Data
    * from Interpretable Models

The first three categories are indirect measures and do not evaluate local explanations with any notion of ground truth. In this post, we want to highlight that the first three categories, along with "Using Ground Truth from Synthetic Data", suffer from a problem we call the *blame problem*. The blame problem is when we are evaluating local explanations. Still, we may end up blaming the explanation technique for providing an unfaithful explanation *wrongfully* when the fault actually comes from the explained model. Let us briefly describe each category of measures and show the appearing blame problem in all these subcategories.


### Robustness measure 
Robustness measures are the most popular type of evaluation measures for local explanations. These measures use the black-box model as an oracle for evaluating the faithfulness of local explanations. They have two different types: Evaluation by Perturbation and Stability. In the **Evaluation by Perturbation**, the important (unimportant) features from the local explanation are taken and nullified (perturbed) in the explained instance. The change in the predicted score of a black-box model before and after this nullification is calculated. The *assumption* behind these measures is that black-box predicted scores must show large (insignificant) change after perturbing importance (least important) features from faithful local explanations. There are numerous variations of these measures: Importance by Deletion (Preservation) [3], Prediction Gap on Important (Unimportant) Feature Perturbation [4] or even (In)fidelity [5], which is slightly different than the rest, but operates on similar perturbation logic.

The second class of these measures, i.e., **Stability**, measures the similarity between 1) the local explanation of an instance and 2) the local explanations of the same instance to which insignificant noise is added. The assumption is that a faithful explanation provides a significantly similar explanation for the explained instance and its noisy variation. Measures such as Local Lipshitz [6], and Stability [7] are examples of these measures.

As it is clear, we rely heavily on the black-box model in both classes of Robustness Measures. In the evaluation by perturbation, we rely on the black-box model, providing calibrated and accurate predictions to instances after their important (unimportant) features are nullified or perturbed. In the stability measures, we expect the black-box model to be robust to noise. Otherwise, black-box models can provide inaccurate predictions when small levels of noise are added to the instance. In the worse case, the noise can cause the instance to become an adversarial example.

So, if the predictions of the black-box as an oracle are not reliable, we end up blaming explanations. That is the blame problem. In the Figure, we show an example of perturbing superpixels of an image with the predicted class of indigo bunting by the ResNET model. The highlighted yellow lines are superpixel features in this instance. These superpixels are frequently used in local explanations, e.g., in LIME and SHAP. 


![Bird Image with Superpixels](/test_image_superpixel.jpeg)

Now let us perturb some of these features:

![Perturbed Image](/perurbed_6_with_labels.jpeg)

We can see that the label of some of the perturbed images, e.g., the first image from the bottom row, does not match the image. There is no feature that shows a bird in the image, and ResNET has memorized the background of the image. Even though these features are correctly detected to be important, the minimal changes in the predicted class will make us blame the explanation for being unfaithful when we use the robustness measures.

For the stability example, we show the example from [7]. Evaluating the Stability of the explanations of the explained instance (marked with a dotted circle)  relies heavily on whether the model is robust in the Euclidean ball of this instance's decision surface. In this case, even small noise can provide black-box predicted output that are largly different than the original instance. With large changes in the predicted score, local explanations can provide very different explanations. Therefore, we can end up blaming the local explanations for instability, whereas the real problem lies in the lack of robustness of the model. 

![Perturbed Image](/continuity_limitations.png)

### Model Randomization

In some studies, local explanations are evaluated before and after some randomized contamination process is induced on a black-box model. The assumption is that local explanations need to show significantly different explanations for these two models, the uncontaminated and contaminated ones. An example of this evaluation is shown in [8]. The authors proposed two of these contamination tests. The first is to randomly re-initialize the weights of neural network models. The other is to independently randomize the weights of a single layer one at a time. Using these randomization tests, the authors show that a large number of local explanations fail to provide significantly different explanations for the uncontaminated and the contaminated models. 

In this case, we again face the blame problem again. We know that black-box models, and especially deep neural networks, tend to memorize and extract accurate knowledge even from random or corrupted labels [9]. Because of this, there is no guarantee that randomizations have necessarily obfuscated the workings of the black-box models enough to induce changes to local explanations after the contamination process. We then might blame local explanations in these scenarios even though to the explanation techniques, both models are functioning equally correctly. 

### Human-grounded Evaluation 
So far, all evaluation measures have been systematic. However, some studies use human subjects to evaluate local explanations. Human ground evaluation of local explanations is usually performed with two types of tasks: 1) Model replay scenario [10] and 2) Agreement with human logic [11]. 

In the first approach, a human subject is presented with a local explanation and an explained instance, and the human subjects are asked to replicate the prediction of the black-box model. The assumption is that if human users can replay the model accurately, these local explanations are accurate. In the second approach, we measure the similarity of local explanations to the explanations obtained from the human subjects for the task. Large agreement (similarity) means that explanations are faithful.

The blame problem in the first case is related to the case where we cannot measure the human subjects' priors on the data, model, and task at hand. In other words, how much of the failure in human subjects are related to their priors and not the local explanation? In this case, we might end up blaming the local explanations. The second approach also suffers from the blame problem because we have no measure of the similarity of the human logic and the internal logic of the black-box model we explain. Local explanations are supposed to be faithful to models, and humans and models have very different logic for performing tasks [12]. 

### Using Ground Truth 
As we mentioned earlier, the most optimal way to evaluate local explanations is to have access to ground truth feature importance scores. Since we cannot extract these from black-box models, some studies have aimed to extract them by simplifying the dataset or simplifying the explained model to interpretable alternatives. 

#### Synthetic Dataset 
In [13, 14, 15], the authors evaluate local explanations by obtaining ground truth importance scores from synthetic datasets. In the majority of these studies, the data generation process is a polynomial function in the form of $Y = W_0 X_0 + W_1 X_0 + ... W_n X_n$. After that, a black-box model is trained on this dataset, and local explanations $\Phi$ are obtained. Using a similarity metric $d$, the faithfulness of local explanations is calculated as $d(\Phi, W)$ where $W = [W_0, ..., W_n]$ is the prior weight vector of the data generation process. The assumption behind this evaluation measure is that faithful explanations will provide $\Phi$ values that are similar to $W$. One benefit of using these measures is that we can evaluate any type of black-box model using ground truth importance scores *directly*.

In this type of evaluation, the blame problem happens again. Particularly when the black-box model does not necessarily learn the prior weights in the data generation function but yet is accurate in predicting $Y$. Since the majority of black-box models have a parameter size that is larger than $n$, this is a prevalent case. In the Figure below, we visualize two examples of synthetic datasets from [10, 11]. The decision boundary of the neural network models trained on these datasets are shown in the figures, and on top of each instance, we can see its ground truth importance scores: 

Orange Label Dataset [13]             |  Gaussian Dataset [15]
:-------------------------:|:-------------------------:
![](/orange_label.png)  |  ![](/synthetic_example_openxai_synthetic.png)


From the Figure, it is visible that the decision surface of the model are **not** in agreement with these ground truth scores. For example, in the Orange Skin dataset, even though both features are considered equally important, changing the feature along the y-axis has no significant effect on the predicted score of those instances. So, in this case, we will blame the local explanations even if they accurately set low importance scores to the feature along the y-axis in the explanations of those instances. 


#### Interpretable Models
A new set of studies proposes that we should obtain ground truth importance scores from simpler models. In these evaluation measures, we have no restriction on the datasets we evaluate the local explanations. The ground truth importance scores are directly from the model we explain. In this case, the models are interpretable. Similar to the previous case, using a similarity metric $d$, we evaluate $d(\Phi, W)$ where $W$ are the extracted ground truth feature importance scores from the interpretable models. When $\Phi$ and $W$ have large similarity measures, we can conclude that the local explanation is faithful.

The blame problem does not appear in this category of evaluation measures. This is because we are extracting these ground truth importance scores directly from the models we explain, and we no longer depend on the data or the black box as an oracle. 

The number of studies using this type of evaluation measure are limited. In [15], the authors use the weight of Logistic Regressions as the ground truth feature importance scores. In [16], the authors show that these weights do not have a local interpretation. They propose Model-Intrinsic Additive Scores (MIAS) that are extracted from prediction functions of Logistic and Linear Regression and Gaussian Naive Bayes models. In fact, MIAS is extracted from additive scores that appear in prediction functions of these models. See the Figure below from [16] that shows the MIAS scores for a Logistic Regression model. The ground truth feature importance scores are visualized on top of the instance. We can see that the ground truth importance scores correctly are in agreement with the decision surface of the model. 

![Mias Scores](/mias_scores.png)

We would like to highlight that even though there is no blame problem in this category of evaluation measures, there are **other limitations** associated with them. A local explanation technique that provides faithful explanations when explaining Linear Regression is not necessarily faithful when explaining a deep neural network. Another important notion is that, even though we do have ground truth feature importance scores in this case, the notion of similarity metric can affect the evaluation process. In [16], the authors show an example of this.


### Conclusion
In our post, we showed that numerous categories of evaluation measures, e.g., Robustness measures, model randomization, human-grounded evaluation, and using Ground Truth from Synthetic Data, suffer from a blame problem. The blame problem is when we blame the explanation wrongfully for their unfaithful explanations when the fault can be traced back to the black-box model.

The only category of evaluation measures that do not have the blame problem is "Using ground truth from interceptable models". This is because we are directly extracting ground truth feature importance scores from the model we explain without altering its state, like in the model randomization measures.

However, this category of evaluation measures has further limitations. Explanation techniques that provide faithful explanations to Logistic Regression do not necessarily have faithful explanations when explaining neural networks.

Therefore, 

> Evaluating local explanations remains an open and important research problem.


### Citation:

In case you would like to cite my work, please use the following Bibtex: 

```
@article{rahnama2023blame,
  title={The Blame Problem in Evaluating Local Explanations, and How to Tackle it},
  author={Rahnama, Amir Hossein Akhavan},
  journal={Workshop: XAI methods, challenges and applications, 26th European Conference on Artificial Intelligence},
  year={2023}
}

```

### References: 

[1]: Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "" Why should i trust you?" Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 2016.

[2]: Covert, Ian, Scott Lundberg, and Su-In Lee. "Explaining by removing: A unified framework for model explanation." Journal of Machine Learning Research 22.209 (2021): 1-90.

[3]: Fong, Ruth C., and Andrea Vedaldi. "Interpretable explanations of black boxes by meaningful perturbation." Proceedings of the IEEE international conference on computer vision. 2017.

[4]: Agarwal, Chirag, et al. "Openxai: Towards a transparent evaluation of model explanations." Advances in Neural Information Processing Systems 35 (2022): 15784-15799.

[5]: Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity of explanations." Advances in Neural Information Processing Systems 32 (2019).

[6]: Alvarez-Melis, David, and Tommi S. Jaakkola. "On the robustness of interpretability methods." arXiv preprint arXiv:1806.08049 (2018).

[7]: Agarwal, Chirag, et al. "Rethinking stability for attribution-based explanations." arXiv preprint arXiv:2203.06877 (2022).

[8]: Adebayo, Julius, et al. "Sanity checks for saliency maps." Advances in neural information processing systems 31 (2018).

[9]: Zhang, Chiyuan, et al. "Understanding deep learning (still) requires rethinking generalization." Communications of the ACM 64.3 (2021): 107-115.

[10]: Poursabzi-Sangdeh, Forough, et al. "Manipulating and measuring model interpretability." Proceedings of the 2021 CHI conference on human factors in computing systems. 2021.

[11]: Lundberg, Scott M., et al. "From local explanations to global understanding with explainable AI for trees." Nature machine intelligence 2.1 (2020): 56-67.

[12]: Dai, Wang-Zhou, et al. "Bridging machine learning and logical reasoning by abductive learning." Advances in Neural Information Processing Systems 32 (2019).

[13]: Chen, Jianbo, et al. "Learning to explain: An information-theoretic perspective on model interpretation." International conference on machine learning. PMLR, 2018.

[14]: Guidotti, Riccardo. "Evaluating local explanation methods on ground truth." Artificial Intelligence 291 (2021): 103428.

[15]: Agarwal, Chirag, et al. "Openxai: Towards a transparent evaluation of model explanations." Advances in Neural Information Processing Systems 35 (2022): 15784-15799.

[16]: Rahnama, Amir Hossein Akhavan, et al. "Can local explanation techniques explain linear additive models?." Data Mining and Knowledge Discovery (2023): 1-44.

