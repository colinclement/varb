---
layout: post
title:  "Variational Bayes Polynomial Fitting"
date:   2016-2-3 17:39:53
categories: [bayesian, inference]
---

We are going to use a general polynomial model to understand how to use variational bayes.
The model we will use is

$$\begin{equation}
    y(x; \theta) = \sum_k \theta_k x^k
\end{equation}$$

where $\theta_k$ is the coefficient of the $k^{th}$ order term of the polynomial.
If we then measure data $d = y(x) + \eta$ where $\eta \sim \mathcal{N}(0, \sigma^2)$ is
some uncorrelated noise, the probability that we measured $d$ given some points
$x$ and that our model is parameterized by $\theta$ is true is given by:

$$\begin{equation}
    p(d|\theta, x) \propto \exp{\frac{-1}{2 \sigma^2}||d - \sum_k \theta_k x^k||^2}
\end{equation}$$

We will assume in this discussion that $x$ is known and that its uncertainties are negligible,
so that it is merely conditions of the experiment.
Typical solutions of this problem begin by writing down Bayes theorem which states that

$$\begin{equation}
    p(\theta | d) = \frac{p(d|\theta)p(\theta)}{p(d)}
\end{equation}$$

which is referred to the posterior distribution. That is, $p(\theta|d)$ is the probability
of our model parameters adjusted for having measured $d$. $p(\theta)$ is encodes the prior information
we may or may not have about $\theta$, and $p(d)$ is called the `evidence,' where again we have implicitly
assumed constant $x$ for this experiment. $p(d)$ is useful for performing model selection, that is we can
vary the order of our model $k$ and check $p(d)$ to see which model is really more probable. Of course
most of these quantities are intractable to work with, so we will proceed a different way. Let us propose
a trial distribution $q(\theta|\lambda)$ with the intention of approximating $p(\theta|d,x)$. Using some
tricks below, we can derive a variational lower bound to the evidence $p(d)$.
We begin by writing $p(d)$ as the likelihood ($\mathcal{L}(\theta | d) \equiv p(d|\theta)$) as the
marginal probability over $\theta$:

$$\begin{eqnarray}
    \log p(d) &=& \log \int p(d|\theta)p(\theta) d\theta \\
              &=& \log \int p(d|\theta)\frac{q(\theta|\lambda)}{q(\theta|\lambda)} p(\theta)d\theta\\
              &\geq& \int q(\theta|\lambda) \log \frac{p(d|\theta)p(\theta)}{q(\theta|\lambda)}d\theta \\
    \log p(d) &\geq& \langle \log p(d|\theta)\rangle_{q(\theta|\lambda)} - \text{KL}(q(\theta|\lambda)||p(\theta)) \equiv - \mathcal{F}
\end{eqnarray}$$

The third line inequality was achieved using Jensen's inequality, which works due to the concavity 
of $\log(x)$. We define this quantity $\mathcal{F}$ to be the variational free energy.
We can vary some parametrization of $q(\theta|\lambda)$ for $\lambda$
to maximize the right hand side above, which will form a lower bound on the evidence.
That is,

$$\begin{eqnarray}
    \lambda^* &=& \text{argmin}_\lambda \mathcal{F}(\lambda, d)\\
    \text{so} \log p(d) &\geq& \mathcal{F}(\lambda^*,d)
\end{eqnarray}$$

And if we did a good job parameterizing $q$ and optimizing $\mathcal{F}$ then we could have a good lower bound.
We get a number of things from this approach:

1. An estimate for $p(d)$, with which we can perform model selection.
2. $p(\theta\vert d)$ is estimated by $q(\theta\vert\lambda)$.
3. The optimal $q(\theta\vert\lambda)$ gives us estimate of 
the best $\theta$ including error estimates.

We will make the simplifying assumption that $p(\theta)$ is constant, then the KL-divergence
simply becomes the entropy of $q$. Therefore when optimizing this function $\mathcal{F}$ we can 
interpret the two terms:

$$\begin{eqnarray}

    \mathcal{F} &=& \frac{1}{2\sigma^2}\int q(\theta|\lambda) ||d - \sum_k \theta_k x^k||^2 + \int q(\theta|\lambda)\log \frac{1}{q(\theta|\lambda)}
                &=& \frac{1}{2 \sigma^2}\left(\langle E \rangle_q - 2\sigma^2S[q]\right)
\end{eqnarray}$$

Where we are writing $p(d\vert\theta) \propto \exp(-\beta E)$ and $S$ is the entropy functional.
The first term tries to minimize the 'energy' of our probability distribution, that is it tries to match
our data and reduce the common cost function. The second term is called the
'complexity cost' and prevents overfitting by penalizing distributions with large entropy, or models
which try to explain large swaths of $\theta-$space.

All of the analysis is of course predicated on the simplicity or tractability of $q$. We will therefore
assume a product of independent Gaussian distributions, one for each coefficient $\theta_k$:

$$\begin{equation}
    q(\theta|\lambda=\{\mu, \sigma\}) = \prod_k \frac{1}{\sqrt{2 \pi \sigma_k^2}}\exp{-\frac{(\theta_k - \mu_k)^2}{2\sigma_k^2}}
\end{equation}$$

In this form the entropy is $S[q] = \frac{1}{2}(N + \sum_k \log 2\pi \sigma_k^2) = \sum_k \log \sigma_k + C$ where $C$ is some constant that is irrelevant for optimization. In this simple form we see simply that minimizing the variational free energy will prefer models with small $\sigma$'s.
The next question is: how do we optimize this quantity when we cannot marginalize over 


