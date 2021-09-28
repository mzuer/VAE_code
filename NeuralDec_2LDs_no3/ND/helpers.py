import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli

def my_softplus(x):
    return torch.log(1.0 + torch.exp(x))

def calculate_KLqp(p, q, n_samples=100):
    sample_q = q.rsample(sample_shape=(n_samples,))
    logp = p.log_prob(sample_q)
    logq = q.log_prob(sample_q)
    KL = torch.mean(logq - logp)
    return KL


def approximate_KLqp(logitsp, logitsq):
    pp = torch.sigmoid(logitsp)
    qq = torch.sigmoid(logitsq)
    return (qq * (torch.log(qq) - torch.log(pp)) + (1 - qq) * (torch.log(1 - qq) - torch.log(1 - pp))).sum()

def rsample_RelaxedBernoulli(temperature, logits):
    p = LogitRelaxedBernoulli(temperature, logits=logits)
    return torch.sigmoid(p.rsample())

def expand_grid(a, b):
    nrow_a = a.size()[0]
    nrow_b = b.size()[0]
    ncol_b = b.size()[1]
    x = a.repeat(nrow_b, 1) # The number of times to repeat this tensor along each dimension 
    y = b.repeat(1, nrow_a).view(-1, ncol_b)
   # retuned as tuple
    return x, y

    
def KL_standard_normal(mu, sigma):
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    q = Normal(mu, sigma)
    return torch.sum(torch.distributions.kl_divergence(q, p))
    
def KL_standard_normal_2ld(mu_1, mu_2, sigma_1, sigma_2):  ### => to test -> should return something similar than calculate_KLqp ??!!!!
    p1 = Normal(torch.zeros_like(mu_1), torch.ones_like(mu_1))
    q1 = Normal(mu_1, sigma_1)
    p2 = Normal(torch.zeros_like(mu_2), torch.ones_like(mu_2))
    q2 = Normal(mu_2, sigma_2)
    kl1 = torch.sum(torch.distributions.kl_divergence(q1, p1))
    kl2 = torch.sum(torch.distributions.kl_divergence(q2, p2))
    return kl1+kl2 
