import numpy as np
import torch
import torch.nn.functional as F
import random

def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def fix_python_seed(seed):
    print('seed-----------python', seed)
    random.seed(seed)
    np.random.seed(seed)


def fix_torch_seed(seed):
    print('seed-----------torch', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)

    return total_kld

def optimize_beta(beta, MI_loss,alpha2=1e-3):
    beta_new = max(0, beta + alpha2 * (MI_loss - 1) )

    # return the updated beta value:
    return beta_new

def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)


def slerp(low, high, batch):
    low = low.repeat(batch, 1)
    high = high.repeat(batch, 1)
    val = ((0.6 - 0.4) * torch.rand(batch,) + 0.4).cuda()
    omega = torch.acos((low*high).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1)*high
    return res


def get_source_centroid(feature, label, num_class, flag=True, centroids=None):
    new_centroid = torch.zeros(num_class, feature.size(1)).cuda()

    dist = 0
    for i in range(num_class):
        class_mask = (label == i)

        if flag:
            centroids = centroids.cuda()
            new_centroid[i] = 0.5 * torch.mean(feature[class_mask], dim=0) + 0.5 * centroids[i]

        else:
            new_centroid[i] = torch.mean(feature[class_mask], dim=0)
    dist += torch.mean(torch.nn.functional.pairwise_distance(feature[class_mask], new_centroid[i]))
    return new_centroid, dist

def optimize_beta(beta, distance,alpha2=0.5):
    beta_new = min(1, beta + alpha2 * distance )

    # return the updated beta value:
    return beta_new

def get_domain_vector_avg(feature, prototype, label, num_class):
    dist = torch.zeros(num_class).cuda()
    for i in range(num_class):
        class_feature = feature[label == i]
        dist[i] = torch.sqrt((prototype[i] - class_feature).pow(2).sum(1)).mean()
    return dist.mean() + dist.var()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError('ver == 1 or 2')

    return loss

def conditional_mmd_rbf(source, target, label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    for i in range(num_class):
        source_i = source[label==i]
        target_i = target[label==i]
        loss += mmd_rbf(source_i, target_i)
    return loss / num_class

def domain_mmd_rbf(source, target, num_domain, d_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    loss_overall = mmd_rbf(source, target)
    for i in range(num_domain):
        source_i = source[d_label == i]
        target_i = target[d_label == i]
        loss += mmd_rbf(source_i, target_i)
    return loss_overall - loss / num_domain

def domain_conditional_mmd_rbf(source, target, num_domain, d_label, num_class, c_label):
    loss = 0
    for i in range(num_class):
        source_i = source[c_label == i]
        target_i = target[c_label == i]
        d_label_i = d_label[c_label == i]
        loss_c = mmd_rbf(source_i, target_i)
        loss_d = 0
        for j in range(num_domain):
            source_ij = source_i[d_label_i == j]
            target_ij = target_i[d_label_i == j]
            loss_d += mmd_rbf(source_ij, target_ij)
        loss += loss_c - loss_d / num_domain

    return loss / num_class

def DAN_Linear(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    # Linear version
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_linear(src_fea, tar_fea):
    delta = (src_fea - tar_fea).squeeze(0)
    loss = torch.pow(torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1))),2)
    return torch.sqrt(loss)

def diverse_conditional_mmd(source, target, label, num_class, iter, d_label=None, num_domain=3):
    loss = 0
    selected_d = iter % num_domain
    for i in range(num_class):
        source_i = source[label == i]
        target_i = target[label == i]
        d_label_i = d_label[label == i]

        source_is = source_i[d_label_i == selected_d]
        target_is = target_i[d_label_i == selected_d]

        source_iu = source_i[d_label_i != selected_d]
        target_iu = target_i[d_label_i != selected_d]

        if source_is.size(0) > 0 and source_iu.size(0) > 0:
            loss += (mmd_rbf(source_iu, target_iu) - 0.4 * mmd_rbf(source_is, target_is))

    return torch.clamp_min_(loss / num_class, 0)


def entropy_loss(x):
    out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    out = -1.0 * out.sum(dim=1)
    return out.mean()

def reparametrize(mu, logvar, factor=0.2):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + factor*std*eps

def loglikeli(mu, logvar, y_samples):
    return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean()#.sum(dim=1).mean(dim=0)

def club(mu, logvar, y_samples):

    sample_size = y_samples.shape[0]
    # random_index = torch.randint(sample_size, (sample_size,)).long()
    random_index = torch.randperm(sample_size).long()

    positive = - (mu - y_samples) ** 2 / logvar.exp()
    negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
    upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    return upper_bound / 2.