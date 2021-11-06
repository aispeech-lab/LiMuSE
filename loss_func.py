import torch
from itertools import permutations
from torch.autograd import Variable

EPS = 1e-8

def SI_SNR(_s, s, zero_mean=True):
    '''
         Calculate the SNR indicator between the two audios. 
         The larger the value, the better the separation.
         input:
               _s: Generated audio
               s:  Ground Truth audio
         output:
               SNR value 
    '''
    if zero_mean:
        _s = _s - torch.mean(_s)
        s = s - torch.mean(s)
    s_target = sum(torch.mul(_s, s))*s/torch.pow(torch.norm(s, p=2), 2)
    e_noise = _s - s_target
    return 20*torch.log10(torch.norm(s_target, p=2)/torch.norm(e_noise, p=2))


def permute_SI_SNR(_s_lists, s_lists):
    '''
        Calculate all possible SNRs according to 
        the permutation combination and 
        then find the maximum value.
        input:
               _s_lists: Generated audio list
               s_lists: Ground truth audio list
        output:
               max of SI-SNR
    '''
    length = len(_s_lists)
    results = []
    for p in permutations(range(length)):
        s_list = [s_lists[n] for n in p]
        result = sum([SI_SNR(_s, s) for _s, s in zip(_s_lists, s_list)])/length
        results.append(result)
    return max(results)


def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def si_snr_loss(ests, egs):
    # spks x n x S
    refs = egs["ref_wav"]
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = egs["mix_wav"].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def cal_sisnr_order_loss(source, estimate_source, source_lengths):
    max_snr = cal_si_snr_with_order(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(max_snr)
    return loss

def cal_si_snr_with_order(source, estimate_source, source_lengths):
    """Calculate SI-SNR with given order.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with order
    # reshape to use broadcast
    s_target = zero_mean_target  # [B, C, T]
    s_estimate = zero_mean_estimate  # [B, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]
    #print(pair_wise_si_snr)
    return torch.sum(pair_wise_si_snr,dim=1)/C

def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return Variable(mask).cuda()

if __name__ == "__main__":
    a_t = torch.tensor([1, 2, 3], dtype=torch.float32)
    b_t = torch.tensor([1, 4, 6], dtype=torch.float32)
    print(permute_SI_SNR([a_t], [b_t]))
