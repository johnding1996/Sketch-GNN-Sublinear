import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


def gcn_response(size, freqs=None, avg_deg=None, nlayer=3, activation='ReLU'):
    """
    Compute the theoretical spectral response of GCN.
    
    :size: input graph size.
    :freqs: array of frequencies in acesding order. default evenly distributed between [0,2].
    :avg_deg: average degree of graph, default +inf.
    :nlayer: number of layer of GCN.
    :activation: activation function type, default ReLU.
    :return: response ratios of each frequencies.
    """
    # validate
    assert isinstance(size, int)
    assert size >= 1
    if freqs is not None:
        assert isinstance(freqs, np.ndarray)
        assert freqs.ndim == 1 and len(freqs) == size
        assert np.all(freqs[:-1] <= freqs[1:])
        assert np.min(freqs) >= 0 and np.max(freqs) <= 2
        freqs = freqs.astype(np.float64)
    else:
        freqs = np.linspace(0, 2, size, dtype=np.float64)
    if avg_deg is not None:
        assert isinstance(avg_deg, int) or isinstance(avg_deg, float)
        assert avg_deg >= 1
        avg_deg = float(avg_deg)
    else:
        avg_deg = np.Inf
    assert isinstance(nlayer, int) and nlayer >= 1
    assert activation in ['ReLU']
    # compute
    return np.power(np.abs(1-1/(1+1/avg_deg)*freqs)/np.sqrt(2), nlayer)


def _get_quant_mask(responses, offset, org_size):
    return np.clip(np.floor(offset - np.log2(responses)), 0, org_size)


def _get_compress_ratio(qmasks, org_size):
    return 1-np.sum(qmasks)/(org_size*len(qmasks))


def _get_func_offset_to_comp_ratio(responses, org_size):
    return np.vectorize(
        lambda x: _get_compress_ratio(
            _get_quant_mask(responses=responses, offset=x, org_size=org_size), 
            org_size=org_size
        )
    )


def _get_optimal_offset(comp_ratio, responses, org_size):
    func = _get_func_offset_to_comp_ratio(responses=responses, org_size=org_size)
    return fsolve(lambda x: func(x)-comp_ratio, x0=0., xtol=1e-5, factor=0.1)[0]


def quant_mask(responses, comp_ratio, nlayer=3, org_size=8):
    """
    Compute the quantization mask of each frequency.
    
    :responses: response ratios of each frequencies.
    :offset: desired compression ratio.
    :nlayer: number of layer of GCN.
    :org_size: original size in # of bits, we assume 1<= qmask <= org_size.
    :return: quantization mask of each frequencies.
    """
    # validate
    assert isinstance(responses, np.ndarray)
    assert np.min(responses) >= 0 and np.max(responses) <= 1
    assert isinstance(nlayer, int)
    assert nlayer >= 1
    # compute
    offset = _get_optimal_offset(comp_ratio, responses, org_size)
    return _get_quant_mask(responses, offset, org_size)


def plot_predicted_spectral_response(size, freqs, avg_deg, max_nlayer=3, activation='ReLU'):
    assert activation in ['ReLU']
    
    
    plt.figure(figsize=(10, 6))
    for nlayer in range(1, max_nlayer+1):
        responses = gcn_response(size, freqs, avg_deg, nlayer, activation)
        plt.plot(freqs, responses, label="{}-Layer GCN with ReLU".format(nlayer))
    plt.xlabel("Frequency", fontsize=14)
    plt.ylabel("Spectral Response", fontsize=14)
    plt.xlim([0, 2])
    plt.legend(fontsize=14)
    plt.show()


def plot_offset_vs_compress_ratio(size, freqs, avg_deg, nlayer=3, org_size=8, activation='ReLU'):
    responses = gcn_response(size, freqs, avg_deg, nlayer, activation)
    func = _get_func_offset_to_comp_ratio(responses, org_size)
    
    offset_min = np.floor(_get_optimal_offset(1.0, responses, org_size))
    offset_max = np.ceil(_get_optimal_offset(1.0/org_size, responses, org_size)) + 5

    offsets = np.linspace(offset_min, offset_max, 512)
    comp_ratios = func(offsets)

    plt.figure(figsize=(10, 6))
    plt.plot(offsets, comp_ratios)
    plt.xlabel("Offset", fontsize=14)
    plt.ylabel("Compress Ratio", fontsize=14)
    plt.show()
    
    
def plot_predicted_quantization_mask(size, freqs, avg_deg, nlayer, org_size, activation='ReLU', 
                                     comp_ratios=[1.0, 0.7, 0.5, 0.2, 0.0]):
    assert activation in ['ReLU']
    
    responses = gcn_response(size, freqs, avg_deg, nlayer, activation)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for comp_ratio in comp_ratios:
        qmasks = quant_mask(responses, comp_ratio, nlayer, org_size)
        ax1.plot(freqs, 1-qmasks/org_size, 
                 label="Comp. Ratio = {:.4f}".format(_get_compress_ratio(qmasks, org_size)))
    ax2 = ax1.twinx()
    ax2.plot(freqs, responses, '--', color='black', linewidth=2, 
             label="Predicted Spectral Response\n of {}-Layer GCN with ReLU".format(nlayer))
    ax1.set_xlabel("Frequency", fontsize=14)
    ax1.set_ylabel("Quantization Mask", fontsize=14)
    ax2.set_ylabel("Spectral Response", fontsize=14)
    ax1.set_yticks(np.arange(0, org_size+1)/org_size)
    ax1.set_yticklabels(np.power(2, np.arange(org_size, -1, -1)))
    ax1.set_xlim([0, 2])
    ax1.legend(fontsize=14)
    ax2.legend(fontsize=14)
    plt.show()