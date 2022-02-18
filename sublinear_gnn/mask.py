import numpy as np
from scipy.optimize import fsolve


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