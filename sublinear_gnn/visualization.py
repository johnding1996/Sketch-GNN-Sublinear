from theory import gcn_response

import matplotlib.pyplot as plt


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