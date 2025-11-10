import os
import torch
# from torch import nn
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import setproctitle
# from datetime import datetime
# from sklearn.decomposition import TruncatedSVD
# from collections import defaultdict, Counter
# import scipy.sparse as sp
# from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import cosine_similarity
# from recbole.data.interaction import Interaction
# import psutil


"""
data augmentation util
"""
# add noise
def gaussian_noise(source, noise_base=0.1, dtype=torch.float32):
    x = noise_base + torch.zeros_like(source, dtype=dtype, device=source.device)
    noise = torch.normal(mean=torch.tensor([0.0]).to(source.device), std=x).to(source.device)
    return noise

def add_gaussian_noise(source, noise_base=0.1, dtype=torch.float32):
    noise = gaussian_noise(source, noise_base, dtype)
    source = source + noise
    return source

def mul_gaussian_noise(source, noise_base=0.1, dtype=torch.float32):
    noise = gaussian_noise(source, noise_base, dtype)
    source = torch.mul(source, 1 + noise)
    return source


# representation
def representation(config, count_dict, interaction, embeddings):
    if not config['open_represent']:
        return

    if count_dict['idx'] and not (count_dict['idx'] % 20 == 0 or count_dict['idx'] == 2):
        count_dict['idx'] += 1
        return
    else:
        count_dict['idx'] += 1

    if not config['rep_index']:
        config['rep_index'] = 0

    counts = interaction['iu_count']
    path = r'{}/{}_{}.png'.format(config['represent_path'], config['dataset'], str(config['rep_index']))
    # begin
    counts = counts * 100 / counts.max()

    xs1, ys1 = [], []
    xs2, ys2 = [], []
    xs3, ys3 = [], []
    xs4, ys4 = [], []

    for idx in range(embeddings.shape[0]):
        # i = pos_item[idx]
        in_ = counts[idx]
        i_e = embeddings[idx]
        i_e = i_e.view(2, -1).detach()
        if in_ <= 25:
            xs1.extend(i_e[0].cpu().numpy().tolist())
            ys1.extend(i_e[1].cpu().numpy().tolist())
        elif in_ <= 50:
            xs2.extend(i_e[0].cpu().numpy().tolist())
            ys2.extend(i_e[1].cpu().numpy().tolist())
        elif in_ <= 75:
            xs3.extend(i_e[0].cpu().numpy().tolist())
            ys3.extend(i_e[1].cpu().numpy().tolist())
        elif in_ <= 100:
            xs4.extend(i_e[0].cpu().numpy().tolist())
            ys4.extend(i_e[1].cpu().numpy().tolist())

    # g = sns.JointGrid()
    # sns.set_style("darkgrid")  # darkgrid, whitegrid, dark, white, ticks
    # # sns.set_context("notebook")
    # sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 3})
    #
    # color = '#fde725'  # yellow
    # # df = pd.DataFrame({"x": xs1, "y": ys1})
    # sns.scatterplot(x=xs1, y=ys1, ax=g.ax_joint, color=color)
    # sns.kdeplot(x=xs1, ax=g.ax_marg_x, color=color)
    # sns.kdeplot(y=ys1, ax=g.ax_marg_y, color=color)
    #
    # color = '#35b779'  # green
    # sns.scatterplot(x=xs2, y=ys2, ax=g.ax_joint, color=color)
    # sns.kdeplot(x=xs2, ax=g.ax_marg_x, color=color)
    # sns.kdeplot(y=ys2, ax=g.ax_marg_y, color=color)
    #
    # color = '#440154'  # violet
    # sns.scatterplot(x=xs3, y=ys3, ax=g.ax_joint, color=color)
    # sns.kdeplot(x=xs3, ax=g.ax_marg_x, color=color)
    # sns.kdeplot(y=ys3, ax=g.ax_marg_y, color=color)
    #
    # color = '#31688e'  # blue
    # sns.scatterplot(x=xs4, y=ys4, ax=g.ax_joint, color=color)
    # sns.kdeplot(x=xs4, ax=g.ax_marg_x, color=color)
    # sns.kdeplot(y=ys4, ax=g.ax_marg_y, color=color)
    #
    # # plt.colorbar()
    # g.set_axis_labels('', '')
    # plt.savefig(path)
    # plt.close()
    #
    # config['rep_index'] += 1

def node_dropout(seq, seq_len, dropout_rate=0.1):
    r"""Randomly discard some points.
    """
    seq_len = torch.clone(seq_len) # clone
    mask = torch.rand([seq.shape[0],seq.shape[1]]).to(seq.device) >= dropout_rate
    mask[:, 0] = True
    seq1 = torch.mul(seq, mask)
    arr = seq1.tolist()

    for i in range(len(arr)):
        row = arr[i]
        lens = seq_len[i]
        for j in range(lens-1):
            # first value is skipped
            j += 1
            item = row[j]
            if item == 0:
                lens -= 1
                del row[j]
                row.append(0)
    seq = torch.tensor(arr).to(seq.device)
    return seq, seq_len

# def calc_similarity_batch(a, b):
#     representations = torch.cat([a, b], dim=0)
#     return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
#
# def cos_sim(z1: torch.Tensor, z2: torch.Tensor):
#     '''
#     cos similarity
#     '''
#     z1, z2 = z1.unsqueeze(1), z2.unsqueeze(0)
#     z1 = F.normalize(z1)
#     z2 = F.normalize(z2)
#     sim = torch.mm(z1, z2.t())
#     return sim

# def sim(z1: torch.Tensor, z2: torch.Tensor, temp=1.0, simf='cos'):
#     z = torch.cat((z1, z2), dim=0)  # 2B * D
#     # sim = torch.mm(z, z.T) / temp  # 2B * 2B
#     if simf == 'cos':
#         # dim: 0: cos similarity is calculated between column vectors of corresponding columns
#         # 1(default): calculates the similarity between row vectors
#         # -1: is to calculate the similarity between row vectors
#         sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
#     elif simf == 'dot':
#         sim = torch.mm(z, z.T) / temp
#     else:
#         sim = torch.mm(z, z.T) / temp
#     return sim
#
# def cal_item_x_y_count(config, item_seq, item_seq_len, method_arr):
#     item_x_y_count_path = r'{}/{}_item_x_y_count.npy'.format(config['checkpoint_dir'], config['dataset'])
#     if os.path.exists(item_x_y_count_path):
#         item_x_y_count = np.load(item_x_y_count_path)
#     else:
#         # x_y_count = {x_y: count}
#         item_x_y_count = {}
#         for i in range(len(item_seq)):
#             seq = item_seq[i]
#             seq_len = item_seq_len[i]
#             if seq_len <= 1:
#                 continue
#
#             for i1 in range(seq_len):
#                 x = seq[i1]
#                 for i2 in range(seq_len - 1):
#                     y = seq[i2 + 1]
#                     if x == y:
#                         continue
#                     key = (str(x.item()) + '_' + str(y.item())) if y < x else (str(y.item()) + '_' + str(x.item()))
#                     if item_x_y_count.__contains__(key):
#                         item_x_y_count[key] += 1
#                     else:
#                         item_x_y_count[key] = 1
#
#         # sort by count(r) reverse
#         item_x_y_count = np.array(sorted(item_x_y_count.items(), key=lambda d: d[1], reverse=True))
#         np.save(item_x_y_count_path, item_x_y_count)
#
#     # item_x_y_count
#     # train_data.dataset.item_x_y_count = item_x_y_count
#     # print(item_x_y_count)
#     replace_rate = float(method_arr[2])
#     item_x_y_count_mini = item_x_y_count[0:int(len(item_x_y_count) * replace_rate)]
#     item_x_y_dict = {}
#     for i in item_x_y_count_mini:
#         key = int(i[0].split('_')[0])
#         val = int(i[0].split('_')[1])
#         if item_x_y_dict.__contains__(key):
#             item_x_y_dict[key].append(val)
#         else:
#             item_x_y_dict[key] = [val]
#     return item_x_y_dict

# def startCommonSet(model,dataset,config_dict):
#     # set proc
#     setproctitle.setproctitle("RS@" + model + "." + dataset)
#     # check dir log,saved
#     if not os.path.exists("./log"):
#         os.makedirs("./log")
#     if not os.path.exists("./saved"):
#         os.makedirs("./saved")
#
#     print("running_flag:", config_dict['running_flag'])
#
#     # Gets the current process object
#     current_process = psutil.Process()
#     print("current_process:", current_process)
#
# def checkRunningFlag(running_flag):
#     try:
#         runningFlagPath = "./zone/running_flag.txt"
#         if os.path.exists(runningFlagPath):
#             with open(runningFlagPath, 'r') as rf:
#                 text = rf.read()
#                 if text.__contains__(running_flag):
#                     # del running flag
#                     with open(runningFlagPath, 'w') as wf:
#                         wf.write('SUCCESS')
#                     return True
#     except Exception as e:
#         pass
#     return False
#
# def log_result(logger,set_color,log_dir,config,test_result):
#     testRes = None
#     for key in test_result:
#         # tools
#         if (str.startswith(key, 'hit@')
#                 or str.startswith(key, 'ndcg@')
#                 or str.startswith(key, 'gauc')):
#             if testRes == None:
#                 testRes = ''
#             else:
#                 testRes += '	'
#             testRes += str(test_result[key])
#
#     logger.info(set_color('test value', 'red') + f': {testRes}')
#     configVal = joint_config(config)
#     logger.info(set_color('config value', 'red') + f': {configVal}')
#
#     with open(f'{log_dir}/{config["running_flag"]}RESULT.log', 'a') as f:
#         f.write(f'test value: {testRes} \n')
#         f.write(f'config value: {configVal} \n\n')
#
#
# def joint_config(config):
#     config_ = 'model:' + str(config['model']) +',dataset:' + str(config['dataset'])
#     if config['batch_size']:
#         config_ += ',batch_size:' + str(config['batch_size'])
#     if config['loss_func_temp']:
#         config_ += ',loss_func:' + str(config['loss_func_temp'])
#     if config['open_cl']:
#         config_ += ',open_cl:' + str(config['open_cl'])
#     if config['data_aug_method']:
#         config_ += ',data_aug:' + str(config['data_aug_method'])
#     if config['attn_dropout_prob']:
#         config_ += ',dropout:' + str(config['attn_dropout_prob'])
#     if config['hidden_size']:
#         config_ += ',hidden_size:' + str(config['hidden_size'])
#     if config['phi']:
#         config_ += ',phi:' + str(config['phi'])
#     if config['tf_weight']:
#         config_ += ',tf_weight:' + str(config['tf_weight'])
#     if config['gnn_weight']:
#         config_ += ',gnn_weight:' + str(config['gnn_weight'])
#     if config['reg_weight']:
#         config_ += ',reg_weight:' + str(config['reg_weight'])
#     if config['cl_weight']:
#         config_ += ',cl_weight:' + str(config['cl_weight'])
#     if config['lambda1']:
#         config_ += ',lambda1:' + str(config['lambda1'])
#     if config['negative_sample_batch']:
#         config_ += ',negative_sample_batch:' + str(config['negative_sample_batch'])
#     if config['nd_rate']:
#         config_ += ',nd_rate:' + str(config['nd_rate'])
#     if config['noise_base']:
#         config_ += ',noise_base:' + str(config['noise_base'])
#     if config['pgd']:
#         config_ += ',pgd:' + str(config['pgd'])
#     if config['noise_grad_base']:
#         config_ += ',noise_grad_base:' + str(config['noise_grad_base'])
#     if config['simf']:
#         config_ += ',simf:' + str(config['simf'])
#
#     return config_
#
# def cal_item_seq_count(config, item_seq, item_seq_len, method_arr):
#     item_seq_dict = None
#     return item_seq_dict
#
# def rand_sample(high, size=None, replace=True):
#     r"""Randomly discard some points or edges.
#     Args:
#         high (int): Upper limit of index value
#         size (int): Array size after sampling
#     Returns:
#         numpy.ndarray: Array index after sampling, shape: [size]
#     """
#     a = np.arange(high)
#     sample = np.random.choice(a, size=size, replace=replace)
#     return sample
#
# def cal_alignment_and_uniformity(z_i, z_j, origin_z, batch_size):
#     """
#     We do not sample negative examples explicitly.
#     Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
#     """
#     N = 2 * batch_size
#
#     z = torch.cat((z_i, z_j), dim=0)
#
#     # pairwise l2 distace
#     sim = torch.cdist(z, z, p=2)
#
#     sim_i_j = torch.diag(sim, batch_size)
#     sim_j_i = torch.diag(sim, -batch_size)
#
#     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
#     alignment = positive_samples.mean()
#
#     # pairwise l2 distace
#     sim = torch.cdist(origin_z, origin_z, p=2)
#     mask = torch.ones((batch_size, batch_size), dtype=bool)
#     mask = mask.fill_diagonal_(0)
#     negative_samples = sim[mask].reshape(batch_size, -1)
#     uniformity = torch.log(torch.exp(-2 * negative_samples).mean())
#
#     return alignment, uniformity
#
# def tranfer_dict(parameter_dict,parameter_dict1):
#     if parameter_dict1:
#         for key in parameter_dict1.keys():
#             parameter_dict[key] = parameter_dict1[key]
#     return parameter_dict
#
# def exportData(model,dataset,log_dir,config):
#     print('===exportData===')
#     # build data
#     embedding_matrix = model.item_embedding.weight[1:].cpu().detach().numpy()
#     svd = TruncatedSVD(n_components=2)
#     svd.fit(embedding_matrix)
#     comp_tr = np.transpose(svd.components_)
#     proj = np.dot(embedding_matrix, comp_tr)
#
#     cnt = {}
#     for i in dataset['item_id']:
#         if i.item() in cnt:
#             cnt[i.item()] += 1
#         else:
#             cnt[i.item()] = 1
#
#     freq = np.zeros(embedding_matrix.shape[0])
#     for i in cnt:
#         freq[i - 1] = cnt[i]
#
#     # freq /= freq.max()
#
#     # export data
#     x = proj[:, 0]
#     y = proj[:, 1]
#     colors = freq
#
#     myStyle = {
#         "figure.facecolor": "white",
#         "axes.labelcolor": ".8",
#         "xtick.direction": "out",
#         "ytick.direction": "out",
#         "xtick.color": ".8",
#         "ytick.color": ".8",
#
#         "axes.axisbelow": True,
#         "grid.linestyle": "--",
#
#         "text.color": ".8",
#         "font.family": ["sans-serif"],
#         "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
#
#         "lines.solid_capstyle": "round",
#         "patch.edgecolor": "w",
#         "patch.force_edgecolor": True,
#
#         "image.cmap": "rocket",
#
#         "xtick.top": False,
#         "ytick.right": False,
#         "axes.grid": True,
#     }
#     # sns.set_style(myStyle)
#     # sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
#     # plt.figure(figsize=(6, 4.5))
#
#     # import pandas as pd
#     # reps_data = {'x': x, 'y': y, 'colors': colors}
#     # reps_df = pd.DataFrame(reps_data)
#     # dateStr = datetime.now().strftime("%Y%m%d%H%M%S")
#     # reps_df.to_csv(log_dir + f'/{config["model"]}-{config["dataset"]}_{dateStr}.csv')
#     #
#     # plt.scatter(x, y, s=1, c=colors, cmap='viridis_r')
#     # plt.colorbar()
#     # plt.xlim(-2, 2)
#     # plt.ylim(-2, 2)
#     # # plt.axis('square')
#     # # plt.show()
#     # # p1: item embedding representation
#     # plt.savefig(log_dir + f'/{config["model"]}-{config["dataset"]}_{dateStr}.pdf', format='pdf', transparent=False,
#     #             bbox_inches='tight')
#     # plt.savefig(log_dir + f'/{config["model"]}-{config["dataset"]}_{dateStr}.eps', format='eps', transparent=False,
#     #             bbox_inches='tight')
#     #
#     # from scipy.linalg import svdvals
#     # svs = svdvals(embedding_matrix)
#     # svs /= svs.max()
#     # np.save(log_dir + f'/{config["model"]}-{config["dataset"]}_svs_{dateStr}.npy', svs)
#     # svs_data = {'svs': svs}
#     # svs_df = pd.DataFrame(svs_data)
#     # svs_df.to_csv(log_dir + f'/{config["model"]}-{config["dataset"]}_svs_{dateStr}.csv')
#     #
#     # sns.set_style(myStyle)
#     # sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
#     # plt.figure(figsize=(6, 4.5))
#     # plt.plot(svs)
#     # # plt.show()
#     # # p2: Model training matrix singular value SVD.
#     # # The singular value of a matrix is an important concept in linear algebra. For an m×n matrix A, Singular Value Decomposition (SVD) is the decomposition of matrix A into the product of three matrices: A = UΣV^T, where U and V are orthogonal matrices and Σ is a diagonal matrix.
#     # # In singular value decomposition, the elements on the diagonal of Σ are called singular values of the matrix A. The singular values are arranged in order from greatest to smallest, indicating the importance of the matrix A in each dimension. The number of singular values is equal to the rank of matrix A.
#     # # Singular value decomposition is widely used in many fields, such as data dimensionality reduction, image compression, recommendation system, etc. It can extract the important features of the matrix, and the original matrix can be represented by an approximate matrix of lower dimensions, thus reducing the storage space and computational complexity.
#     # plt.savefig(log_dir + f'/{config["model"]}-{config["dataset"]}_svs_{dateStr}.pdf', format='pdf', transparent=False, bbox_inches='tight')
#     # plt.savefig(log_dir + f'/{config["model"]}-{config["dataset"]}_svs_{dateStr}.eps', format='eps', transparent=False, bbox_inches='tight')
#
# # DCRec data aug
# def build_external_data(config, train_data, test_data):
#     adj_graph, user_edges = build_adj_graph(train_data.dataset)
#     adj_graph_test, _ = build_adj_graph(test_data.dataset, "test")
#     sim_graph = build_sim_graph(train_data.dataset, config["sim_group"])
#     sim_graph_test = build_sim_graph(test_data.dataset, config["sim_group"], "test")
#
#     # model loading and initialization
#     external_data = {
#         "adj_graph": adj_graph,
#         "sim_graph": sim_graph,
#         "user_edges": user_edges,
#         "adj_graph_test": adj_graph_test,
#         "sim_graph_test": sim_graph_test
#     }
#     return external_data
#
# def build_adj_graph(dataset, phase="train"):
#     import dgl
#     # graph_file = dataset.config['data_path']+f"/adj_graph_{phase}.bin"
#     # user_edges_file = dataset.config['data_path']+"/user_edges.pkl.zip"
#     # try:
#     #     if phase == "test":
#     #         g = dgl.load_graphs(graph_file, [0])
#     #         return g[0][0], None
#     #     g = dgl.load_graphs(graph_file, [0])
#     #     user_edges = pd.read_pickle(user_edges_file)
#     #     print("loading graph from DGL binary file...")
#     #     return g[0][0], user_edges
#     # except:
#     print("constructing DGL graph...")
#     item_adj_dict = defaultdict(list)
#     item_edges_of_user = dict()
#     inter_feat = dataset.inter_feat
#     for line in range(len(inter_feat)):
#         item_edges_a, item_edges_b = [], []
#         uid = inter_feat[dataset.uid_field][line].item()
#         item_seq = inter_feat[dataset.item_id_list_field][line].tolist()
#         seq_len = inter_feat[dataset.item_list_length_field][line].item()
#         item_seq = item_seq[:seq_len]
#         for i in range(seq_len):
#             if i > 0:
#                 item_adj_dict[item_seq[i]].append(item_seq[i - 1])
#                 item_adj_dict[item_seq[i - 1]].append(item_seq[i])
#                 item_edges_a.append(item_seq[i])
#                 item_edges_b.append(item_seq[i - 1])
#             if i + 1 < seq_len:
#                 item_adj_dict[item_seq[i]].append(item_seq[i + 1])
#                 item_adj_dict[item_seq[i + 1]].append(item_seq[i])
#                 item_edges_a.append(item_seq[i])
#                 item_edges_b.append(item_seq[i + 1])
#
#         item_edges_of_user[uid] = (np.asarray(item_edges_a, dtype=np.int64), np.asarray(item_edges_b, dtype=np.int64))
#     item_edges_of_user = pd.DataFrame.from_dict(item_edges_of_user, orient='index',
#                                                 columns=['item_edges_a', 'item_edges_b'])
#     # item_edges_of_user.to_pickle(user_edges_file)
#     cols = []
#     rows = []
#     values = []
#     for item in item_adj_dict:
#         adj = item_adj_dict[item]
#         adj_count = Counter(adj)
#
#         rows.extend([item] * len(adj_count))
#         cols.extend(adj_count.keys())
#         values.extend(adj_count.values())
#
#     adj_mat = csr_matrix((values, (rows, cols)), shape=(
#         dataset.item_num + 1, dataset.item_num + 1))
#     adj_mat = adj_mat.tolil()
#     adj_mat.setdiag(np.ones((dataset.item_num + 1,)))
#     rowsum = np.array(adj_mat.sum(axis=1))
#     d_inv = np.power(rowsum, -0.5).flatten()
#     d_inv[np.isinf(d_inv)] = 0.
#     d_mat = sp.diags(d_inv)
#
#     norm_adj = d_mat.dot(adj_mat)
#     norm_adj = norm_adj.dot(d_mat)
#     norm_adj = norm_adj.tocsr()
#
#     g = dgl.from_scipy(norm_adj, 'w', idtype=torch.int64)
#     g.edata['w'] = g.edata['w'].float()
#     # print("saving DGL graph to binary file...")
#     # dgl.save_graphs(graph_file, [g])
#     return g, item_edges_of_user
#
#
# def build_sim_graph(dataset, k, phase="train"):
#     import dgl
#     # graph_file = dataset.config['data_path']+f"/sim_graph_g{k}_{phase}.bin"
#     # try:
#     #     g = dgl.load_graphs(graph_file, [0])
#     #     print("loading isim graph from DGL binary file...")
#     #     return g[0][0]
#     # except:
#     print("building isim graph...")
#     row = []
#     col = []
#     inter_feat = dataset.inter_feat
#     for line in range(len(dataset.inter_feat)):
#         uid = inter_feat[dataset.uid_field][line].item()
#         item_seq = inter_feat[dataset.item_id_list_field][line].tolist()
#         seq_len = inter_feat[dataset.item_list_length_field][line].item()
#         item_seq = item_seq[:seq_len]
#         col.extend(item_seq)
#         row.extend([uid] * seq_len)
#     row = np.array(row)
#     col = np.array(col)
#     # n_users, n_items
#     cf_graph = csr_matrix(([1] * len(row), (row, col)), shape=(
#         dataset.user_num + 1, dataset.item_num + 1), dtype=np.float32)
#     similarity = cosine_similarity(cf_graph.transpose())
#     # filter topk connections
#     sim_items_slices = []
#     sim_weights_slices = []
#     i = 0
#     while i < similarity.shape[0]:
#         similarity = similarity[i:, :]
#         sim = similarity[:256, :]
#         sim_items = np.argpartition(sim, -(k + 1), axis=1)[:, -(k + 1):]
#         sim_weights = np.take_along_axis(sim, sim_items, axis=1)
#         sim_items_slices.append(sim_items)
#         sim_weights_slices.append(sim_weights)
#         i = i + 256
#     sim = similarity[256:, :]
#     sim_items = np.argpartition(sim, -(k + 1), axis=1)[:, -(k + 1):]
#     sim_weights = np.take_along_axis(sim, sim_items, axis=1)
#     sim_items_slices.append(sim_items)
#     sim_weights_slices.append(sim_weights)
#
#     sim_items = np.concatenate(sim_items_slices, axis=0)
#     sim_weights = np.concatenate(sim_weights_slices, axis=0)
#     row = []
#     col = []
#     for i in range(len(sim_items)):
#         row.extend([i] * len(sim_items[i]))
#         col.extend(sim_items[i])
#
#     sim_weights += 1e-8
#     values = sim_weights / sim_weights.sum(axis=1, keepdims=True)
#     values = np.where(np.isnan(values), np.zeros_like(values), values)
#     # np.is
#
#     values = np.nan_to_num(values).flatten()
#     adj_mat = csr_matrix((values, (row, col)), shape=(
#         dataset.item_num + 1, dataset.item_num + 1))
#     g = dgl.from_scipy(adj_mat, 'w')
#     g.edata['w'] = g.edata['w'].float()
#     # print("saving isim graph to binary file...")
#     # dgl.save_graphs(graph_file, [g])
#     return g
#
#
# def sequential_augmentation(dataset):
#     print("building sr aug...")
#     max_item_list_len = dataset.config['MAX_ITEM_LIST_LENGTH']
#     old_data = dataset.inter_feat
#     new_data = {dataset.uid_field: [],
#                 dataset.iid_field: [],
#                 dataset.item_list_length_field: [],
#                 dataset.item_id_list_field: [], }
#     for i in range(len(old_data)):
#         seq = old_data[dataset.item_id_list_field][i]
#         uid = old_data[dataset.uid_field][i].item()
#         seq_len = old_data[dataset.item_list_length_field][i]
#         # print(f'uid: {uid}, seq_len: {seq_len}')
#         new_data[dataset.uid_field].append(uid)
#         new_data[dataset.iid_field].append(old_data[dataset.iid_field][i].item())
#         new_data[dataset.item_list_length_field].append(seq_len)
#         new_data[dataset.item_id_list_field].append(seq)
#         seq = seq[:seq_len]
#         for end_point in range(1, seq_len):
#             new_seq = seq[:end_point]
#             new_truth = seq[end_point].item()
#             new_seq_len = len(new_seq)
#             new_seq = torch.cat((new_seq, torch.zeros(max_item_list_len - new_seq_len, dtype=torch.long)), dim=0)
#
#             new_data[dataset.uid_field].append(uid)
#             new_data[dataset.iid_field].append(new_truth)
#             new_data[dataset.item_list_length_field].append(new_seq_len)
#             new_data[dataset.item_id_list_field].append(new_seq)
#
#     new_data[dataset.item_id_list_field] = torch.stack(new_data[dataset.item_id_list_field], dim=0)
#     new_data[dataset.item_list_length_field] = torch.tensor(new_data[dataset.item_list_length_field], dtype=torch.long)
#     new_data[dataset.uid_field] = torch.tensor(new_data[dataset.uid_field], dtype=torch.long)
#     new_data[dataset.iid_field] = torch.tensor(new_data[dataset.iid_field], dtype=torch.long)
#     dataset.inter_feat = (Interaction(new_data))
#

