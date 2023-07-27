import networkx as nx
import os

from networkx.algorithms.tree.operations import join

def calc_graph(config):
    
    f = open(os.path.join(config.data.data_dir,config.data.hierarchy), "r")
    # taxonomy_data = """
    # Root	CCAT	ECAT	GCAT	MCAT
    # CCAT	C11	C12	C13	C14	C15	C16	C17	C18	C21	C22	C23	C24	C31	C32	C33	C34	C41	C42
    # C15	C151	C152
    # C151	C1511
    # C17	C171	C172	C173	C174
    # C18	C181	C182	C183
    # C31	C311	C312	C313
    # C33	C331
    # C41	C411
    # ECAT	E11	E12	E13	E14	E21	E31	E41	E51	E61	E71
    # E12	E121
    # E13	E131	E132
    # E14	E141	E142	E143
    # E21	E211	E212
    # E31	E311	E312	E313
    # E41	E411
    # E51	E511	E512	E513
    # GCAT	G15	GCRIM	GDEF	GDIP	GDIS	GENT	GENV	GFAS	GHEA	GJOB	GMIL	GOBIT	GODD	GPOL	GPRO	GREL	GSCI	GSPO	GTOUR	GVIO	GVOTE	GWEA	GWELF
    # G15	G151	G152	G153	G154	G155	G156	G157	G158	G159
    # MCAT	M11	M12	M13	M14
    # M13	M131	M132
    # M14	M141	M142	M143
    # """.strip().split('\n')
    taxonomy_data = f.read().strip().split('\n')
    
    graph = nx.DiGraph()

    
    for line in taxonomy_data:
        nodes = line.split()
        parent = nodes[0]
        children = nodes[1:]
        for child in children:
            graph.add_edge(parent, child)

    
    
    # adjacency_matrix = nx.adjacency_matrix(graph)

    
    # dense_adjacency_matrix = adjacency_matrix.todense()
    undirected_graph = graph.to_undirected()
    # nx.shortest_path_length(undirected_graph, 'C11', 'C1511')
    return undirected_graph
    
def calc_acm(config, corpus_vocab):
    graph = calc_graph(config) 

    ac_matrix = [[nx.shortest_path_length(graph, corpus_vocab[i], corpus_vocab[j]) 
            if nx.has_path(graph, corpus_vocab[i], corpus_vocab[j]) else float('inf') 
            for j in range(len(corpus_vocab))] for i in range(len(corpus_vocab))]

    # print(acm[54][63], "acm")
    return ac_matrix

    # nx.shortest_path_length(graph, corpus_vocab[i], corpus_vocab[j])




    