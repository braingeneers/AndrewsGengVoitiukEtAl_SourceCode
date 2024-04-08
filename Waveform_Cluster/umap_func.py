from base import *
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

def plot_umap(full_data, figure_name, res=None, save_to=None):
    if res is None:
        res = RESOLUTION
    reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
                        random_state=RAND_STATE)
    mapper = reducer.fit(full_data)
    embedding = reducer.transform(full_data)

    umap_df = pd.DataFrame(embedding, columns=('x', 'y'))
    umap_df['waveform'] = list(full_data)

    G = nx.from_scipy_sparse_matrix(mapper.graph_)
    clustering = cylouvain.best_partition(G, resolution=res)
    clustering_solution = list(clustering.values())
    umap_df['color'] = clustering_solution
    
    cluster_colors = [CUSTOM_PAL_SORT_3[i] for i in clustering_solution]

    f, arr = plt.subplots(1, figsize=[7, 4.5], tight_layout={'pad': 0})
    f.tight_layout()
    arr.scatter(umap_df['x'].tolist(), umap_df['y'].tolist(),
                marker='o', c=cluster_colors, s=32, edgecolor='w',
                linewidth=0.5)
    arr.spines['top'].set_visible(False)
    arr.spines['bottom'].set_visible(False)
    arr.spines['left'].set_visible(False)
    arr.spines['right'].set_visible(False)
    arr.set_xticks([])
    arr.set_yticks([])
    # arr.set_xlim(-4, 12)
    # arr.set_ylim(0, 12)

    arr.arrow(-3, 0.8, 0, 1.5, width=0.05, shape="full", ec="none", fc="black")
    arr.arrow(-3, 0.8, 1.2, 0, width=0.05, shape="full", ec="none", fc="black")

    arr.text(-3, 0.3, "UMAP 1", va="center")
    arr.text(-3.5, 1.0, "UMAP 2", rotation=90, ha="left", va="bottom")

    N_CLUST = len(set(clustering_solution))

    if save_to is not None:
        plt.savefig(f"{os.path.join(save_to, figure_name)}_umap.png", dpi=300)
        plt.savefig(f"{os.path.join(save_to, figure_name)}_umap.pdf", dpi=300)
    else:
        plt.savefig(f"figure/{figure_name}_umap.png", dpi=300)
        plt.savefig(f"figure/{figure_name}_umap.pdf", dpi=300)
    return clustering_solution, umap_df


# Defines a nice function that plots all the waveforms in long column.
def plot_group(label_ix, labels, groups_df, colors,
               color_clusters=[], color=None, 
               loc = True,
               mean_only=False, detailed=False, arr=None):
    group_ixs = [i for i,x in enumerate(labels) if x == label_ix-1]
    group_waveforms = groups_df.iloc[group_ixs]['waveform'].tolist()
    if loc:
        group_waveforms = [x[:50] for x in group_waveforms]
    
    if arr is None:
        f, arr = plt.subplots(figsize=(3.0*0.65, 1.8*0.65))
    
    # f.set_figheight(1.8*0.65)
    # f.set_figwidth(3.0*0.65)
    if not mean_only:
        for i,_ in enumerate(group_waveforms):
            if label_ix in color_clusters:
                arr.plot(group_waveforms[i],c=colors[color],alpha=0.3,linewidth=1.5)
            else:
                arr.plot(group_waveforms[i],c=colors[label_ix-1],alpha=0.3,linewidth=1.5)
    
    if not mean_only:
        arr.plot(np.mean(group_waveforms,axis=0),c='k',linestyle='-')
    else:
        if label_ix in color_clusters:
            arr.plot(np.mean(group_waveforms,axis=0),c=colors[color],linestyle='-')
        else:
            arr.plot(np.mean(group_waveforms,axis=0),c=colors[label_ix-1],linestyle='-')

    arr.spines['right'].set_visible(False)
    arr.spines['top'].set_visible(False)

    if detailed:
        
        avg_peak = np.mean([np.argmax(x) for x in group_waveforms[14:]])
        arr.axvline(avg_peak,color='k',zorder=0)
        
        arr.set_ylim([-1.3,1.3])
        arr.set_yticks([])
        # arr.set_xticks([0,7,14,21,28,35,42,48])
        arr.tick_params(axis='both', which='major', labelsize=12)
        # arr.set_xticklabels([0,'',0.5,'',1.0,'',1.5,''])
        arr.spines['left'].set_visible(False)
        arr.grid(False)
        arr.set_xlim([0,48])

    if not detailed:
        arr.set(xticks=[],yticks=[])

        if not mean_only:
            x,y = 2.1,0.7
            ellipse = mpl.patches.Ellipse((x,y), width=9.0, height=0.72, facecolor='w',
                                 edgecolor='k',linewidth=1.5)
            label = arr.annotate(str(label_ix), xy=(x-0.25, y-0.15),fontsize=12, color = 'k', ha="center")
            arr.add_patch(ellipse)

            # # if i != -1:
            # x, y = 2.3,-0.7
            x, y = 23,-0.7 # 17
            n_waveforms = arr.text(x, y, 
                                    'n = '+str(len(group_waveforms))+
                                    ' ('+str(round(len(group_waveforms)/len(groups_df)*100,2))+'%)'
                                    , fontsize=10)

    return arr