import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import pickle
import rand_cmap
from astropy.time import Time


def hierarchical_clustering(ra, dec):
    X = np.array([ra, dec]).T
    Z = linkage(X, 'complete')

    max_d = 0.0167

    clusters = fcluster(Z, max_d, criterion='distance')

    return clusters


def select_by_bullshit(unique_clusters):
    cluster_ok_bool_inds = []
    for cl in unique_clusters:
        inds = np.where(clusters == cl)
        filts = filters[inds]
        print np.unique(filts, return_counts=True)
        assert False


def separate_by_epochs(unique_clusters):
    for cl in unique_clusters:
        inds = np.where(clusters == cl)
        cluster_stop_dates = stop_date[inds]
        print stop_date[inds]
        assert False


def select_by_date(unique_clusters, days_buffer=100):
    cluster_ok_bool_inds = []
    for cl in unique_clusters:
        inds = np.where(clusters == cl)
        cluster_stop_date = stop_date[inds]
        current_mjd = cluster_stop_date[0]
        cluster_is_ok = 0
        for date in cluster_stop_date:
            if np.abs(date - current_mjd) < days_buffer:
                continue
            else:
                cluster_is_ok = 1
                print 'cluster', cl, 'is ok with', date, '!=', current_mjd
                break
        cluster_ok_bool_inds.append(cluster_is_ok)

    cluster_ok_bool_inds = np.array(cluster_ok_bool_inds, dtype=np.bool)

    unique_clusters = unique_clusters[cluster_ok_bool_inds]

    return unique_clusters


def select_by_num_obs(clusters, min_num_obs=8):
    unique, unique_inds, unique_counts = np.unique(clusters,
                                                   return_counts=True,
                                                   return_index=True)

    inds_where_many_obs = np.where(unique_counts >= 4)

    clusters_with_many_obs = unique[inds_where_many_obs]

    return clusters_with_many_obs


def sort_cluster_coords(unique_clusters, ra, dec):
    new_ims_ra = np.empty(0)
    new_ims_dec = np.empty(0)
    clusters_new = np.empty(0)
    ra_cluster = []
    dec_cluster = []

    for cl in unique_clusters:
        inds = np.where(clusters == cl)
        new_ims_ra = np.concatenate((new_ims_ra, ra[inds]))
        new_ims_dec = np.concatenate((new_ims_dec, dec[inds]))
        clusters_new = np.concatenate((clusters_new, clusters[inds]))
        ra_cluster.append(ra[inds][0])
        dec_cluster.append(dec[inds][0])

    return new_ims_ra, new_ims_dec, clusters_new, \
        ra_cluster, dec_cluster


def plot_ra_dec_scatter(ra, dec, clusters):
    new_cmap = rand_cmap.rand_cmap(len(ra), type='bright',
                                   first_color_black=False,
                                   last_color_black=False,
                                   verbose=False)
    plt.figure(figsize=(10, 8))
    plt.scatter(ra, dec,
                alpha=0.3, c=clusters, cmap=new_cmap)
    plt.ylim(-80, 80)
    plt.xlim(0, 270)


if __name__ == '__main__':

    wfc3_data = pickle.load(open('../../data/wfc3_data.p', 'rb'))

    # data
    ra = wfc3_data['RA']
    dec = wfc3_data['Dec']
    stop_date = wfc3_data['stop_date']
    filters = wfc3_data['filter']

    t = Time(stop_date)
    stop_date = t.mjd

    clusters = hierarchical_clustering(ra, dec)
    unique_clusters = np.unique(clusters)

    unique_clusters = select_by_num_obs(clusters)

    unique_clusters = select_by_date(unique_clusters)


    ra_all_ims, \
        dec_all_ims, \
        clusters_all_ims, \
        ra_cluster, \
        dec_cluster = sort_cluster_coords(unique_clusters, ra, dec)

    plot_ra_dec_scatter(ra, dec, clusters)

    plot_ra_dec_scatter(ra_all_ims, dec_all_ims, clusters_all_ims)

    plt.show()

    with open('clustered_wfc3_coords.txt', 'wb') as fl:
        for n in range(len(ra_cluster)):
            fl.write('%0.4f\t%0.4f\n' % (ra_cluster[n], dec_cluster[n]))
