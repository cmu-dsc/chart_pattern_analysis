# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:10:58 2019

@author: timcr
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 19:11:52 2019

@author: timcr
"""

from scipy.spatial.distance import pdist
import numpy as np
import time

def between_two_parallel_lines(line_top, line_bottom, tail_top, tail_bottom, pts):
    '''
    line: np.array shape (1,2)
        If tail is one point on the line, and there's some other point on the
        line that is the head, then line is the vector b - a
    tail_top: np.array shape (1,2)
        The tail point of a line (direction of the line is away from the tail)
        that is above the other of the two parallel lines
    tail_bottom: np.array shape (1,2)
        The tail point of a line (direction of the line is away from the tail)
        that is below the other of the two parallel lines
    pts: np.array shape (n, 2)
        The list of points that you want to test
        
    Return: bool
        True if all points are between the two parallel lines. False o/w
        
    Purpose: compute if all points in pts are between two given parallel lines
        in a vectorized way. It does this by making sure all points are to 
        the right of the top line and to the left of the bottom line. It does
        this with 2D cross-products (determinants)
    '''
    
    return np.all(np.sign(line_top[0] * (pts[:,1] - tail_top[1]) - 
                          line_top[1] * (pts[:,0] - tail_top[0])) \
                            <= 0 ) and \
            np.all(np.sign(line_bottom[0] * (pts[:,1] - tail_bottom[1]) - 
                          line_bottom[1] * (pts[:,0] - tail_bottom[0])) \
                            >= 0)


def find_channels(maxes, mins, tol, poke_out_tol):
    '''
    N: int
        The number of relative maxima in a chart
    maxes: np.array shape (N,2)
        Each row is the coordinate of a local maximum in a chart where the x
        axis (0th column) time and the y axis (1th column) is price. The order
        should be in increasing x value as you go from top to bottom in this
        maxes array.
    M: int
        The number of relative minima in a chart
    mins: np.array shape (M,2)
        Each row is the coordinate of a local minimum in a chart where the x
        axis (0th column) time and the y axis (1th column) is price. The order
        should be in increasing x value as you go from top to bottom in this
        mins array.
    max_vecs: np.array shape (N(N - 1)/2, 2)
        Each row is a vector with tail at a point in maxes and a head a 
        point in maxes that is further in the x direction (time). Thus each 
        vector in max_vecs points forward in time. Note that the first point in
        maxes will be the tail to N - 1 vectors because there are N - 1 points
        in maxes ahead of the first point in time. It is assumed that maxes is 
        ordered on the 0th column (time) from lowest to highest if going from
        top to bottom in maxes. Thus, the first N - 1 entries of max_vecs are 
        with maxes[0] as the tail, and the following N - 2 entries of max_vecs
        are with maxes[1] as the tail, etc.
    min_vecs: np.array shape (M(M - 1)/2, 2)
        Each row is a vector with tail at a point in mins and a head a 
        point in maxes that is further in the x direction (time). Thus each 
        vector in max_vecs points forward in time. Note that the first point in
        mins will be the tail to M - 1 vectors because there are M - 1 points
        in mins ahead of the first point in time. It is assumed that maxes is 
        ordered on the 0th column (time) from lowest to highest if going from
        top to bottom in maxes. Thus, the first M - 1 entries of max_vecs are 
        with mins[0] as the tail, and the following M - 2 entries of max_vecs
        are with mins[1] as the tail, etc.
    max_pt_idx: np.array shape (N(N - 1)/2,)
        The ith element of max_pt_idx is the index in maxes of the tail of the
        ith vector in max_vecs.
    min_pt_idx: np.array shape (M(M - 1)/2,)
        The jth element of min_pt_idx is the index in mins of the tail of the
        jth vector in min_vecs
    positive_max_vec_idx: np.array shape (N(N - 1)/2,)
        Each element is a bool. If True at the ith element, then the ith vector
        in max_vecs is directed up and to the right (hence "positive"). Only
        "positive" vectors will be in a rising channel.
    positive_min_vec_idx: np.array shape (M(M - 1)/2,)
        Each element is a bool. If True at the jth element, then the jth vector
        in min_vecs is directed up and to the right (hence "positive"). Only
        "positive" vectors will be in a rising channel.
    n: int
        n = number of True's in positive_max_vec_idx
    positive_max_vecs: np.array shape (n,2)
        Each row is a vector in max_vecs that is directed up and to the right
        (hence "positive"). Only "positive" vectors will be in a rising channel.
    m: int
        m = number of True's in positive_min_vec_idx
    positive_min_vecs: np.array shape (m,2)
        Each row is a vector in min_vecs that is directed up and to the right
        (hence "positive"). Only "positive" vectors will be in a rising channel.
    crit_vecs: np.array shape (n + m,2)
        positive_max_vecs stacked vertically on top of positive_min_vecs.
    pdists_arr: np.array shape ((n**2 + n(m - 1))/2,)
        See pdist documentation. The idea is to get the cosine distance between
        every pair of vectors in positive_max_vecs and positive_min_vecs. This 
        function will give the pairwise distance for every pair in crit_vecs,
        but we will extract out the pairs of vectors between positive_max_vecs 
        and positive_min_vecs only later. While this function computes unnecessary
        pairwise distances (those between positive_max_vecs and positive_max_vecs
        or between positive_min_vecs and positive_min_vecs), pdist is much faster
        than other methods (even those which only calculate the desired output).
        Note that the first n - 1 + m entries in pdists_arr are distances to
        the 0th vector in crit_vecs, the next n - 2 + m entries in pdists_arr
        are distances to the 1th vector in crit_vecs, etc.
    pdists: np.array shape (n,m)
        The (i,j)th entry in pdists is the cosine distance between the ith vector
        in positive_max_vecs and the jth vector in positive_min_vecs. See 
        documentation of pdist for info about cosine distance. These were extracted
        out of pdists_arr using the following methodology for indexing: The 
        first n - 1 entries are pairwise distance from the 0th vector in 
        positive_max_vecs to the other vectors in positive_max_vecs and so
        can be ignored. We want the pairwise distances from the 0th vector in 
        positive_max_vecs to the m vectors in positive_min_vecs so that will
        be the next m entries after the first n - 1 entries. To get the pairwise
        distances to the 1th vector in positive_max_vecs to the m vectors in
        positive_min_vecs, start at n - 1 + m + n - 2 and go until n - 1 + m +
        n - 2 + m. We see a pattern of n (i + 1) + m (i) - (partial sum of
        k = 1 through i + 1). The partial sum evaluates to (i + 1)(i + 2)/2
        so the entire expresses simplifies to n * (i + 1) - ((i ** 2) + 3 * 
        i + 2) // 2 + m * i and the right side of the slice just adds an m to this.
    tol: float
        tolerance. The cosine distance between two vectors will be close to 0
        if two vectors are near parallel.
    pdists_below_tol: np.array shape (2, # of entries in pdists below tol)
        Each column in pdists_below_tol is the (row, column) index of an entry
        in pdists that is below tol. Thus the 0th row of pdists_below_tol
        represents indices of vectors in positive_max_vecs and the 1th row of
        pdists_below_tol represents indices of vectors in positive_min_vecs
        where each pair of vectors pdists_below_tol[0,i] pdists_below_tol[1,i]
        are parallel (within tolerance tol).
    max_vecs_below_tol: np.array shape (number of vecs in positive_max_vecs 
        that had a pair in positive_min_vecs that had a pairwise cosine distance
        less than tol,2)
        Each row in max_vecs_below_tol is a positive vector connecting two points
        in maxes pointing forwards in time that has a cosine distance below tol to
        the corresponding vector in min_vecs_below_tol. (i.e. the 0th vector in
        max_vecs_below_tol has a cosine distance less than tol with the 0th vector
        in min_vecs_below_tol).
    min_vecs_below_tol: np.array shape (len(max_vecs_below_tol),)
        Each row in inx_vecs_below_tol is a positive vector connecting two points
        in mins pointing forwards in time that has a cosine distance below tol to
        the corresponding vector in max_vecs_below_tol. (i.e. the 0th vector in
        max_vecs_below_tol has a cosine distance less than tol with the 0th vector
        in min_vecs_below_tol).
    tails_of_maxes: np.array shape (len(max_vecs_below_tol),2)
        Each row in tails_of_maxes is a point in maxes which is the tail of a 
        vector in max_vecs_below_tol.
    tails_of_mins: np.array shape (len(max_vecs_below_tol),2)
        Each row in tails_of_mins is a point in mins which is the tail of a 
        vector in min_vecs_below_tol.
    signs: np.array shape (len(max_vecs_below_tol),)
        An entry is -1 if a the tail of the min vector is to the right of the line
        formed by the max vector that this min vector is parallel to. If it is
        to the right, then the min vector is below the max vector which is
        required for a rising channel.
    mins_below_maxes_idx: tuple
        The 0th element of mins_below_maxes_idx is a np.array which has shape
        (number of -1's in signs,). mins_below_maxes_idx is the indices of -1's
        in signs.
    max_vecs_below_tol_with_min_below_max: np.array shape
        (len(mins_below_maxes_idx[0]),2)
        Each row is a vector in max_vecs_below_tol that also corresponds to a
        vector in min_vecs_below_tol that is underneath this vector in 
        max_vecs_below_tol. (i.e. the 0th vector in
        max_vecs_below_tol_with_min_below_max is parallel to and above the 0th
        vector in min_vecs_below_tol_with_min_below_max)
    min_vecs_below_tol_with_min_below_max: np.array shape 
        (len(max_vecs_below_tol_with_min_below_max),2)
        Each row is a vector in min_vecs_below_tol that also corresponds to a
        vector in max_vecs_below_tol that is above this vector in 
        min_vecs_below_tol. (i.e. the 0th vector in
        max_vecs_below_tol_with_min_below_max is parallel to and above the 0th
        vector in min_vecs_below_tol_with_min_below_max)
    tails_of_maxes_with_min_below_max: np.array shape
        (len(max_vecs_below_tol_with_min_below_max),2)
        These are the points in maxes that are the tails of vectors in 
        max_vecs_below_tol_with_min_below_max
    tails_of_mins_with_min_below_max: np.array shape
        (len(max_vecs_below_tol_with_min_below_max),2)
        These are the points in mins that are the tails of vectors in 
        min_vecs_below_tol_with_min_below_max
    heads_of_maxes_with_min_below_max: np.array shape
        (len(max_vecs_below_tol_with_min_below_max),2)
        These are the points in maxes that are the heads of vectors in 
        max_vecs_below_tol_with_min_below_max
    heads_of_mins_with_min_below_max: np.array shape
        (len(max_vecs_below_tol_with_min_below_max),2)
        These are the points in mins that are the heads of vectors in 
        min_vecs_below_tol_with_min_below_max
    channel_idx: tuple
        See the documentation for np.where. The expression evaluates to True
        if we have the order in x: tail of max vector, tail of min vector,
        head of max vector, head of tail vector. This is a required ordering 
        for a rising channel.
    channel_max_tails: np.array shape (number of rising channels,2)
        Each row is a point in maxes that is the tail of a max vector that
        is part of a rising channel.
    channel_min_tails: np.array shape (number of rising channels,2)
        Each row is a point in mins that is the tail of a min vector that
        is part of a rising channel.
    channel_max_heads: np.array shape (number of rising channels,2)
        Each row is a point in maxes that is the heads of a max vector that
        is part of a rising channel.
    channel_min_heads: np.array shape (number of rising channels,2)
        Each row is a point in mins that is the heads of a min vector that
        is part of a rising channel.
        
    Return: channel_max_tails, channel_min_tails, channel_max_heads,
        channel_min_heads
        These are the points that can be used to define a rising channel.
        
    Purpose: Given an array of local maxima and minima in a (time,price) chart,
        find any rising channels. A decent option is to smooth the data with 
        volume weighted average prices (VWAP) and to use the argrelextrema
        function from scipy.signal for getting local maxima and minima.
    '''
    # Instantiate relative max and min vectors for testing purposes. The real
    # vectors would be gotten probably by scipy's argrelextrema function.
    #maxes = np.array([[1,3],[4,1],[8,9],[10,11]])
    #mins = np.array([[0,2],[2,7],[5,6]])
    
    N = len(maxes)
    M = len(mins)
    
    crit_pts = np.vstack([maxes, mins])
    #Scale from 0 to 1
    sup = np.max(crit_pts, axis=0)
    inf = np.min(crit_pts, axis=0)
    maxes = (maxes - inf) / (sup - inf)
    mins = (mins - inf) / (sup - inf)
    crit_pts = (crit_pts - inf) / (sup - inf)
    
    # Construct all vectors from a given extremum to all vectors after it
    max_vecs = np.vstack([maxes[i + 1:] - maxes[i] for i in range(N)])
    min_vecs = np.vstack([mins[j + 1:] - mins[j] for j in range(M)])

    # Keep track of point indices
    max_pt_idx = np.hstack([np.repeat(i, N - i - 1) for i in range(N)])
    min_pt_idx = np.hstack([np.repeat(j, M - j - 1) for j in range(M)])
    
    # We only need vectors directed up and to the right
    positive_max_vec_idx = np.all(max_vecs >= 0, axis=1)
    positive_min_vec_idx = np.all(min_vecs >= 0, axis=1)
    
    positive_max_vecs = max_vecs[positive_max_vec_idx]
    positive_min_vecs = min_vecs[positive_min_vec_idx]
    
    # Maintain the point indices so we can keep track of the points in maxes
    # and mins
    max_pt_idx = max_pt_idx[positive_max_vec_idx]
    min_pt_idx = min_pt_idx[positive_min_vec_idx]

    n = len(positive_max_vecs)
    m = len(positive_min_vecs)
    
    if n == 0 or m == 0:
        return [],[],[],[],[],[],[],[]
    
    # pdist is the fastest way to compute pairwise cosine distances
    crit_vecs = np.vstack([positive_max_vecs, positive_min_vecs ])
    pdists_arr = pdist(crit_vecs, 'cosine')
    
    # Extract just the distances from vectors in maxes to vectors in mins
    pdists = np.vstack([pdists_arr[n * (i + 1) - ((i ** 2) + 3 * i + 2) // 2 + m * i : n * (i + 1) - ((i ** 2) + 3 * i + 2) // 2 + m * (i + 1)] for i in range(n)])
    
    pdists_below_tol = np.vstack(np.where(pdists < tol))
    
    if len(pdists_below_tol) == 0:
        return [],[],[],[],[],[],[],[]
    
    # Extract the pairs of vectors that parallel
    max_vecs_below_tol = positive_max_vecs[pdists_below_tol[0]]
    min_vecs_below_tol = positive_min_vecs [pdists_below_tol[1]]
    
    # Get the tails of these vectors
    tails_of_maxes = maxes[max_pt_idx[pdists_below_tol[0]]]
    tails_of_mins = mins[min_pt_idx[pdists_below_tol[1]]]
    
    # Make sure that the min vector is below the max vector
    signs = np.sign(max_vecs_below_tol[:,0] * (tails_of_mins[:,1] - tails_of_maxes[:,1]) - max_vecs_below_tol[:,1] * (tails_of_mins[:,0] - tails_of_maxes[:,0]))
    mins_below_maxes_idx = np.where(signs == -1)
    
    # Get the pairs of vectors that are parallel, positive, and the max vector
    # is above the min vector.
    max_vecs_below_tol_with_min_below_max = max_vecs_below_tol[mins_below_maxes_idx]
    min_vecs_below_tol_with_min_below_max = min_vecs_below_tol[mins_below_maxes_idx]
    tails_of_maxes_with_min_below_max = tails_of_maxes[mins_below_maxes_idx]
    tails_of_mins_with_min_below_max = tails_of_mins[mins_below_maxes_idx]
    
    if len(max_vecs_below_tol_with_min_below_max) == 0 or \
        len(min_vecs_below_tol_with_min_below_max) == 0 or \
        len(tails_of_maxes_with_min_below_max) == 0 or \
        len(tails_of_mins_with_min_below_max) == 0:
        return [],[],[],[],[],[],[],[]
    
    heads_of_maxes_with_min_below_max = tails_of_maxes_with_min_below_max + max_vecs_below_tol_with_min_below_max
    heads_of_mins_with_min_below_max = tails_of_mins_with_min_below_max + min_vecs_below_tol_with_min_below_max
    
    # This is another check. Check to see if the order of x coordinate is
    # tail of the max vec, tail of the min vec, head of the max vec, head of
    # the min vec. This is part of the definition of a rising channel.
    valid_rel_pos_idx = np.where((tails_of_maxes_with_min_below_max[:,0] < tails_of_mins_with_min_below_max[:,0]) \
                    & (tails_of_mins_with_min_below_max[:,0] < heads_of_maxes_with_min_below_max[:,0]) \
                    & (heads_of_maxes_with_min_below_max[:,0] < heads_of_mins_with_min_below_max[:,0]))
    
    valid_rel_pos_max_tails = tails_of_maxes_with_min_below_max[valid_rel_pos_idx]
    valid_rel_pos_min_tails = tails_of_mins_with_min_below_max[valid_rel_pos_idx]
    valid_rel_pos_max_heads = heads_of_maxes_with_min_below_max[valid_rel_pos_idx]
    valid_rel_pos_min_heads = heads_of_mins_with_min_below_max[valid_rel_pos_idx]
    valid_rel_pos_max_vecs = max_vecs_below_tol_with_min_below_max[valid_rel_pos_idx]
    valid_rel_pos_min_vecs = min_vecs_below_tol_with_min_below_max[valid_rel_pos_idx]
    
    if len(valid_rel_pos_max_tails) == 0 or \
        len(valid_rel_pos_min_tails) == 0 or \
        len(valid_rel_pos_max_heads) == 0 or \
        len(valid_rel_pos_min_heads) == 0 or \
        len(valid_rel_pos_max_vecs) == 0 or \
        len(valid_rel_pos_min_vecs) == 0:
        return [],[],[],[],[],[],[],[]
    
    # Shift vectors with valid relative positions of heads and tails of the
    # maxes and mins by their normal (unit) vector * some tolerance scalar and
    # also scale by something to make the tolerance consistent for any
    # chart (have one tolerance that works with various price ranges of different
    # charts)
    shift_vecs = np.zeros(valid_rel_pos_max_vecs.shape)
    shift_vecs[:,0], shift_vecs[:,1] = -valid_rel_pos_max_vecs[:,1], valid_rel_pos_max_vecs[:,0]
    
    shift_vecs = (poke_out_tol) * shift_vecs / np.linalg.norm(shift_vecs, axis=1, keepdims=True)
    
    shifted_max_tails = valid_rel_pos_max_tails + shift_vecs
    shifted_min_tails = valid_rel_pos_min_tails - shift_vecs
    shifted_max_heads = valid_rel_pos_max_heads + shift_vecs
    shifted_min_heads = valid_rel_pos_min_heads - shift_vecs

    #Sort on x coordinate 
    sorted_crit_pts = crit_pts[crit_pts[:,0].argsort()]
    # Find first index + 1 with x coordinate greater than the tail of the max vec.
    start_idx = np.searchsorted(sorted_crit_pts[:,0], valid_rel_pos_max_tails[:,0], 'right')
    # Find last index with x coordinate less than the head of the min vec
    end_idx = np.searchsorted(sorted_crit_pts[:,0], valid_rel_pos_min_heads[:,0])
    
    channel_idx = [between_two_parallel_lines(valid_rel_pos_max_vecs[i], valid_rel_pos_min_vecs[i],
                                shifted_max_tails[i], shifted_min_tails[i],
                                sorted_crit_pts[np.arange(start_idx[i], end_idx[i])])
                    for i in range(len(start_idx))]
    
    # Return the tails and heads of channel vectors.
    channel_max_tails = valid_rel_pos_max_tails[channel_idx]
    channel_min_tails = valid_rel_pos_min_tails[channel_idx]
    channel_max_heads = valid_rel_pos_max_heads[channel_idx]
    channel_min_heads = valid_rel_pos_min_heads[channel_idx]
    #shift_vecs = shift_vecs[channel_idx]
    shifted_max_tails = shifted_max_tails[channel_idx]
    shifted_min_tails = shifted_min_tails[channel_idx]
    shifted_max_heads = shifted_max_heads[channel_idx]
    shifted_min_heads = shifted_min_heads[channel_idx]
    
    if len(channel_max_tails) == 0 or \
        len(channel_min_tails) == 0 or \
        len(channel_max_heads) == 0 or \
        len(channel_min_heads) == 0 or \
        len(shifted_max_tails) == 0 or \
        len(shifted_min_tails) == 0 or \
        len(shifted_max_heads) == 0 or \
        len(shifted_min_heads) == 0:
        return [],[],[],[],[],[],[],[]
    
    #Rescale back to original scale
    channel_max_tails = channel_max_tails * (sup - inf) + inf
    channel_min_tails = channel_min_tails * (sup - inf) + inf
    channel_max_heads = channel_max_heads * (sup - inf) + inf
    channel_min_heads = channel_min_heads * (sup - inf) + inf
    shifted_max_tails = shifted_max_tails * (sup - inf) + inf
    shifted_min_tails = shifted_min_tails * (sup - inf) + inf
    shifted_max_heads = shifted_max_heads * (sup - inf) + inf
    shifted_min_heads = shifted_min_heads * (sup - inf) + inf

    return channel_max_tails, channel_min_tails, channel_max_heads, channel_min_heads, shifted_max_tails, shifted_min_tails, shifted_max_heads, shifted_min_heads
    
    
def main(num_maxes, num_mins):
    dim = 2
    maxes = np.random.randint(0,100 + 1, (num_maxes,dim))
    mins = np.random.randint(0,100 + 1, (num_mins,dim))
    all_pts = np.vstack((maxes, mins))
    
    pdist_times = []
    list_times = []
    cosine_sim_times = []
    cosine_times = []
    
    for it in range(1):
        start_time = time.time()
        pdists = pdist(all_pts, 'cosine')
        print('pdists.shape', pdists.shape)
        pdist_times.append(time.time() - start_time)
        
        start_time = time.time()
        max_vecs = np.vstack([maxes[i + 1:] - maxes[i] for i in range(num_maxes)])
        min_vecs = np.vstack([mins[i + 1:] - mins[i] for i in range(num_mins)])
        list_times.append(time.time() - start_time)
        print('max_vecs.shape', max_vecs.shape)
        print('max_vecs.size', max_vecs.size)
        #max_vecs = [max_vec[np.all(max_vec >= 0, axis=1)] for max_vec in max_vecs]
        #min_vecs = [min_vec[np.all(min_vec >= 0, axis=1)] for min_vec in min_vecs]
        #start_time = time.time()
        #cosine_dists = [[1 - cosine(max_vec, min_vec) for min_vec in min_vecs] for max_vec in max_vecs]
        #cosine_times.append(time.time() - start_time)
        #start_time = time.time()
        #cosine_sims = [[cosine_similarity(max_vec, min_vec) for min_vec in min_vecs if len(max_vec) > 0 and len(min_vec) > 0] for max_vec in max_vecs]
        #cosine_sim_times.append(time.time() - start_time)
        #norms_max_vecs = [np.linalg.norm(max_vec) for max_vec in max_vecs]
        #norms_min_vecs = [np.linalg.norm(min_vec) for min_vec in min_vecs]
        #cosine_dists = [[np.dot(min_vecs[j], max_vecs[i].T) - norms_max_vecs[i] * norms_min_vecs[j] for j in range(len(min_vecs))] for i in range(len(max_vecs))]
        
        
    print('time for pdist', sum(pdist_times) / float(len(pdist_times)))
    print('time for lists', sum(list_times) / float(len(list_times)))
if __name__ == '__main__':
    main(1000,1000)
