"""
This file contains functions for the interaction model, which computes overall
opportunities for interaction based on time-series state data.

Copyright (C) 2020  Andrew Sonta, Rishee K. Jain

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

@author: Andrew Sonta
"""

import numpy as np
import pandas as pd
from src import TIME_STEP

time_step = TIME_STEP

def fill_opp_vec(v):
    vopp = np.zeros(v.shape)
    vmin = np.min(v)
    vmax = np.max(v)
    if vmin == vmax: return vopp # edge case with no activity
#     start = 0
#     end = v.shape[0]
#     if v[0] > vmin: 
    start = np.where(v==vmin)[0][0] # something left on from previous day
#     if v[-1] > vmin: 
    end = np.where((v==vmin) | (v<v[-1]))[0][-1] # something left on at night
    for i in range(start, end):
        if (v[i] != vmin) and (v[i] != vmax):
            vopp[i] = 1
    try:
        #daystart = np.where(v[start:end]>vmin)[0][0]+start # start of workday occurs between start and end
        daystart = np.where(vopp>0)[0][0] # start of day occurs after first middle state
    except:
        daystart = 0
    try:
        #dayend = np.where(v[start:end]>vmin)[0][-1]+start # end of workday occurs between start and end
        dayend = np.where(vopp>0)[0][-1] # end of day occurs before last middle state
    except:
        dayend = 0
    for j in range(daystart, dayend):
        if (v[j] != vmax):
            vopp[j] = 1
    return vopp

def interaction_matrix(data):
    '''
    Requires the input parameter "data" which is a pandas dataframe
    structured as follows: index is time, columns are occupants, rows
    are states
    '''
    
    occupants = range(data.values.shape[1])
    timesteps = int(1440/time_step)
    overlaps = np.zeros((len(occupants), len(occupants), timesteps))
    similarities = np.zeros((len(occupants), len(occupants)))
    for day in range(int(len(data)/timesteps)):
        opportunity_vecs = []
        vectordata = data.values[timesteps*day:timesteps*(day+1),:]
        for occupant in occupants:
            vec = fill_opp_vec(vectordata[:,occupant])
            opportunity_vecs.append(vec)
            
#         # Old Version w/o using Jaccard Similarity
#         i = 0
#         for vec1 in opportunity_vecs:
#             j = 0
#             for vec2 in opportunity_vecs:
#                 overlap = 0
#                 if vec1 is not vec2:
#                     for k in range(len(vec1)):
#                         if vec1[k] == 1 and vec2[k] == 1:
#                             overlaps[i,j,k] += 1                
#                 j += 1
#             i += 1

        # New version using Jaccard Similarity
        i=0
        for vec1 in opportunity_vecs:
            j=0
            for vec2 in opportunity_vecs:
                intersection = 0
                similarity = 0
                if vec1 is not vec2:
                    for k in range(len(vec1)):
                        if vec1[k] == 1 and vec2[k] == 1:
                            intersection += 1.0
                    if sum(vec1) + sum(vec2) - intersection > 0:
                        similarity = intersection/(sum(vec1) + sum(vec2) - intersection)
                    similarities[i,j] += similarity
                    for k in range(len(vec1)):
                        if vec1[k] == 1 and vec2[k] == 1:
                            overlaps[i,j,k] += 1
                            #overlaps[i,j,k] += 1.0*similarity
                            #overlaps[i,j,k] = 1.0/(sum(vec1)+sum(vec2))
                j += 1
            i += 1
                        
    return overlaps, similarities