function [nn, dist2] = findNeighbours(descrs1, descrs2, numNeighbours)
% FINDNEIGHBOURS  Find nearest neighbours
%   NN = FINDNEIGHBOURS(DESCRS1, DESCRS2) finds for each column of
%   DESCRS1 the closest column of DESRS2 (in Eulclidean distance)
%   storing the index to NN. The function uses a KDTree for
%   apporximate but fast matching.
%
%   NN = FINDNEIGHBOURS(DESCRS1, DESCRS2, NUMNEIGHBOURS) returns
%   NUMNEIGHBOUSRS neighbours by increasing distance, storing them
%   as successive rows of NN.
%
%   [NN, DIST2] = FINDNEIGHBOURS(...) returns the corresponding
%   matrix of distances DIST2 as well.

% Authors: Andrea Vedaldi

if nargin <= 2, numNeighbours = 1 ; end
vl_twister('state',0) ;
tree = vl_kdtreebuild(descrs2,'numTrees',2) ;
[nn, dist2] = vl_kdtreequery(tree, descrs2, descrs1, ...
                             'maxNumComparisons', 100, ...
                             'numNeighbors', numNeighbours) ;
