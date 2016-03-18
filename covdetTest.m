run E:/MatlabTools/vlfeat-0.9.20-bin/toolbox/vl_setup.m;
clear;
im1 = imread('img1.jpg');
im1 = rgb2gray(im1);
im1 = im2single(im1) ;
[f1,d1] = vl_covdet(im1, 'DoubleImage', true, 'EstimateAffineShape', true, 'PatchRelativeExtent', 10, 'PatchRelativeSmoothing', 1, 'Method', 'DoG', 'verbose') ;

im2 = imread('img2.jpg');
im2 = rgb2gray(im2);
im2 = im2single(im2) ;
[f2,d2] = vl_covdet(im2, 'DoubleImage', true, 'EstimateAffineShape', true, 'PatchRelativeExtent', 10, 'PatchRelativeSmoothing', 1, 'Method', 'DoG', 'verbose') ;

%maxNumComparisons设置得越大精度越高
vl_twister('state',0) ;
tree = vl_kdtreebuild(d2, 'numTrees',8, 'ThresholdMethod', 'mean', 'Distance', 'L2') ;
[nn, dist2] = vl_kdtreequery(tree, d2, d1, ...
                             'maxNumComparisons', 800, ...
                             'numNeighbors', 2) ;
%dist2 = sqrt(dist2); %未开方
 
%ED = EuDist2(d1', d2');
%[sED, iDex] = sort(ED, 2);

%[pD,pI] = pdist2(d2',d1','euclidean','Smallest',2);

% [f,d] = vl_covdet(...
%   im, ...
%   'DoubleImage', true, ...
%   'EstimateAffineShape', true, ...
%   'PatchRelativeExtent', 10, ...
%   'PatchRelativeSmoothing', 1, ...
%   'Method', 'DoG') ; %f:6*n_points,d:128*n_points

% Accept neighbours if their second best match is sufficiently far off
nnThreshold = 0.8 ;
ratio2 = dist2(1,:) ./ dist2(2,:) ;
ok = ratio2 <= nnThreshold^2 ;

% Construct a list of filtered matches
matches_2nn = [find(ok) ; nn(1, ok)] ;

im1 = imread('img1.jpg');
im2 = imread('img2.jpg');

% Display the matches
figure(1) ; clf ;
set(gcf,'name', 'Part I.D: SIFT descriptors - geometric verification') ;
plotMatches(im1,im2,f1,f2,matches_2nn) ;
title('2NN') ;

[inliers, H] = geometricVerification(f1, f2, matches_2nn, 'numRefinementIterations', 8) ;
matches_geo = matches_2nn(:, inliers) ;

% Display the matches
figure(2) ; clf ;
set(gcf,'name', 'Part I.D: SIFT descriptors - geometric verification') ;
plotMatches(im1,im2,f1,f2,matches_geo, 'homography', H) ;
title('Matches filtered by geometric verification') ;