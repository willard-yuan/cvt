function [frames, descrs, im] = getFeatures(im, varargin)
% GETFEATURES  Extract feature frames (keypoints) and descriptors
%   [FRAMES, DESCRS] = GETFEATURES(IM) computes the SIFT features
%   from image IM.
%
%   Options:
%
%   AffineAdaptation:: false
%     Set to TRUE to turn on affine adaptation.
%
%   Orientation:: true
%     Set to FALSE to turn off the detection of the feature
%     orientation.
%
%   Method:: Hessian
%     Set to DoG to use the approximated Laplacian operator score.
%
%   MaxHeight:: +inf
%     Rescale the image to have the specified maximum height.
%     Use [~, ~, IM] = GETFEATURES(...) to obtain the rescaled image.

% Author: Andrea Vedaldi

opts.method = 'dog' ;
opts.affineAdaptation = false ;
opts.orientation = true ;
opts.peakThreshold = 28 / 256^2 ;
opts.maxHeight = +inf ;
opts = vl_argparse(opts, varargin) ;

if size(im,3) > 1, im = rgb2gray(im) ; end
im = im2single(im) ;

if size(im,1) > opts.maxHeight
  im = imresize(im, [opts.maxHeight, NaN]) ;
end

[frames, descrs] = vl_covdet(im, 'DoubleImage', true, 'EstimateAffineShape', true, 'PatchRelativeExtent', 10, 'PatchRelativeSmoothing', 1, 'Method', 'DoG', 'verbose') ;

% [frames, descrs] = vl_covdet(im, ...
%                              'EstimateAffineShape', opts.affineAdaptation, ...
%                              'EstimateOrientation', opts.orientation, ...
%                              'DoubleImage', false, ...
%                              'Method', opts.method, ...
%                              'PeakThreshold', opts.peakThreshold, ...
%                              'Verbose') ;
frames = single(frames) ;
