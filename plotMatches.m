function plotMatches(im1,im2,f1,f2,matches,varargin)
% PLOTMATCHES  Plot matching features between images
%   PLOTMATCHES(IM1, IM2, F1, F2, MATCHES) displays the images IM1 and
%   IM2 overlaying the feature frames F1 and F2 as well as lines
%   connecting them as specified by MATCHES. Each column of MATCHES
%   paris the frame F1(:, MATCHES(1,i)) to the frame F2(:,
%   MATCHES(2,i)).
%
%   Options:
%
%   plotallFrames:: false
%     Set to true in order to plot all the frames, regardless of
%     whether they are matched or not.
%
%   Homography:: []
%     Set to an homography matrix from the first image to the second
%     to display the homography mapping interactively.

% Author: Andrea Vedaldi

opts.plotAllFrames = false ;
opts.homography = [];
opts = vl_argparse(opts, varargin) ;

dh1 = max(size(im2,1)-size(im1,1),0) ;
dh2 = max(size(im1,1)-size(im2,1),0) ;

o = size(im1,2) ;
if size(matches,1) == 1
  i1 = find(matches) ;
  i2 = matches(i1) ;
else
  i1 = matches(1,:) ;
  i2 = matches(2,:) ;
end

hold on ;
f2p = f2 ;
f2p(1,:) = f2p(1,:) + o ;

cla ; set(gca,'ydir', 'reverse') ;
imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
axis image off ;
set(gca,'xlimmode', 'manual') ;
set(gca,'ylimmode', 'manual') ;
if opts.plotAllFrames
  vl_plotframe(f1,'linewidth',2) ;
  vl_plotframe(f2p,'linewidth',2) ;
else
  vl_plotframe(f1(:,i1),'linewidth',2) ;
  vl_plotframe(f2p(:,i2),'linewidth',2) ;
end
line([f1(1,i1);f2p(1,i2)], [f1(2,i1);f2p(2,i2)]) ;
title(sprintf('number of matches: %d', size(matches,2))) ;

if ~isempty(opts.homography)
  s.axes = gca ;
  s.cursor1 = [0;0];
  s.cursor2 = [0;0];
  s.size1 = size(im1) ;
  s.size2 = size(im2) ;
  if verLessThan('matlab', '8.4.0')
    s.point1 = plot(0,0,'g+','MarkerSize', 40, 'EraseMode','xor') ;
    s.point2 = plot(0,0,'r+','MarkerSize', 40, 'EraseMode','xor') ;
  else
    s.point1 = plot(0,0,'g+','MarkerSize', 40) ;
    s.point2 = plot(0,0,'r+','MarkerSize', 40) ;
  end
  s.H = inv(opts.homography) ;
  set(gcf, 'UserData', s)
  set(gcf, 'WindowButtonMotionFcn', @mouseMove) ;
end

function mouseMove(object, eventData)
s = get(object, 'UserData') ;
point = get(s.axes, 'CurrentPoint') ;
if point(1) <= s.size1(2)
  s.cursor1 = point(1, 1:2)' ;
  z = s.H * [s.cursor1;1] ;
  s.cursor2 = z(1:2) / z(3) ;
else
  s.cursor2 = point(1, 1:2)' - [s.size1(2) ; 0] ;
  z = inv(s.H) * [s.cursor2;1] ;
  s.cursor1 = z(1:2) / z(3) ;
end
set(s.point1, 'XData', s.cursor1(1) , 'YData', s.cursor1(2)) ;
set(s.point2, 'XData', s.cursor2(1) + s.size1(2) , 'YData', s.cursor2(2)) ;
if ~ verLessThan('matlab', '8.4.0'), drawnow expose ; end
 
