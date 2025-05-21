% pagerank_eigen.m
% Simple PageRank via Eigenvector of the Google Matrix
% Author: Toluwanimi
% Date: May 2025

clc; clear; close all;

%% 1. Define the link structure
% Suppose we have 5 pages. L(i,j) = 1 if page j links to page i.
L = [0 1 1 0 0;   % page1 is pointed to by page2 & page3
     0 0 1 0 0;   % page2 ← page3
     1 0 0 1 1;   % page3 ← page1,4,5
     0 0 1 0 0;   % page4 ← page3
     0 0 1 0 0];  % page5 ← page3

n = size(L,1);

%% now we  build the column-stochastic matrix S
out_degree = sum(L,1);
S = zeros(n);
for j = 1:n
    if out_degree(j) == 0
        % Handle dangling nodes by redistributing uniformly
        S(:,j) = 1/n;
    else
        S(:,j) = L(:,j) / out_degree(j);
    end
end

%%  we get a sample Google matrix G = α S + (1–α)(1/n) * 1·1ᵀ
alpha = 0.85;                  % damping factor
G = alpha * S + (1 - alpha) * (ones(n) / n);

%% here we now compute PageRank vector via eigen decomposition
% We want the eigenvector of G corresponding to eigenvalue = 1
[V, D] = eig(G);
% Extract the eigenvector with eigenvalue closest to 1
[~, idx] = min(abs(diag(D) - 1));
r = V(:, idx);
r = real(r);       % discard any tiny complex residuals
r = r / sum(r);    % normalize to sum to 1


disp('PageRank scores (principal eigenvector):');
for i = 1:n
    fprintf(' Page %d: %.4f\n', i, r(i));
end

% Bar chart
figure;
bar(r);
xlabel('Page Index');
ylabel('PageRank Score');
title('PageRank via Eigenvector of Google Matrix');
grid on;