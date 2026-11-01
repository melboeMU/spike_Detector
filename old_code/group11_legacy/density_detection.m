
n = 9;
result =[];
for region = 1:n 
 
    importFileName = sprintf('Pat1/glm_input_clinical_unfiltered_pat_1_region%d_factor5.mat',region);
    load(importFileName) 

    num_ones  = nnz(X == 1);   % number of non-zero logicals
    num_zeros = nnz(X == 0);

    fprintf('num_seroz (sparsity) = %d\n',num_zeros);
    fprintf('num_ones (density) = %d\n',num_ones);
    
    % check value 
    total = num_ones + num_zeros;
   
    fprintf('Total = %d \n', total);
    
    
    density_unfiltered_clinical_data = num_ones / total ;
    
    fprintf(' %d  density_unfiltered_clinical_data = %4f \n\n\n', region, density_unfiltered_clinical_data);
    
    % m = 5/256 ;
    % 
    % average_ones_per_second =  average_ones_per_bin / m ;
    % fprintf('average_ones_per_second = %4f\n\n',average_ones_per_second);
    % 
    % result(region).average_ones_per_bin = average_ones_per_bin;   
    % result(region).average_ones_per_second = average_ones_per_second;



end

% labels = { ...
%     'Temporal\_Mid\_R', ...
%     'ParaHippocampal\_R', ...
%     'Amyg\_Hipp\_R', ...
%     'Insula\_R', ...
%     'Frontal\_inf\_All\_R', ...
%     'Frontal\_Mid\_All\_R', ...
%     'Precentral\_R' };
% 
% fprintf('\n\n Region     avg1_bin    avg1_sec \n');
% fprintf('-------------------------------------\n');
% 
% for i = 1:length(result)
%     fprintf('%s           %4f      %4f   \n', ...
%         labels{i}, ...
%         result(i).average_ones_per_bin, ...
%         result(i).average_ones_per_second);
% end
% 

