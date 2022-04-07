

function [z, out] = ClusteringCost(s, X, Method)

    % Method: DB or CS
    % Feature Selection
    if sum(s)==0
      z = 1000; 
      out.Select = 0;
    else
    Select = s > 0;
    Feats = X(:,Select);
    
    [ind, m] = kmeans(Feats, 2);
    
    switch Method
        case 'DB'
            [z, out] = DBIndex(m, X(:,Select));
            out.m = m;
            
        case 'CS'
            [z, out] = CSIndex(m, X(:,Select));
            
    end
    
    out.Select = Select;
    end
end