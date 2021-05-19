function obj=calc_obj_fun(D,Sigma1,Sigmas)
    
    obj = 0;
    for i=1:length(D)
        obj = obj + norm(D{i}'*Sigma1*D{i}-Sigmas{i},'fro');
    end
    obj=obj./length(D);
    obj=obj+trace(Sigma1);
end