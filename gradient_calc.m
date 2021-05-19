function [grad,D1] = gradient_calc(P,Sigmao,L,softp,partition,Sigmas,Di,Dj,D1)
    tt=randperm(partition);
    grad=zeros(size(Sigmao));
    for j=1:floor(partition/1) %calcula el gradiente
        R=((P{tt(j)}'*Di{tt(j)})'*((P{tt(j)}'*(Sigmao)*P{tt(j)})-Sigmas{tt(j)})*(Dj{tt(j)}*P{tt(j)}'));
        Sigmao = Sigmao - ((1/L).*R);
        grad=grad+R;
    end
    if(softp>0) %filtrar o no el gradiente
        g1=grad-D1;
        grad=imgaussfilt(grad-D1,softp);
        D1 = 0.8.*D1 + 0.2.*(g1-grad);
    else
        grad=grad-D1;
    end
end