clear
clc
% 1 Carga de dataset
dataset = 'pavia512.mat'; % Nombre del dataset
if(~strcmp(dataset,'random.mat'))
    load(dataset)%carga la imagen
    load('Projection.mat')
    [M,N,L] = size(hyperimg);
    F = reshape(hyperimg,[M*N,L])';
end

% 2 Preprocesa la data y asigna parámetros
F = F./max(F(:)); % Normaliza el dataset

noises = 5:5:30; % cuanto ruido se le agrega

type = 0; % 0-gaussian, 1-uniform, 2-binary, que matriz de sensado usar
m= [1, 2, 4, 8, 16];% cuantos "snapshots" usar (compresion)
shots = m;


rho =0; % por ahora dejar en 0
reps = 3; %cuantas repeticioneshacer

%Sumar a la reconstruccion !!
meanhy = mean(F,2); %calcula la media 
hy = F;
hy = hy-meanhy;

Sigmareal= (hy)*(hy)'./size(hy,2);%este es el ground truth

S = 1;

partitions = [4,8,16,32,64,128,256, 512, 1024]; %variar este numero, usualmente 32 o 64 funciona

% Inicializacion de variables de error y para guardar
Error_psnr = zeros(reps, length(partitions));
Sigmas2 = cell(reps, length(partitions));
psnrs_imrec = zeros(reps, length(partitions)); 
psnrs_rec = zeros(reps, length(partitions));
reconstrucciones = cell(reps, length(partitions));

auxiliarinsv2 = reshape(hy', [M, N, L]);

softpar = [1, 1.5, 2, 3]; %se usa un gradiente filtrado, eso dice que tan fuerte es el filtro gausiano usado

for rep = 1:reps        
        for p = 1:length(softpar)
           partition = 32; %partitions(p);
           samples_t = (M/S)*(N/S);
           samples = floor(samples_t/partition);
                      
            D =kron(speye(L),kron(speye(M/S),kron(ones(1,S),kron(speye(N/S),ones(1,S)))))/(S^2);
            Y = reshape(D* auxiliarinsv2( : ) , [M/S,N/S,L]);

            Low_res = zeros(N, M, L);
            for i=1:L
                Low_res(:,:,i) = kron(ones(S,S),Y(:,:,i));
            end
                        
            Low_res = reshape(Low_res,[M*N,L])';

            it = 1800;%iteraciones del algoritmo principal
            X=cell(1,partition);%aca se van a almacenar las particiones 
            Xl = cell(1,partition);
            indices = cell(1,partition);
            
            st=0;

            hl = reshape(Y,[(M/S)*(N/S),L])';

            ddd = randperm((M/S)*(N/S));
            vectorpos = 1:samples;            

            for i=1:partition            
                Xl{i} = hl(:, ddd(vectorpos));
                indices{i} = ddd(vectorpos);
                vectorpos = vectorpos+samples;
                
            end

            Yl = cell(1,partition);

            type = 0; % 0-gaussian, 1-uniform, 2-binary, que matriz de sensado usar

            P1 = P(1:partition);
            s=1;

            for i=1:partition%realiza el proceso de muestreo y usa los valores singulares para mejor inversion              
                Yl{i} = P1{i}(:, 1:shots(end))'*Xl{i};
                P1{i} = P1{i}(:, 1:shots(end));
            end

            Y2 = Yl;                
            
            mt = 0.002;%controla el rango de la solucion
            tic
            for i=1:20 %aca busco un parametro mt que produzca una matrix con un rago ente 10 y 12
                [Sigma1,~]=estimate_cov6(P1,Y2,300,mt,noises(end),Sigmareal, softpar(p),0,0,0);
                if(rank(Sigma1)>=10 && rank(Sigma1)<=12)
                   break; 
                elseif rank(Sigma1)<10
                    mt=mt/1.5;
                elseif rank(Sigma1)>12
                    mt=mt*1.4;
                end
            end

            [Sigma2, Yn2]  = estimate_cov6(P1,Y2,it,mt,noises(end),Sigmareal,softpar(p),0,0,0);%este es el algoritmo principal        
            Sigmas2{rep, p} = Sigma2;

            Error_psnr(rep, p) = fun_PSNR(Sigmareal, Sigma2);
            
            [W, ~, ~] = svd(Sigma2);
            Fr = cell(1, partition);
            imrec = zeros(L, (M/S)*(N/S));
            r = rank(Sigma2);
            % Algoritmo de reconstrucción
            for i = 1:partition
                Fr{i} = W(:,1:r)*pinv((P1{i}(:, 1:shots(end))')*W(:,1:r))*Yl{i};
                imrec(:, indices{i}) = Fr{i};
            end
            
            psnrs_imrec(rep, p) = fun_PSNR(hl, imrec); 
            
            auxiliarinsv = reshape(imrec', [M/S, N/S, L]);
            
            Low_res = reshape(Low_res', [M, N, L]);
            psnrs_rec(rep, p) = fun_PSNR(Y, auxiliarinsv); 
            reconstrucciones{rep, p} = auxiliarinsv; 
        end
       S = S*2;
end

%% Varying m with S = 4

close all

ejeX = 0:length(shots) - 1;


b = bar(ejeX, Error_psnr(:, 1:size(shots, 2)), 'grouped');

for i = 1:size(Error_psnr, 1)
    xtips1 = b(i).XEndPoints;
    ytips1 = b(i).YEndPoints;
    labels1 = string(b(i).YData);
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
        'VerticalAlignment','bottom')
    b(i).DisplayName = strcat("S=", num2str(2^(i-1)));
    %legend( strcat("S=", num2str(2^(i-1))) )
end
legend()
 
%legend( "S = 1", "S = 2", "S = 4", "S = 8", "S = 16", "S = 32")
xlabel("2^{Numero de particiones}") , ylabel("PSNR"), title("Resultados")

[maximos, I] = max(Error_psnr(1:reps, :)');
figure;

I = 2.^(I );

ejeX2 = categorical({  strcat('S=1, Partitions = ' , num2str(I(1))) , strcat('S=2, Partitions = ' , num2str(I(2))), strcat('S=4, Partitions = ' , num2str(I(3))), strcat('S=8, Partitions = ' , num2str(I(4))) });
b1 = bar(ejeX2, maximos);

xtips2 = b1(1).XEndPoints;
ytips2 = b1(1).YEndPoints;
labels2 = string(b1(1).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xlabel("Best results of each division S") , ylabel("PSNR"), title("Best results")
%% Covariance matrix noise images S = 4

close all,
figure, sgtitle("S = 4")
for i = 1:size(noises, 2)
    subplot(2,3,i),
    imshow(Sigmas2{3, i}, [min(min(Sigmareal)), max(max(Sigmareal))]), title(strcat("SNR = ", num2str(noises(i)))), colormap parula
end
%% Noise results

close all

ejeX = noises(1):5:noises(end);

b = bar(ejeX, Error_psnr(:, 1:size(noises, 2)), 'grouped');

for i = 1:size(Error_psnr, 1)
    xtips1 = b(i).XEndPoints;
    ytips1 = b(i).YEndPoints;
    labels1 = string(b(i).YData);
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
        'VerticalAlignment','bottom')
    b(i).DisplayName = strcat("S=", num2str(2^(i-1)));
    %legend( strcat("S=", num2str(2^(i-1))) )
end
legend()
 
%legend( "S = 1", "S = 2", "S = 4", "S = 8", "S = 16", "S = 32")
xlabel("Noise") , ylabel("PSNR"), title("Resultados")

[maximos, I] = max(Error_psnr(1:reps, :)');
figure;


ejeX2 = categorical({  strcat('S=1, Noise = ', num2str(I(1)) ) , strcat('S=2, Noise = ' , num2str(I(2))), strcat('S=4, Noise = ' , num2str(I(3))), strcat('S=8, Noise = ' , num2str(I(4))) });
b1 = bar(ejeX2, maximos);

xtips2 = b1(1).XEndPoints;
ytips2 = b1(1).YEndPoints;
labels2 = string(b1(1).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xlabel("Best noise results of each division S") , ylabel("PSNR"), title("Best results")

%%
close all

subplot(1,2,1), sgtitle("Reconstructed image")
imshow(reconstrucciones{1, 8}(:,:, 5), []), colormap gray, title("Reconstructed image")
subplot(1,2,2)
imshow(auxiliarinsv2(:,:, 5), []), colormap gray, title("Original image")
%%
close all
subplot(1,2,1)
imshow(Sigmas2{1, 4}, []), colormap parula
subplot(1,2,2)
imshow(Sigmareal, []), colormap parula

Sigma2_n = ( Sigmas2{1, 4} - min(min(Sigmas2{1, 4})) )./(max(max(Sigmas2{1, 4})) - min(min(Sigmas2{1, 4})));
Sigmareal_n = ( Sigmareal - min(min(Sigmareal)) )./ ( max(max(Sigmareal)) -  min(min(Sigmareal)) );

fun_PSNR(Sigmareal, Sigmas2{1, 4})
fun_PSNR(Sigmareal_n, Sigma2_n)

figure, 
subplot(1,2,1)
imshow(Sigma2_n), colormap gray
subplot(1,2,2)
imshow(Sigmareal_n), colormap gray
%% Resultados reconstrucciones
close all

ejeX = 2:length(partitions)+1;

b = bar(ejeX, psnrs_rec(1, :), 'grouped');

for i = 1:size(psnrs_rec(1,:), 1)
    xtips1 = b(i).XEndPoints;
    ytips1 = b(i).YEndPoints;
    labels1 = string(b(i).YData);
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
        'VerticalAlignment','bottom')
    b(i).DisplayName = strcat("S=", num2str(2^(i-1)));
end
legend()

xlabel("2^{Numero de particiones}") , ylabel("PSNR"), title("Best results")


figure, sgtitle("Reconstrucciones S = 1")

for i = 1:size(psnrs_rec(1,:), 2)
    subplot(3,3, i),
    imshow(reconstrucciones{1, i}(:,:, 5), []), colormap gray,
    title(strcat("PSNR = ", num2str(psnrs_rec(1,i))))
end
%%
% Results of psnrs
close all

ejeX = 2:length(partitions)+1;


b = bar(ejeX, Error_psnr, 'grouped');

for i = 1:size(Error_psnr, 1)
    xtips1 = b(i).XEndPoints;
    ytips1 = b(i).YEndPoints;
    labels1 = string(b(i).YData);
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
        'VerticalAlignment','bottom')
    b(i).DisplayName = strcat("S=", num2str(2^(i-1)));
    %legend( strcat("S=", num2str(2^(i-1))) )
end
legend()
 
%legend( "S = 1", "S = 2", "S = 4", "S = 8", "S = 16", "S = 32")
xlabel("2^{Numero de particiones}") , ylabel("PSNR"), title("Resultados")

[maximos, I] = max(Error_psnr(1:reps, :)');
figure;

I = 2.^(I + 1);

ejeX2 = categorical({  strcat('S=1, Partitions = ' , num2str(I(1))) , strcat('S=2, Partitions = ' , num2str(I(2))), strcat('S=4, Partitions = ' , num2str(I(3))), strcat('S=8, Partitions = ' , num2str(I(4))) });
b1 = bar(ejeX2, maximos);

xtips2 = b1(1).XEndPoints;
ytips2 = b1(1).YEndPoints;
labels2 = string(b1(1).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xlabel("Best results of each division S") , ylabel("PSNR"), title("Best results")

%%
psnr(Sigmareal, Sigmas2{3, 2})

%%
figure;
subplot(2,2,1), sgtitle("Sigmas Visualization")
imshow(Sigmareal, []), colormap parula, title("Ground Truth"),

subplot(2,2, 2),
imshow(Sigmas2{1,4}, [min(min(Sigmareal)), max(max(Sigmareal))]), colormap parula, title("S = 1, PSNR = " + 47.1599),

subplot(2,2, 3),
imshow(Sigmas2{2,2}, [min(min(Sigmareal)), max(max(Sigmareal))]), colormap parula, title("S = 2, PSNR = " + 17.7557),

subplot(2,2, 4),
imshow(Sigmas2{3,2}, [min(min(Sigmareal)), max(max(Sigmareal))]), colormap parula, title("S = 4, PSNR = " + 10.9753),


%% Comparando sigma128 y sigmareal

im128 = reshape(Y,[128*128,L])';
im128 = im128./max(im128(:));

meanhy = mean(im128,2); %calcula la media 
h128 = im128;
h128=h128-meanhy;
h128=h128./max(abs(h128(:))); %normaliza la imagen

sigma128 = (h128)*(h128)'./size(h128,2);%este es el ground truth
psnr128 = psnr(sigma128, Sigmareal);

figure
subplot(1,2,1) , sgtitle("PSNR = " + psnr128)
imshow(sigma128, []), colormap parula, title("Sigma128")

subplot(1,2,2)
imshow(Sigmareal, []), colormap parula, title("Sigmareal")

%%
imshow(Sigmas2{2}, []), title("PSNR = " + max(Error_psnr(:))), colormap parula

%% Eigenvectors and eigenvalues
close all

[Vo, ~, ~] = svd(Sigmas2{3, 6});
[Vt, ~, ~] = svd(Sigmareal);

rank(Sigmas2{3, 6})
figure,
subplot(1,2,1), sgtitle("PSNR = " + fun_PSNR(Vt,Vo))
imshow(Vo, []), colormap parula, title("Obtained Covariance matrix eigenvectors")

subplot(1,2,2)
imshow(Vt, []), colormap parula, title("Ground truth eigenvectors")


figure, sgtitle("Eigenvectors comparison")
for i = 1:8
    subplot(2,4, i)
    plot(Vo(:, i), 'LineWidth', 2), hold on, plot(Vt(:,i), 'LineWidth', 2), legend("Estimated", "Real"), title("Eivengector Comparison " + i),
    xlim([1, 102])
    hold off
    
    angulo(i) = acosd( abs(( Vo(:, i)'*Vt(:,i) ) / (sqrt(Vo(:, i)'*Vo(:, i))*sqrt(Vt(:, i)'*Vt(:, i))) ));
end

figure,

b1 = bar(1:8, angulo); title("Angle between eigenVectors in degrees"), ylabel("Angle"), xlabel("Eigenvector")
xtips2 = b1(1).XEndPoints;
ytips2 = b1(1).YEndPoints;
labels2 = string(b1(1).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

%% 
S = 1;
D =kron(speye(L),kron(speye(M/S),kron(ones(1,S),kron(speye(N/S),ones(1,S)))))/(S^2);
Y = reshape(D* auxiliarinsv2( : ) , [M/S,N/S,L]);