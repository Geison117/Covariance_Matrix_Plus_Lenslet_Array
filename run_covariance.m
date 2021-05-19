clear
clc
dataset = 'pavia512.mat'; %cambiar por random.mat para usar una señal random
if(~strcmp(dataset,'random.mat'))
    load(dataset)%carga la imagen
    load('Projection.mat')
    [M,N,L] = size(hyperimg);
    F = reshape(hyperimg,[M*N,L])';
end

F = F./max(F(:));

meanhp = mean(F,2); %calcula la media

noise = 30; % cuanto 
samples_t = M*N;


type = 0; % 0-gaussian, 1-uniform, 2-binary, que matriz de sensado usar
m=16;% cuantos "snapshots" usar (compresion)
shots = m;


rho =0; % por ahora dejar en 0
reps = 1; %cuantas repeticioneshacer
Sigmas = cell(1,reps);


meanhy = mean(F,2); %calcula la media 
hy = F;
hy=hy-meanhy;
hy=hy./max(abs(hy(:))); %normaliza la imagen

Sigmareal= (hy)*(hy)'./size(hy,2);%este es el ground truth

S = 4;

partitions = [2,8,16,32,64,128,256,512,1024]; %variar este numero, usualmente 32 o 64 funciona
Error_psnr = zeros(reps, length(partitions));
Sigmas2 = cell(reps, length(partitions));

for rep = 1:reps        
        
        for p = 1:length(partitions)
           partition = partitions(p);
           samples = floor(samples_t/partition);
            Low_res_aux = imresize(hyperimg,[N/S,M/S]);
            Low_res = zeros(N, M, L);
            for i=1:L
                Low_res(:,:, i)= kron(ones(S,S),Low_res_aux(:,:,i));
            end
            Low_res = reshape(Low_res,[M*N,L])';
            Low_res = Low_res./max(Low_res(:));
            
            meanhq = mean(Low_res,2); %calcula la media        

            it = 1800;%iteraciones del algoritmo principal
            X=cell(1,partition);%aca se van a almacenar las particiones 
            Xl = cell(1,partition);
            st=0;

            hl = Low_res;

            ddd = randperm(M*N);
            %sss=floor(M*N/partition);
            vectorpos=1:samples;       

            hl=hl-meanhq;
            hl=hl./max(abs(hl(:))); %normaliza la imagen

            for i=1:partition            
                Xl{i} = hl(:, ddd(vectorpos));
                vectorpos = vectorpos+samples;
            end

            Yl = cell(1,partition);

            type = 0; % 0-gaussian, 1-uniform, 2-binary, que matriz de sensado usar
            
            m=16;% cuntos "snapshots" usar (compresion)
            shots=m;
            P1 = P(1:partition);
            s=1;

            for i=1:partition%realiza el proceso de muestreo y usa los valores singulares para mejor inversion              
                Yl{i} = P1{i}(:, 1:shots)'*Xl{i};
            end

            Y2 = Yl;                

            softpar=1;%se usa un gradiente filtrado, eso dice que tan fuerte es el filtro gausiano usado
            mt = 0.002;%controla el rango de la solucion
            tic
            for i=1:20 %aca busco un parametro mt que produzca una matrix con un rago ente 10 y 12
                [Sigma1,~]=estimate_cov6(P1,Y2,300,mt,noise,Sigmareal, softpar,0,0,0);
                if(rank(Sigma1)>=10 && rank(Sigma1)<=12)
                   break; 
                elseif rank(Sigma1)<10
                    mt=mt/1.5;
                elseif rank(Sigma1)>12
                    mt=mt*1.4;
                end
            end

            [Sigma2, Yn2]  = estimate_cov6(P1,Y2,it,mt,noise,Sigmareal,softpar,0,0,0);%este es el algoritmo principal        
            Sigmas2{rep, p} = Sigma2;
            
            Error_psnr(rep, p) = psnr(Sigmareal, Sigma2);
        end
       S = S*2;
end
%%
%imshow(Low_res_aux(:,:,1), [])

ejeX = [2,8,16,32,64,128,256,512,1024];

aux = 1;
for i=1:6
    plot(ejeX, Error_psnr(i,:), "LineWidth", 2), hold on
    aux = aux*2;
end
xlabel("Numero de particiones") , ylabel("PSNR"), title("Resultados"), ylim([20, 80])
legend( "S = 1", "S = 2", "S = 4", "S = 8", "S = 16", "S = 32")


maximos = max(Error_psnr(1:6, :)');
figure;

plot(1:6, maximos, "LineWidth", 2), 
xlabel("Mejor resultado de cada división S") , ylabel("PSNR"), title("Mejores Resultados")

Low_res_aux = imresize(hyperimg,[N/4,M/4]);
Low_res = zeros(N, M, L);
for i=1:L
    Low_res(:,:, i)= kron(ones(4,4),Low_res_aux(:,:,i));
end

figure;
imshow(Low_res(:,:, 1), []), colormap gray, title("Simulación de microlentes")


%%
figure;
subplot(2,3, 1),
imshow(Sigmas2{1,5}, []), colormap parula, title("S = 1, PSNR = " + 76.7350),

subplot(2,3, 2),
imshow(Sigmas2{2,2}, []), colormap parula, title("S = 2, PSNR = " + 45.7928),

subplot(2,3, 3),
imshow(Sigmas2{3,2}, []), colormap parula, title("S = 4, PSNR = " + 40.4628),

subplot(2,3, 4),
imshow(Sigmas2{4,1}, []), colormap parula, title("S = 8, PSNR = " + 38.5571),

subplot(2,3, 5),
imshow(Sigmas2{5,1}, []), colormap parula, title("S = 16, PSNR = " + 38.2757),

subplot(2,3, 6),
imshow(Sigmas2{6,1}, []), colormap parula, title("S = 32, PSNR = " + 36.8151),