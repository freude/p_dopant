clear all;

% num_cells=4;
% T=80;

%num_cells=60;
%T=8;

EigVec1=load([pwd,'/dis_scr/M.mat']);
spectrum=EigVec1.a1;
kk=EigVec1.kk;
Nbands=EigVec1.Nbands;
bands=EigVec1.bands;
EigVec1=EigVec1.EigVec1;

num_cells=40;
T=8;

coorsys=CoordSys(num_cells,T,'au');
coorsys.set_origin_cells(21);
x=coorsys.x();
[X,Y,Z]=meshgrid(x,x,x);
s=size(X);

coorsys1=CoordSys(num_cells,1,'au');
coorsys1.set_origin_cells(21);
x1=coorsys1.x_ep();
[X2,Y2,Z2]=ndgrid(x1,x1,x1);

M=zeros(length(x1),length(x1),length(x1));
M1=zeros(Nbands,3,length(x),length(x),length(x));

aa=0;

for jj1=1:3    
    jj1
    p1 = importdata([pwd,'/dis_scr/v',num2str(4-jj1-1),'/ff_0.dat']);    
    for jj3=1:Nbands
        
        j1=0;
        for j=1:(length(squeeze(p1(:,1))))
            if (p1(j,1)==111)&&(p1(j,2)==111)&&(p1(j,3)==111)
                j1=j1+1;
                aa(j1)=j;
            end;
        end;
        
        F1=squeeze(p1(aa(bands(jj3))+1:aa(bands(jj3)+1)-1,4));
        X1=squeeze(p1(aa(bands(jj3))+1:aa(bands(jj3)+1)-1,1));
        Y1=squeeze(p1(aa(bands(jj3))+1:aa(bands(jj3)+1)-1,2));
        Z1=squeeze(p1(aa(bands(jj3))+1:aa(bands(jj3)+1)-1,3));
        
        Fq = TriScatteredInterp(X1,Y1,Z1,F1);
        M=Fq(X2,Y2,Z2);
        M(isnan(M))=0;
        ME=trapz(x1,trapz(x1,trapz(x1,abs(squeeze(M)).^2,3),2),1);
       
       if abs(max(M(:)))<abs(min(M(:)))
           M = -M;
       end;
       
       %regular grid interp
       F = griddedInterpolant(X2, Y2, Z2, M, 'cubic');
       M = F(X, Y, Z);   
       M(isnan(M))=0;
       M1(jj3,jj1,:,:,:) = M./sqrt(ME);                
       
   end;
end;
 
%%

wf1 = zeros(6,length(x),length(x),length(x));
for j=1:6
    wf1(j,:,:,:) = abi_read(num_cells,T,kk(j,:)).*exp(1i*(kk(j,1).*X+kk(j,2).*Y+kk(j,3).*Z));
end;

num_bs=12;

bas_fun = cell(num_bs+1, 1);
bas_fun{1} = x;

for jj=1:num_bs
    F=zeros(length(x),length(x),length(x));
    alt=1;
    
    for j2=1:Nbands
        for j3=1:6
            if (sum(kk(j3,:))<0)
                alt = 1;
            else
                alt = 1;
            end;
            if (j3==1)||(j3==2)
                jjj=1;                
            end;
            if (j3==3)||(j3==4)
                jjj=2;                
            end;
            if (j3==5)||(j3==6)
                jjj=3;                
            end;            
            F = F+alt*EigVec1(j2+Nbands*(j3-1),jj).*squeeze(M1(j2,jjj,:,:,:)).*squeeze(wf1(j3,:,:,:,:));
        end;
    end;
    
    ME=trapz(x,trapz(x,trapz(x,abs(F).^2,3),2),1);
    if (max(abs(real(F(:))))>max(abs(imag(F(:)))))
        bas_fun{jj+1} = real(F)/sqrt(ME);
    else
        bas_fun{jj+1} = imag(F)/sqrt(ME);
    end;
end;

%bas_fun_c{1} = fftshift(fftfreq(length(x)+400, x(3)-x(2)));
save(strcat(pwd,'/dis_scr/bas_fun.mat'), 'bas_fun', '-v7.3');
main_script_p3
%save(['/data/users/mklymenko/abinitio_software/abi_files/tpaw/ready_code/bas_fun_c.mat'], 'bas_fun_c', '-v7.3');

% figure(4);surf(abs(squeeze(Ff(:,:,411))),'LineStyle','none'),view(0,90)
fftmap=cmapdef(2^12,[0 0.02 0.05 0.3 1],...
[[0 0 0]/255;...
[255 0 0]/255;...
[200 100 0]/255;...
[200 200 0]/255;...
[255 255 255]/255],...
'linear');
colormap(fftmap)


fftmap=cmapdef(2^12,[0 0.1 0.3 0.5 1],...
[[0 0 0]/255;...
[255 0 0]/255;...
[200 100 0]/255;...
[200 200 0]/255;...
[255 255 255]/255],...
'linear');
colormap(fftmap)

