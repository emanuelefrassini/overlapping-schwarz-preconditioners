clear all; clc;
L = 1;
l = 1;
k = 15;
m = 3;
n = 10;
f = @(x,y) 2*(x.*(1-x)+y.*(1-y));
uex=@(x,y) 1/6*(x.^4-2*x.^3+x+y.^4-2*y.^3+y);
%fine domain m x m [xm,ym]
hm = L/(k*m);
x_m = 0:hm:(L);
y_m = 0:hm:L;
[xm,ym] = meshgrid(x_m,y_m);
figure(4)
surf(xm,ym,uex(xm,ym))
%debugging
% globalindex(3,2,m,2*l)
% globalmapping(1,1,1,2,k,2*l,m)

%divide in overlapping domain over x; stillt do
xl = cell(k,1);
I = 1:m:(k)*m+1+2*l;
xl{1} = xm(1:m+1+l,1:m+1+l);
for i = 2:k-1
    xl{i} = xm(I(i)-l:I(i+1)+l,I(i)-l:I(i+1)+l);
end
xl{k} = xm(end-(m+l):end,end-(m+l):end);

%divide in overlapping domain over y;
yl = cell(k,1);
I = 1:m:(k)*m+1+2*l;
yl{1} = ym(1:m+1+l,1:m+1+l);
for i = 2:k-1
    yl{i} = ym(I(i)-l:I(i+1)+l,I(i)-l:I(i+1)+l);
end
yl{k} = ym(end-(m+l):end,end-(m+l):end);
%now all the couples (xl{i},yl{j}) represent the overlapping domains

%stencil is: [c3 c6 c3;
%             c2 c5 c2;
%             c1 c4 c1];

%construct the matrix for inner subdomain, with elimination of bc-> (m+1)^2;
coeff = 1/hm^2;%1/3;%
c1ex = 0;%coeff;%0
c2ex = -coeff;%-coeff;
c3ex = 0;%coeff;%0
c4ex = -coeff;%-coeff;
c5ex = 4*coeff;%4*coeff;
c6ex = -coeff;%-coeff;
il = diag(ones(1,m+2*l),-1); 
iu = diag(ones(1,m+2*l),1); 
Mc = diag(c2ex*ones(1,m+2*l),1) + diag(c5ex*ones(1,m+1+2*l)) + diag(c2ex*ones(1,m+2*l),-1);
Ml = diag(c1ex*ones(1,m+2*l),1) + diag(c4ex*ones(1,m+1+2*l)) + diag(c1ex*ones(1,m+2*l),-1);
Mu = diag(c3ex*ones(1,m+2*l),1) + diag(c6ex*ones(1,m+1+2*l)) + diag(c3ex*ones(1,m+2*l),-1);
A = kron(iu,Mu)+kron(il,Ml)+kron(eye(m+1+2*l),Mc);
B = A;
%eliminate boundary conditions for A
% A = A(m+4:end,m+4:end);
A=A(m+2+2*l:end,m+2+2*l:end);
z = 0;
for j = 1:m+1
    A(:,1+(j-1)*(m+1+2*l)-z) = []; 
    A(1+(j-1)*(m+1+2*l)-z,:) = []; 
    z = z+1;
    A(:,m+1+2*l+(j-1)*(m+1+2*l)-z) = [];
    A(m+1+2*l+(j-1)*(m+1+2*l)-z,:) = [];
    z = z+1;
end
A(end-(m+1+2*l)+1:end,:) = [];
A(:,end-(m+1+2*l)+1:end) = [];
point = [];
for i = 1:k*m-1
    point = [point,k*m+2+(i-1)*(k*m+1),k*m+2+k*m+(i-1)*(k*m+1)];
end
globalboundary = [1:(k*m+1),point,(k*m+1)^2-(k*m):(k*m+1)^2]; %still domain (0,L)
globalpoints = 1:(k*m+1)^2;




%u = zeros(n,(m+1)^2);
%F_m = zeros((m+1)^2,1);
u0 = zeros(k*m+1,k*m+1);%zeros(k*m+1+2*l,k*m+1+2*l);
%uglobal = zeros(k*m+1+2*l,k*m+1+2*l);
%alternating Schwarz algrotihm
%while err>toll
iter=1;
while(iter<2) 
    for i = 1:k
        for j = 1:k
%------------------------------------------------------------------------%
            %FIRST STEP OF THE ALGORITHM
            % we compute solution in inner points of Omega_i'
            xi = xl{i};
            yj = yl{j};
            sx = size(xi,1);
            sy=size(yj,1);
            xint = xi(1+l:sx-l,1+l:sx-l);
            yint = yj(1+l:sy-l,1+l:sy-l);
            if i==1 
                xint = xi(1:sx-l,1:sx-l);
            end
            if i==k
                xint = xi(1+l:sx,1+l:sx);
            end
            if j==1
                yint = yj(1:sy-l,1:sy-l);
            end
            if j==k
                yint=yj(1+l:sy,1+l:sy);
            end
            if i==1 || i==k || j==1 || j==k
                indexdim=0;
            else 
                indexdim=1;
            end
            if (i==1) ||(i==k)
                layer=l;
            else
                layer=2*l;
            end
            if i==1 || i==k
                xbound=0;
            elseif i~=1 && i~=k
                xbound=1;
            end
            if j==1 || j==k
                ybound=0;
            elseif j~=1 && j~=k
                ybound=1;
            end
            
            local = [];
            for index_I = 1:size(yj,1)-2
                local=[local,size(xi,1)+1+(index_I-1)*size(xi,1),size(xi,1)+1+(size(xi,1)-1)+(index_I-1)*size(xi,1)];
            end
            localboundary = [1:(size(xi,1)),local,local(end)+1:size(xi,1)*size(yj,1)];
            localpoints = 1:size(xi,1)*size(yj,1);
    
            [rows,cols] = find(xm == xi(1,1));
            [row,col] = find(ym == yj(1,1));
            p = cols(1);
            q = row(1);
            F = f(xint,yint);
            for s = 1:m+l
                for t = 1:m+l
                    G = globalindex_m1(s,t,m,l);
                    F_m(G) = f(xint(s),yint(t));
                end
            end
   
            %solving for inner points in the domain (i,j) and put in
            %matrix form the inner solution (m+1)^2
            u = A\F_m';
            
            %ufull = 
            %kkk=1;
            Uin=[];
            for s = 1:m+l
                for t = 1:m+l
                    pp= globalindex_m1(s,t,m,l);
                    Uin(s,t) = u(pp);
                end
            end
           
            %put values computed in uglobal
            ind=0;
            for ind_i = 1:m+l
                for ind_j = 1:m+l
                    ind = globalmapping(p,q,ind_i,ind_j,k,0,m);
                    for ii = 1:k*m+1
                        for jj = 1:k*m+1
                            if ind == globalindex_glob(ii, jj, k,m,l)
                                uglobal(ii,jj) = Uin(ind_i,ind_j); %CHANGED I,J
                            end
                        end
                    end
                end
            end
   
            %-----------------------------------------------------------------------%
            %SECOND STEP OF THE ALGORTIHM
            ucompl = zeros(m+1+l+l*xbound,m+1+l+l*ybound);            
        
            ucompl(1+xbound:end-1,1+ybound:end-1) = Uin(:,:);

          
     
            for ind_i = 1:m+1+l+l*xbound
                for ind_j = 1:m+1+l+l*ybound
                index = globalindex(ind_i,ind_j,m,layer);
                    for tt = 1:length(localboundary)
                        if index == localboundary(tt)
                            for ind_I=1:k*m+1
                                for ind_J=1:k*m+1
                                    if globalindex_glob(ind_I, ind_J, k,m,l)==globalmapping(p,q,ind_i,ind_j,k,layer,m) 
                                        ucompl(ind_i,ind_j) = u0(ind_I,ind_J);
                                        uglobal(ind_I,ind_J)=ucompl(ind_i,ind_j);
                                    end
                                end
                            end
                        end
                    end
                end
            end
       
            %------------------------------------------------------------------------%
            %THIRD STEP OF THE ALGORITHM
            %identify points on Omega\(Omega_i union with its boundary)
            %difference is the vecor containing all the points in that are in Omega but
            %not in the closure of Omega_i
            differen = [];
            z = 1;
            globalindexmapped=[];
            for ii = 1:m+1+l+l*xbound
                for jj = 1:m+1+l+l*ybound
                    globalindexmapped(z) = globalmapping(p,q,ii,jj,k,layer,m);
                    z = z+1;
                end
            end
            globalindexmapped = sort(globalindexmapped);
            jj = 1; 
            for jjj = 1:length(globalpoints)
                z = 0;
                for iii = 1:length(globalindexmapped)
                    if globalindexmapped(iii) == globalpoints(jjj)
                        z = 1;
                    end
                end
                if z == 0
                    differen(jj) = globalpoints(jjj);
                    jj = jj+1;
                end
            end
            z = 1;
            for z = 1:length(differen)
                for ii = 1:k*m+1
                    for jj = 1:k*m+1
                       if differen(z) == globalindex_glob(ii, jj, k,m,l)
                           uglobal(ii,jj) = u0(ii,jj); 
                       end
                    end
                end
            end
    
            u0 = uglobal;
            uuu(iter,:,:)=uglobal;
        end
    end
    iter=iter+1;
end


uexact=uex(xm,ym);
   for ind_ii=1:k*m+1
       for ind_jj=1:k*m+1
           ppp=globalindex_glob(ind_ii,ind_jj,k,m,l);
           for ii=1:length(globalboundary)
               if ppp==globalboundary(ii)
                   uexact(ind_ii,ind_jj)=0;
               end
           end
       end
   end


 
figure(1)
%
% 
 surf(xm,ym,uglobal);
xlabel('$x$', 'Interpreter','latex');
ylabel('$y$', 'Interpreter','latex');
%  hold on
%  surf(xm,ym,uexact,'FaceAlpha',0.5)
% figure(1)
% contourf(xm,ym,uglobal)

%surf(xm(2:end-1,2:end-1),ym(2:end-1,2:end-1),uglobal(2:end-1,2:end-1))
% figure(2)
% surf(xi,yj,ucompl)

%p,q coordinates of the 1st point of the subdomain (1,1) 

function r = globalindex_glob(i, j, k,m,l) 
    r = i + (j-1)*(k*m+1); 
end

function r=globalindex(i,j,m,layer)
    r=i+(j-1)*(m+1+layer);
end
function r= globalmapping(p,q,i,j,k,layer,m)
    loc=globalindex(i,j,m,layer);
    r=loc+p-1+(floor((loc-1)/(m+1+layer))+1-1)*((k-1)*m-layer)+(q-1)*(k*m+1);
end

function r = globalindex_m1(i, j, m,l) %index for inner (m+1)
    r = i + (j-1)*(m+l); 
end