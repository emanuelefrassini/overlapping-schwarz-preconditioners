clear all; clc;
L = 1;
l = 1;
k = 3;
m = 4;
n = 10;
f = @(x,y) 2*(x.*(1-x)+y.*(1-y));
%fine domain m x m [xm,ym]
hm = L/(k*m);
x_m = -l*hm:hm:(L+l*hm);
y_m = -l*hm:hm:(L+l*hm);
[xm,ym] = meshgrid(x_m,y_m);

%divide in overlapping domain over x; stillt do
xl = cell(k,1);
I = 2:m:(k)*m+1+2*l;
xl{1} = xm(1:m+1+2*l,1:m+1+2*l);
for i = 2:k-1
    xl{i} = xm(I(i)-l:I(i+1)+l,I(i)-l:I(i+1)+l);
end
xl{k} = xm(end-(m+2*l):end,end-(m+2*l):end);

%divide in overlapping domain over y;
yl = cell(k,1);
I = 2:m:(k)*m+1+2*l;
yl{1} = ym(1:m+1+2*l,1:m+1+2*l);
for i = 2:k-1
    yl{i} = ym(I(i)-l:I(i+1)+l,I(i)-l:I(i+1)+l);
end
yl{k} = ym(end-(m+2*l):end,end-(m+2*l):end);
%now all the couples (xl{i},yl{j}) represent the overlapping domains

%stencil is: [c3 c6 c3;
%             c2 c5 c2;
%             c1 c4 c1];

%construct the matrix for inner subdomain, with elimination of bc-> (m+1)^2;
coeff = 1/3;%1/hm^2;
c1ex = coeff;%0;
c2ex = coeff;%-coeff;
c3ex = coeff;%0;
c4ex = coeff;%-coeff;
c5ex = 8*coeff;%4*coeff;
c6ex = coeff;%-coeff;
il = diag(ones(1,m+2*l),-1); 
iu = diag(ones(1,m+2*l),1); 
Mc = diag(c2ex*ones(1,m+2*l),1) + diag(c5ex*ones(1,m+1+2*l)) + diag(c2ex*ones(1,m+2*l),-1);
Ml = diag(c1ex*ones(1,m+2*l),1) + diag(c4ex*ones(1,m+1+2*l)) + diag(c1ex*ones(1,m+2*l),-1);
Mu = diag(c3ex*ones(1,m+2*l),1) + diag(c6ex*ones(1,m+1+2*l)) + diag(c3ex*ones(1,m+2*l),-1);
A = kron(iu,Mu)+kron(il,Ml)+kron(eye(m+1+2*l),Mc);
B = A;
%eliminate boundary conditions for A
A = A(m+4:end,m+4:end);
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
for i = 1:k*m+2*l-1
    point = [point,k*m+2+2*l+(i-1)*(k*m+2*l+1),k*m+2+2*l+k*m+2*l+(i-1)*(k*m+1+2*l)];
end
globalboundary = [1:(k*m+1+2*l),point,(k*m+1+2*l)^2-(k*m+2*l):(k*m+1+2*l)^2]; %still domain (0,L)
globalpoints = 1:(k*m+1+2*l)^2;

local = [];
for i = 1:m+1
    local = [local,m+2+2*l+(i-1)*(m+1+2*l),m+2+2*l+m+2+(i-1)*(m+1+2*l)];
end
localboundary = [1:(m+1+2*l),local,(m+1+2*l)^2-(m+2):(m+1+2*l)^2];
localpoints = 1:(m+1+2*l)^2;



%u = zeros(n,(m+1)^2);
%F_m = zeros((m+1)^2,1);
ucompl = zeros(m+1+2*l,m+1+2*l);            
u0 = zeros(k*m+1+2*l,k*m+1+2*l);
%uglobal = zeros(k*m+1+2*l,k*m+1+2*l);
%alternating Schwarz algrotihm
%while err>toll
iter=1;
while(iter<50) %%%%SOLUTION STILL THE SAME FOR EVERY ITERATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for i = 1:k
        for j = 1:k
%------------------------------------------------------------------------%
            %FIRST STEP OF THE ALGORITHM
            % we compute solution in inner points of Omega_i'
            %domainindex = globalindex_k(i, j, k);
            xi = xl{i};
            yj = yl{j};
            s = size(xi,1);
            xint = xi(2:s-1,2:s-1);
            yint = yj(2:s-1,2:s-1);
            [rows,cols] = find(xm == xi(1,1));
            [row,col] = find(ym == yj(1,1));
            p = cols(1);
            q = row(1);
            
            F = f(xint,yint);
            for s = 1:m+1
                for t = 1:m+1
                    G = globalindex_m1(s,t,m);
                    F_m(G) = hm*F(s,t);
                end
            end
%              for iiii=1:length(indii)
%                  F_mm(z) = F_m(indii(iiii));
%                  z=z+1;
%             end
            %solving for inner points in the domain (i,j) and put in
            %matrix form the inner solution (m+1)^2
            u = A\F_m';%u(domainindex,:) = A\F_m;
            
            %ufull = 
            %kkk=1;
            Uin=[];
            for s = 1:m+1
                for t = 1:m+1
                    pp= globalindex_m1(s,t,m);
                    Uin(s,t) = u(pp);
                end
            end
            %put values computed in uglobal
            ind=0;
            for ind_i = 1:m+1
                for ind_j = 1:m+1
                    ind = globalmapping_m1(p,q,ind_i,ind_j,m,l,k);
                    for ii = 1:k*m+1+2*l
                        for jj = 1:k*m+1+2*l
                           if ind == globalindex_glob(ii, jj, k,m,l)
                               uglobal(ii,jj) = Uin(ind_i,ind_j);
                           end
                        end
                    end
                end
            end
%              figure(2);
%             hold on;
%              surf(xi,yj,zeros(size(xi)));
%              figure(1);
%              hold on;
%              surf(xint,yint,Uin)
            %Uin IS A TENSOR CONTAINING IN THE FIRST DIMENSION(i,:,:) THE INNER SOLUTION ON
            %THE DOMAIN i
            
            %-----------------------------------------------------------------------%
            %SECOND STEP OF THE ALGORTIHM
            ucompl = zeros(m+1+2*l,m+1+2*l);            

            ucompl(2:end-1,2:end-1) = Uin(:,:);
            indexmodified = [];
            for ind_i = 1:m+1+2*l
                for ind_j = 1:m+1+2*l
                index = globalindex_m3(ind_i,ind_j,m,l);
                    for tt = 1:length(localboundary)
                        if index == localboundary(tt)
                            ucompl(ind_i,ind_j) = u0(ind_i,ind_j);
                            indexmodified = [indexmodified, globalmapping_m3(p,q,ind_i,ind_j,m,l,k)];
                        end
                    end
                end
            end
            %put values computed in the uglobal
            for ind_i = 1:m+1+2*l%k*m+1+2*l
                for ind_j = 1:m+1+2*l
                    for indglob_i = 1:k*m+1+2*l
                        for indglob_j = 1:k*m+1+2*l
                            for ind = 1:length(indexmodified)
                                if globalindex_glob(indglob_i,indglob_j,k,m,l) == indexmodified(ind)
                                    uglobal(indglob_i,indglob_j) = ucompl(ind_i,ind_j);
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
            for ii = 1:m+1+2*l
                for jj = 1:m+1+2*l
                    globalindexmapped(z) = globalmapping_m3(p,q,ii,jj,m,l,k);
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
                for ii = 1:k*m+1+2*l
                    for jj = 1:k*m+1+2*l
                       if differen(z) == globalindex_glob(ii, jj, k,m,l)
                           uglobal(ii,jj) = u0(ii,jj);
                       end
                    end
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              for ind_ii = 1:k*m+1+2*l
%                  for ind_jj = 1:k*m+1+2*l
%                  ppp = globalindex_glob(ind_ii,ind_jj,k,m,l);
%                      for ii = 1:length(globalboundary)
%                          if ppp == globalboundary(ii)
%                              uglobal(ind_ii,ind_jj) = 0;
%                          end
%                      end
%                  end
%              end
           
            u0 = uglobal;
            uuu(iter,:,:)=uglobal;
        end
    end
    iter=iter+1;
end




   for ind_ii = 1:k*m+1+2*l
                 for ind_jj = 1:k*m+1+2*l
                 ppp = globalindex_glob(ind_ii,ind_jj,k,m,l);
                     for ii = 1:length(globalboundary)
                         if ppp == globalboundary(ii)
                             uglobal(ind_ii,ind_jj) = 0;
                         end
                     end
                 end
   end





 figure(2)
 surf(xm,ym,uglobal);
% figure(1)
% contourf(xm,ym,uglobal)

%surf(xm(2:end-1,2:end-1),ym(2:end-1,2:end-1),uglobal(2:end-1,2:end-1))


%p,q coordinates of the 1st point of the subdomain (1,1) 
function r = globalindex_k(i, j, k) %index for inner (m+1)
    r = i + (j-1)*(k); 
end 

function r = globalindex_glob(i, j, k,m,l) 
    r = i + (j-1)*(k*m+1+2*l); 
end

function r = globalmapping_m3(p,q,i,j,m,l,k)
    loc = globalindex_m3(i,j,m,l);
    %r = loc+(p-1)+(m+2*l+1)*(q-1)+(floor(loc/(m+2*l+1))-1+q)*(k-1)*m;
    r = loc+(p-1)+(m+2*l+1)*(q-1)+(j+q-2)*(k-1)*m;
end

function r = globalmapping_m1(p,q,i,j,m,l,k)
    loc = globalindex_m1(i,j,m);
   % r = loc+(p-1)+(m+2*l+1)*(q-1)+(floor(loc/(m+2*l+1))-1+q)*(k-1)*m;
   % r = loc+(p-1)+(m+1)*(q-1)+(j+q-2)*(k-1)*m;
    r = loc + p + q*(k*m + 2*l + 1) + (j-1)*(m+2)*(k-1);
end

function r = globalindex_m1(i, j, m) %index for inner (m+1)
    r = i + (j-1)*(m+1); 
end

function r = globalindex_m3(i, j, m,l) %index for complete (m+1+2*l)
    r = i + (j-1)*(m+1+2*l); 
end