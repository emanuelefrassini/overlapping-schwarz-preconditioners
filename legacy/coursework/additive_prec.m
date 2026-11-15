function [H,h,time,time_m1,time_m2]=additive_prec(k,m,l)

L = 1;
H=L/k;
h=L/(k*m);
f = @(x,y) 2*(x.*(1-x)+y.*(1-y));
hm = h;
x_m = 0:hm:(L);
y_m = 0:hm:L;
[xm,ym] = meshgrid(x_m,y_m);

%OVERLAPPING DOMAINS SUBDIVISIONS
xl = cell(k,1);
I = 1:m:(k)*m+1+2*l;
xl{1}=xm(1:m+1+2*l,1:m+1+2*l);
for i = 2:k-1
    xl{i} = xm(I(i)-l:I(i+1)+l,I(i)-l:I(i+1)+l);
end
xl{k} = xm(end-(m+2*l):end,end-(m+2*l):end);

yl = cell(k,1);
I = 1:m:(k)*m+1+2*l;
yl{1}=ym(1:m+1+2*l,1:m+1+2*l);
for i = 2:k-1
    yl{i} = ym(I(i)-l:I(i+1)+l,I(i)-l:I(i+1)+l);
end
yl{k} = ym(end-(m+2*l):end,end-(m+2*l):end);

%CONSTRUCT THE GLOBAL STIFFNESS MATRIX K
%stencil is: [c3 c6 c3;
%             c2 c5 c2;
%             c1 c4 c1];
coeff = -1/3;
c1ex = coeff;
c2ex = coeff;
c3ex = coeff;
c4ex = coeff;
c5ex = -8*coeff;
c6ex = coeff;
il = diag(ones(1,k*m),-1); 
iu = diag(ones(1,k*m),1); 
Mc = diag(c2ex*ones(1,k*m),1) + diag(c5ex*ones(1,k*m+1)) + diag(c2ex*ones(1,k*m),-1);
Ml = diag(c1ex*ones(1,k*m),1) + diag(c4ex*ones(1,k*m+1)) + diag(c1ex*ones(1,k*m),-1);
Mu = diag(c3ex*ones(1,k*m),1) + diag(c6ex*ones(1,k*m+1)) + diag(c3ex*ones(1,k*m),-1);
K = kron(iu,Mu)+kron(il,Ml)+kron(eye(k*m+1),Mc);

% PRECONDITIONER CONSTRUCTION
%INTIALIZE PRECONDITIONER
prec=zeros((k*m+1)^2);
for i = 1:k
        for j = 1:k

            %TAKE SUBDOMAIN I,J
            xi = xl{i};
            yj = yl{j};
            [~,cols] = find(xm == xi(1,1));
            [row,~] = find(ym == yj(1,1));
            p = cols(1);
            q = row(1);

            %INITIALIZE PROLONGATION/CONTRACTION MATRIX FOR CURRENT DOMAIN
            lay=size(xi,1)-(m+1);
            Rprolo=sparse((m-1+lay)^2,(k*m+1)^2);

             %BUILD THE PROLONGATION/CONTRACTION MATRIX FOR CURRENT DOMAIN
             rowww=1;
             for iii=1:size(xi,1)
               for jjj=1:size(yj,1)
                  ind = globalmapping_2(p,q,iii,jjj,k,m);
                  Rprolo(rowww,ind)=1;
                  rowww=rowww+1;
               end
             end

            %BUILD LOCAL PRECONDITIONER AND SUM
            Ai=Rprolo*K*Rprolo';
            temp=Rprolo'*(Ai\Rprolo);
            prec=prec+temp; 

        end      
end

%CONSTRUCT THE COARSE GRID OPERATOR
rowww=1;
R0=sparse((k+1)^2,(k*m+1)^2);
for i=1:k+1
    for j=1:k+1
        IND=globalmapping_2(p,q,i,j,k,m);
        R0(rowww,IND)=1;
        rowww=rowww+1;
    end
end

%CONSTRUCT THE COARSE GRID PRECONDITIONER
Ki=R0*K*R0';
prec2=R0'*(Ki\R0)+prec;

%CONSTRUCT THE RHS OF THE SYSTEM
F=f(xm,ym);
for s = 1:k*m+1
    for t = 1:k*m+1
        G = globalindex_glob(s,t,k,m);
        F_m(G) = hm*F(s,t);
    end
end
                
%   SOLVE THE SYSTEM WITH PRECONDITIONED CONJUGATE GRADIENT AND STORE TIME
tic;
u=pcg(K,F_m',1e-3,200);
tstop=toc;
time=double(tstop);

tic;
u_m1=pcg(K,F_m',1e-3,200,prec);
tstop=toc;
time_m1=double(tstop);

tic;
u_m2=pcg(K,F_m',1e-3,200,prec2);
tstop=toc;
time_m2=double(tstop);

% SOLUTION PLOTS
U=zeros(k*m+1,k*m+1);
U_m1=zeros(k*m+1,k*m+1);
U_m2=zeros(k*m+1,k*m+1);
%PUT SOLUTION IN MATRIX FORM
for s = 1:(k*m+1)
    for t =1:(k*m+1)
        pp= globalindex_glob(s,t,k,m);
        U(s,t) = u(pp);
        U_m1(s,t)=u_m1(pp);
        U_m2(s,t)=u_m2(pp);
    end
end
 U(:,1)=0;  U_m1(:,1)=0;  U_m2(:,1)=0; 
 U(:,end)=0;  U_m1(:,end)=0;  U_m2(:,end)=0; 
 U(1,:)=0; U_m1(1,:)=0; U_m2(1,:)=0;
 U(end,:)=0; U_m1(end,:)=0; U_m2(end,:)=0;




%PLOT THE SOLUTION
figure(1)
subplot(1,2,1)
surf(xm,ym,U)
xlabel('$x$', 'Interpreter','latex');
ylabel('$y$', 'Interpreter','latex');
legend('numerical solution $3$D', 'Location','northoutside', 'Interpreter', 'latex');
subplot(1,2,2);
contourf(xm,ym,U);
colorbar("eastoutside");
xlabel('$x$', Interpreter='latex');
ylabel('$y$', Interpreter='latex');
legend('numerical solution - isolines','Location','northoutside');
axis equal;

figure(2)
subplot(1,2,1)
surf(xm,ym,U_m1)
xlabel('$x$', 'Interpreter','latex');
ylabel('$y$', 'Interpreter','latex');
legend('numerical solution $3$D, 1$^st$ preconditioner', 'Location','northoutside', 'Interpreter', 'latex');
subplot(1,2,2);
contourf(xm,ym,U_m1);
colorbar("eastoutside");
xlabel('$x$', Interpreter='latex');
ylabel('$y$', Interpreter='latex');
legend('numerical solution - isolines','Location','northoutside');
axis equal;

figure(3)
subplot(1,2,1)
surf(xm,ym,U_m2)
xlabel('$x$', 'Interpreter','latex');
ylabel('$y$', 'Interpreter','latex');
legend('numerical solution $3$D, 2$^nd$ preconditioner', 'Location','northoutside', 'Interpreter', 'latex');
subplot(1,2,2);
contourf(xm,ym,U_m2);
colorbar("eastoutside");
xlabel('$x$', Interpreter='latex');
ylabel('$y$', Interpreter='latex');
legend('numerical solution - isolines','Location','northoutside');
axis equal;


%MAPPINGS
function r = globalindex_glob(i, j, k, m) 
    r = i + (j-1)*(k*m+1);   
end
function r = globalmapping_2(p,q,i,j,k,m)
    x = p + i - 1;
    y = q + j - 1;
    r = globalindex_glob(x, y, k, m); 
end
end