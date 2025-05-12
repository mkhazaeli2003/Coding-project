sets
i /0*5/
k /1*3/
n /1*2/
p /1*10/
alias(i,j,h);

*dade ha az maghale estekhraj shode ast.

Scalars M/1000000000/;
parameters
X(i) /0 15, 1 9, 2 6.34, 3 5.3, 4 14.33, 5 5.5/
Y(i) /0 7.23, 1 11.3, 2 8.29, 3 6.63, 4 6.54, 5 11.1/
ot(i) /0 0,1 6,2 50,3 110,4 210,5 12/
ct(i) /0 720,1 110,2 390,3 490,4 650,5 410/
st(i) /0 0,1 40,2 50,3 35,4 60,5 50/
sl(n) /1 9,2 8/
vc(n) /1 400,2 300/
v_c(k) /1 2.5,2 1,3 5/
v_s(k) /1 0.66, 2 1.2, 3 0.75/
v_e(k) /1 0.133, 2 0.069, 3 0.183/;

table pr(i,n)
   1   2
0  0   0
1  1   1
2  1   1
3  1   0
4  0   1
5  1   1;

binary variable a(i,j,n,p,k);
positive variable at(i,n,p,k);
positive variable z(n,p);
positive variable zmax;
positive variable w(i,n,p,k);
positive variable wprime(i,j,n,p,k);
free variable Zreal;

Equations
Con1
Con2
Con3
Con4
Con5
Con6
Con7
Con8
Con9
Con10
Con11
Con12
Con13
Con14
Con15
Con16
Con17
Obj1;




Obj1 ..Zreal =e= sum((i,j,n,p,k),(vc(n)+ v_c(k)*sqrt((X(i)-X(j))**2-(Y(i)-Y(j))**2))*a(i,j,n,p,k));
Con1 ..sum((j,n,p,k),a('0',j,n,p,k)) =g= 1;
Con2 ..sum((j,n,p,k),a(j,'0',n,p,k)) =g= 1;
Con3(i) ..sum((j,n,p,k) $(ord(j)<>ord(i)),a(i,j,n,p,k)) =e= 1;
Con4(i,j) ..sum((n,p,k) $(ord(i)<>ord(j)),a(i,j,n,p,k)) =l= 1;
Con5(p) ..sum((j,n,k), a('0',j,n,p,k)) =l= 1;
Con6(i,n) ..sum((j,p,k) $(ord(i)<>ord(j)), a(i,j,n,p,k)) =l= pr(i,n);
Con7 ..sum((j,n,p,k),a('0',j,n,p,k)) =e= sum((j,n,p,k),a(j,'0',n,p,k));
Con8(h,n,p,k) ..sum(i $(ord(i)<>ord(h)),a(i,h,n,p,k)) =e= sum(j $(ord(j)<>ord(h)),a(h,j,n,p,k));
Con9(i,j,n,p,k)$(ord(i)<>ord(j)) ..at(i,n,p,k) + w(i,n,p,k) + st(i) + v_s(k)*sqrt((X(i)-X(j))**2-(Y(i)-Y(j))**2) =l= M*(1-a(i,j,n,p,k))+at(j,n,p,k);
Con10(i,n,p,k) ..ot(i) =l= at(i,n,p,k)+w(i,n,p,k);
Con11(i,n,p,k) ..at(i,n,p,k)+w(i,n,p,k)+st(i) =l= ct(i);
Con12(n,p,k) ..at('0',n,p,k) =e= 0;
Con13(n,p) ..z(n,p) =e= sum((i,j,k) $(ord(j)<>ord(i)), (st(i)+ v_s(k)*sqrt((X(i)-X(j))**2-(Y(i)-Y(j))**2))*a(i,j,n,p,k)+wprime(i,j,n,p,k));
Con14(i,j,n,p,k)$(ord(i)<>ord(j)) ..wprime(i,j,n,p,k) =l= w(i,n,p,k);
Con15(i,j,n,p,k)$(ord(i)<>ord(j)) ..wprime(i,j,n,p,k) =l= M*a(i,j,n,p,k);
Con16(i,j,n,p,k)$(ord(i)<>ord(j)) ..w(i,n,p,k)- wprime(i,j,n,p,k) =l= M *(1-a(i,j,n,p,k));
Con17(n,p) ..z(n,p) =l= zmax;



model housecare /all/;
Solve housecare using mip min Zreal;
Display Zreal.l, a.l, zmax.l, z.l, w.l, at.l;



