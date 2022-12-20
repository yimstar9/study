--1. 다음의 두줄을 주석문 처리하시오.
/*SQL 활용
임성구*/
SET AUTOCOMMIT ON;
show autocommit;

-----------------------------------------------------------------------
--2. EMP테이블에서 급여를 기준으로 내림차순으로 사원 정보를 조회하여 
--정렬하고 급여가 같으면 다시 이름기준으로 내림차순으로 조회하시오.

select * from emp
order by sal desc, ename desc;

-----------------------------------------------------------------------
--3. DEPT테이블의 다음 새로운 부서 정보를 저장하고 전체 레코드를 검색하시오.
/*
부서정보:
DEPTNO: 50
DNAME: Planning
LOC: Seoul.
*/
insert into dept(deptno, dname, loc)
values(50, 'Planning', 'Seoul');
select * from dept;
/*
DEPTNO DNAME LOC
------------------------
10	ACCOUNTING	NEW YORK
20	RESEARCH	DALLAS
30	SALES	CHICAGO
40	OPERATIONS	BOSTON
50	Planning	Seoul
*/

-----------------------------------------------------------------------
--4. EMP테이블에서 부서번호로 그룹화하여 월급의 최고액과 
--최저액을 받는 사원의 정보를 조회하시오
select * from emp
where sal in (select max(sal) from emp group by deptno);

/*
EMPNO ENAME JOB MGR HIREDATE SAL COMM DEPTNO
----------------------------------------------------------
7698	BLAKE	MANAGER	7839	81/05/01	2850		30
7839	KING	PRESIDENT		81/11/17	5000		10
7902	FORD	ANALYST	7566	81/12/03	3000		20
*/

select * from emp
where sal in (select min(sal) from emp group by deptno);
/*
EMPNO ENAME JOB MGR HIREDATE SAL COMM DEPTNO
----------------------------------------------------------
7369	SMITH	CLERK	7902	80/12/17	800	    	20
7900	JAMES	CLERK	7698	81/12/03	950	    	30
7934	MILLER	CLERK	7782	82/01/23	1300    	10
*/

-----------------------------------------------------------------------
--5. EMP테이블에서 부서번호로 그룹화하고 급여가 3800이상인 사원번호와 급여를 조회하시오.
select empno, sal from emp
where sal in (select max(sal) from emp group by deptno having max(sal) >= 3800);

/*
EMPNO SAL
------------
7839	5000
*/

-----------------------------------------------------------------------
--6. EMP테이블과 DEPT테이블을 조인시켜 사원의 사원번호, 사원명, 부서번호, 부서명을 조회하시오
select empno, ename, dept.deptno, dept.dname
from emp, dept
where emp.deptno = dept.deptno;
/*
EMPNO ENAME DEPTNO DNAME
------------------------
7369	SMITH	20	RESEARCH
7499	ALLEN	30	SALES
7521	WARD	30	SALES
7566	JONES	20	RESEARCH
7654	MARTIN	30	SALES
7698	BLAKE	30	SALES
7782	CLARK	10	ACCOUNTING
7839	KING	10	ACCOUNTING
7844	TURNER	30	SALES
7900	JAMES	30	SALES
7902	FORD	20	RESEARCH
7934	MILLER	10	ACCOUNTING
*/

-----------------------------------------------------------------------
--7. 단일행 서브쿼리를 이용하여 EMP테이블에서 SMITH의 부서명을 호출하여 조회하시오.
select dname from dept
where deptno = (select deptno from emp where ename = 'SMITH');
/*
DNAME
--------
RESEARCH
*/

-----------------------------------------------------------------------
--8. 단일행 서브쿼리를 이용하여 EMP테이블에서 사원번호가 7654인 사원의 급여보다 
--급여가 적은 사원의 사원번호, 사원명, 직급, 급여 순으로 조회하시오.
select empno, ename, job, sal from emp
where sal < (select sal from emp where empno = 7654);
/*
EMPNO ENAME JOB SAL
---------------------------
7369	SMITH	CLERK	800
7900	JAMES	CLERK	950
*/

-----------------------------------------------------------------------
--9. 단일행 서브쿼리에 그룹함수를 사용하여 EMP테이블에서 부서번호가 20인 부서의
--최소 급여보다 많은 부서를 조회하시오.
select deptno, min(sal) from emp
group by deptno
having min(sal) > (select min(sal) from emp where deptno = 20);

/*
DEPTNO MIN(SAL)
--------------
30	950
10	1300
*/

-----------------------------------------------------------------------
--10. 다중행 서브쿼리를 이용하여 EMP테이블에서 부서번호 30에 소속된 사원 중 최고 
--급여보다 많은 급여를 받는 사원을 조회하시오.
select * from emp
where sal > all (select sal from emp where deptno = 30);

/*
EMPNO ENAME JOB MGR HIREDATE SAL COMM DEPTNO
----------------------------------------------------------
7566	JONES	MANAGER	7839	81/04/02	2975		20
7902	FORD	ANALYST	7566	81/12/03	3000		20
7839	KING	PRESIDENT		81/11/17	5000		10
*/