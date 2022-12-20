
--1. 다음의 두줄을 주석문 처리하시오.
/*데이터베이스 요구사항 분석
임성구*/

--2. 상품정보(GoodsInfo)테이블 생성하시오.(SQLPLUS 또는 SQL Developer이용)
SET AUTOCOMMIT ON;
show autocommit;
create table GoodsInfo(
             proCode char(5) primary key,
             proName varchar2(30) not null,
             price number(8) not null,
             maker varchar(25) not null);


select proCode 상품코드,
       proName 상품명,
       price 가격,
       maker 제조사
from GoodsInfo;

--3. 생성된 GoodsInfo테이블에 레코드를 추가하시오.
insert into GoodsInfo values('1001','냉장고',1800000,'SM');
insert into GoodsInfo values('1002','세탁기',550000,'LN');
insert into GoodsInfo values('1003','HDTV',280000,'HP');
insert into GoodsInfo values('1004','전자레인지',230000,'SM');
insert into GoodsInfo values('1005','오디오',770000,'LN');
insert into GoodsInfo values('1006','PC',880000,'HP');

--4. 전체 레코드를 검색하시오.
select * from GoodsInfo;

--5. GoodsInfo테이블의 구조를 확인하시오.
desc GoodsInfo;

--6. 상품정보(GoodsInfo)테이블에서 모든 상품의 가격 합을 구하시오.
select sum(price) from GoodsInfo;

--7. GoodsInfo테이블에 할인가(salePrice) 컬럼을 추가하시오
alter table GoodsInfo
add (salePrice number(8));

--8. GoodsInfo테이블에서 가격이 70만원 이상인 상품만 조회하여 상품명과 가격만 display하시오
select proName 상품명, price 가격 from GoodsInfo
where price>=700000;

--9. GoodsInfo테이블에서 상품코드가 1002와 1003를 제외한 상품의 상품명과 가격을 display하시오.
select proName 상품명, price 가격 from GoodsInfo
where proCode not in(1002,1003);

--10. GoodsInfo테이블에서 상품명이 PC인 상품의 정보를 조회하시오.
select proCode 상품코드, proName 상품명, price 가격, maker 제조사 from GoodsInfo
where proName='PC';