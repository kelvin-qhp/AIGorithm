### 1.  RFI count

```
SELECT *-- count(1)--event_name,amount -- count(1)
FROM "etrack_source"
where 1=1
and event_name in('RSD.S' ,'RSD') 
--and big_page = 'SR'
and request_time >= '2022/12/15'
  and request_time < '2022/12/16'
-- 	and anonymous_id= '487914380'
-- 	and ip = '207.161.246.29'
-- 	and current_url = 'https://www.globalsources.com/searchList/products?keyWord=leather%20sectional&pageNum=2'
-- 	and user_id ='1306812827750'
-- 	and product_id = '1174458123'
-- 	and keyword_query = 'usb3.0 flat'

order by request_time desc
```

daily count:

```
SELECT to_char(request_time,'yyyy-mm-dd') as DD, count(1)--event_name,amount -- count(1)
FROM "etrack_source"
where 1=1
and event_name in('RSD.S' ,'RSD') 
--and big_page = 'SR'
and request_time >= '2023/05/01'
  and request_time < '2023/05/16'
-- 	and anonymous_id= '487914380'
-- 	and ip = '207.161.246.29'
-- 	and current_url = 'https://www.globalsources.com/searchList/products?keyWord=leather%20sectional&pageNum=2'
-- 	and user_id ='1306812827750'
-- 	and product_id = '1174458123'
-- 	and keyword_query = 'usb3.0 flat'
GROUP BY to_char(request_time,'yyyy-mm-dd')
order by DD desc
```





### 2. RFI in DB

```

SELECT * --count(1) -- count(distinct(session_id)) -- inquiry_id
 FROM rfi_grp.inquire_all
WHERE 
1=1 --inquiry_type in('PRODUCT_UPSELL','PRODUCT','CATEGORY') 
-- and buyer_id = '1306815300975'
-- and upsell_flag = 't'  
and  create_date >= to_date('2022/12/15','yyyy/mm/dd')
and  create_date < to_date('2022/12/16','yyyy/mm/dd')
and rfi_source = '1'
--  and inquiry_id in( '8005001127014')
ORDER BY create_date desc



SELECT  to_char(create_date,'yyyy/MM/dd') as dt,count(1)
 FROM rfi_grp.inquire_all
WHERE 
1=1 --inquiry_type in('PRODUCT_UPSELL','PRODUCT','CATEGORY') 
-- and buyer_id = '1306815300975'
-- and upsell_flag = 't'  
and  create_date >= to_date('2023/07/15','yyyy/mm/dd')
and  create_date < to_date('2023/07/17','yyyy/mm/dd')
and rfi_source = '1'
--  and inquiry_id in( '8005001127014')
GROUP BY to_char(create_date,'yyyy/MM/dd')
ORDER BY dt desc
```

### 3. PV in eTrack

```
SELECT count(1) -- product_id, count(1) as cnt , count(DISTINCT(anonymous_id)) as UV--event_name,amount -- count(1)
FROM "etrack_source"
where 1=1
and event_type='ie' and event_name='PG'
-- and big_page = 'TS'
-- and tag in('DT') 
--and big_page = 'SR'
   and request_time >= '2022/12/14'
  and request_time < '2022/12/15'
  and user_id ='1306816858651'
order by request_time desc

```

### 4. PV in Sensor

```
select event,user_id,time,$url,channel_traffic_last_touch ,channel_traffic_last_touch ,$referrer-- * --rfiid
from events
where 1=1 --event in ('$pageview')
and event in ('$pageview')
-- and distinct_id = '17f3379346813e-0ab86eb0011fad-5e133e12-2073600-17f337934695ef'
--and distinct_id='1306816385378'
and  date>= '2022-12-14' and date <  '2022-12-15'
and distinct_id= '1306816858651'
-- and 	user_id = -5854867417783537953
-- and rfiid = '8005001170957'

and cast(channel_traffic_last_touch as int) between 5001001 and 5999999

order by time desc
```

```

```









### 5. User first page by Day

```

	select * from 
	(
	SELECT
		anonymous_id,request_time,event_type,event_name,big_page,sub_page,module,utm_source,current_url,land_url,"row_number"() OVER (partition by to_char(request_time,'yyyy/mm/dd'),anonymous_id ORDER BY request_time) rank
	FROM etrack_source es
	where 1=1
-- 	 and event_name in('RSD.S' ,'RSD') 
	-- and event_type='ie'
	--  and event_name = 'REG'
	and request_time >= '2023/03/01'
	and request_time < '2023/03/08' 
	
	and  EXISTS
		(
		SELECT to_char(request_time,'yyyy/mm/dd'), anonymous_id
		FROM "etrack_source"
		where 1=1
		 and event_name in('RSD.S' ,'RSD') 
		-- and event_type='ie'
		--  and event_name = 'REG'
		and request_time >= '2023/03/01'
		and request_time < '2023/03/08' 
		and es.anonymous_id =anonymous_id
		and to_char(es.request_time,'yyyy/mm/dd') =to_char(request_time,'yyyy/mm/dd')
		GROUP BY to_char(request_time,'yyyy/mm/dd'),anonymous_id

		)
	
	ORDER BY request_time
	) t where rank = 1

```

### 6. Grafana

```
SELECT
  $__timeGroupAlias(request_time,'10m',0),
  count(1) as cnt
FROM etrack_source
WHERE 1=1
and event_name in('RSD.S' ,'RSD') 
-- and request_time >= '2023/05/13'
and  $__timeFilter(request_time)
GROUP BY  1 
order by 1 desc


SELECT
  $__timeGroupAlias(request_time,'1d',0),
  count(1) as cnt
FROM etrack_source
WHERE 1=1
and event_name in('RSD.S' ,'RSD') 
and request_time >=  to_date('${__from:date}', 'YYYY/MM/dd') 
-- and request_time <  ${__to:date}
GROUP BY  1 
order by 1 desc

```

```
SELECT
  to_date(request_time::text, 'yyyy-MM-dd') as dd,
  count(1) as cnt
FROM etrack_source
WHERE 1=1
and event_name in('RSD.S' ,'RSD') 
-- and request_time >= '2023/05/13'
and $__timeFilter(request_time)
GROUP BY  to_date(request_time::text, 'yyyy-MM-dd')
order by dd desc
```

```



select dd,
sum(CASE WHEN bp= 'SR' THEN cnt ELSE 0 END ) as SR,
sum(CASE WHEN bp= 'CAT' THEN cnt ELSE 0 END ) as CAT,
sum(CASE WHEN bp= 'HP' THEN cnt ELSE 0 END ) as HP,
sum(CASE WHEN bp= 'SEM' THEN cnt ELSE 0 END ) as SEM,
sum(CASE WHEN bp= 'SEO' THEN cnt ELSE 0 END ) as SEO,
sum(CASE WHEN bp= 'SHP' THEN cnt ELSE 0 END ) as SHP,
sum(CASE WHEN bp= 'CNT' THEN cnt ELSE 0 END ) as CNT,
sum(CASE WHEN bp is null THEN cnt ELSE 0 END ) as other,
sum(cnt)
from
(
SELECT to_char(request_time, 'YYYY/MM/dd') as dd, big_page as bp, count(1) as cnt
FROM etrack_source
WHERE 1=1
and event_name in('RSD.S' ,'RSD') 
and request_time >= '2023/05/13'
  and request_time < '2023/05/16'
GROUP BY to_char(request_time, 'YYYY/MM/dd') ,big_page
) t
GROUP BY dd

```

### 7. User

~~~
select * from user_edm_preference uep 
select * from user_info ui where user_id  
select * from user_extend 

--9291755
-- 885166
--1666644
~~~

### 8. DB query for paid supplier cnt

~~~
select count(DISTINCT vs.supplier_id)
FROM supplier_grp.vs_header vs
LEFT JOIN supplier_grp.online_section os ON to_number(concat (200, vs.content_org_id), '9999999999999999999999') = os.org_id
WHERE os.website_type = 'GSOL'
  AND os.section_code = 'MAIN'
  AND os.target_website_status = 'Online'
  AND ((vs.website_type = 'GSOL'
  AND vs.SHOWROOM_TYPE IN ('CDWIMGSTAR1', 'CDWIMGSTAR2', 'CDWIMGSTAR3', 'CDWIMGSTAR4', 'CDWIMGSTAR5', 'CDWIMGSTAR6', 'CDWIMGSPC', 'CDWIMGNEWPRD')))
  AND vs.showroom_status = 'ACT'
  AND vs.web_start_date <= now() :: TIMESTAMP + '1 day'
  AND COALESCE (vs.extension_date, vs.web_end_date)>= now() :: TIMESTAMP
~~~



### 9. Query Feature supplier cnt

~~~
 select count(1)  from isearch_grp.feature_supplier fp  where delete_flag = false
~~~

### 10.  Inconsistant data between Feature table and source DB data(contract table)

~~~
 select supplier_id  from isearch_grp.feature_supplier fp  where delete_flag = false
   and not exists(
   select 1
   FROM supplier_grp.vs_header vs
LEFT JOIN supplier_grp.online_section os ON to_number(concat (200, vs.content_org_id), '9999999999999999999999') = os.org_id
WHERE os.website_type = 'GSOL'
  AND os.section_code = 'MAIN'
  AND os.target_website_status = 'Online'
  AND vs.website_type = 'GSOL'
  AND vs.SHOWROOM_TYPE IN ('CDWIMGSTAR1', 'CDWIMGSTAR2', 'CDWIMGSTAR3', 'CDWIMGSTAR4', 'CDWIMGSTAR5', 'CDWIMGSTAR6', 'CDWIMGSPC', 'CDWIMGNEWPRD')
  AND vs.showroom_status = 'ACT'
  AND vs.web_start_date <= now() :: TIMESTAMP + '1 day'
  AND COALESCE (vs.extension_date, vs.web_end_date)>= now() :: TIMESTAMP 
  and vs.supplier_id = fp.supplier_id
  )
  
  ----------------------------------------------------------------------------------
  
   select supplier_id  from isearch_grp.feature_supplier fp  where delete_flag = false
   and not exists(
   select 1
   FROM supplier_grp.vs_header vs
LEFT JOIN supplier_grp.online_section os ON to_number(concat (200, vs.content_org_id), '9999999999999999999999') = os.org_id
WHERE os.website_type = 'GSOL'
  AND os.section_code = 'MAIN'
  AND os.target_website_status = 'Online'
  AND vs.website_type = 'GSOL'
  AND vs.SHOWROOM_TYPE IN ('CDWIMGSTAR1', 'CDWIMGSTAR2', 'CDWIMGSTAR3', 'CDWIMGSTAR4', 'CDWIMGSTAR5', 'CDWIMGSTAR6', 'CDWIMGSPC', 'CDWIMGNEWPRD')
  AND vs.showroom_status = 'ACT'
  AND vs.web_start_date <= now() :: TIMESTAMP + '1 day'
  AND COALESCE (vs.extension_date, vs.web_end_date)>= now() :: TIMESTAMP 
  and vs.supplier_id = fp.supplier_id
  )
  
  
~~~



### 11. Online Prouct count in source table:

~~~
  select count(1)
   from product_grp.online_product op 
   where  op.website_type = 'GSOL' AND op.target_website_status = 'Online'
   and op.org_id in (
   select supplier_id from  isearch_grp.feature_supplier fp  where  delete_flag = false
   )
   
   --2666110
~~~



###  12. Product cnt in feature product table:

~~~
   	select  count(1) from  isearch_grp.feature_product fp,isearch_grp.feature_supplier fs where fp.supplier_id = fs.supplier_id  and fs.delete_flag = false and fp.delete_flag = false   	
   	--2666032
~~~



### 13. data diff between source table and feature product table:

~~~
select * --count(1)
from product_grp.online_product op 
where  op.website_type = 'GSOL' AND op.target_website_status = 'Online'
and op.org_id in (
select supplier_id from  isearch_grp.feature_supplier fp  where  delete_flag = false
)
and op.product_id not in (
    select fp.product_id from  isearch_grp.feature_product fp,isearch_grp.feature_supplier fs where fp.supplier_id = fs.supplier_id  and fs.delete_flag = false and fp.delete_flag = false   	
)
~~~

### 14. check exception data SQL:

~~~
--1. check feature supplier
	select * from  isearch_grp.feature_supplier fp  where  supplier_id in(2003002353801) and delete_flag = false
		
--2. check contract:	
    SELECT *
    FROM supplier_grp."vs_header"
    WHERE showroom_status = 'ACT'
    AND showroom_type IN 	('CDWIMGNEWPRD','CDWIMGSTAR1','CDWIMGSTAR2','CDWIMGSTAR3','CDWIMGSTAR4','CDWIMGSTAR5','CDWIMGSTAR6','AGGREGATE')
    AND web_start_date <= now() + interval '1 day'
    --AND COALESCE (extension_date, web_end_date) >= now() :: TIMESTAMP
    and supplier_id in (2002100143516)
    order by mdate desc 
    
 --3. check online_product:
     SELECT product_id,org_id 
     FROM product_grp.online_product op 
     WHERE website_type = 'GSOL' AND target_website_status = 'Online' 
     and op.org_id  in( 2002100143516)
~~~

### 15.  Get supplier by SP level

~~~
SELECT SHOWROOM_TYPE,count(1) 
FROM (
SELECT
CONTENT_ORG_ID,
SHOWROOM_TYPE,
CONTRACT_CODE,
GROUP_TYPE,
GROUP_CODE,
SUPPLIER_ID,
ROW_NUMBER() OVER(PARTITION BY CONTENT_ORG_ID ORDER BY CONTRACT_CODE DESC NULLS LAST)
FROM
supplier_grp.VS_HEADER VH
LEFT JOIN supplier_grp.online_section os ON to_number(concat (200, VH.content_org_id), '9999999999999999999999') = os.org_id
WHERE 
VH.website_type = 'GSOL'
AND VH.SHOWROOM_STATUS = 'ACT'
and os.website_type = 'GSOL'
 AND os.section_code = 'MAIN'
AND VH.SHOWROOM_TYPE IN ('CDWIMGNEWPRD','CDWIMGSTAR1','CDWIMGSTAR2','CDWIMGSTAR3','CDWIMGSTAR4','CDWIMGSTAR5','CDWIMGSTAR6','AGGREGATE')
AND VH.WEB_START_DATE <=  CURRENT_DATE + 1
AND COALESCE (VH.extension_date, VH.web_end_date) > CURRENT_DATE
--and VH.supplier_id IN 
--(
--2003002328217,
--2008852621682,
--2003002306929,
--2003002320783,
--2003002321981
--)
) t
WHERE t.row_number = 1
--   AND t.SHOWROOM_TYPE IN ('CDWIMGSTAR3')
   group by SHOWROOM_TYPE
~~~

## 2.1 Get paid online product

~~~
1. By vs_header

select count(1)  from product_grp.online_product op where target_website_status ='Online' and website_type ='GSOL'
and org_id not in(
SELECT distinct org_id  FROM supplier_grp."vs_header"
        WHERE showroom_status = 'ACT'
          AND web_start_date <= now() + interval '1 day'
          AND extension_date > now()
          AND showroom_type IN ('CDWIMGNEWPRD','CDWIMGSTAR1','CDWIMGSTAR2','CDWIMGSTAR3','CDWIMGSTAR4','CDWIMGSTAR5','CDWIMGSTAR6')

--        AND showroom_type = 'AGGREGATE'
         )
         
2. By Feature table:
select count(1)  from isearch_grp.feature_product fp 
where delete_flag = false  



~~~

```

```

