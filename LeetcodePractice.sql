-- 建立 Loans 資料表 (對應 C# 專案中的 LoanRecord 模型)
CREATE TABLE IF NOT EXISTS loans ( -- 改成小寫
    id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100),
    amount DECIMAL,
    status VARCHAR(20),
    loan_date DATE
);

-- 寫入三筆初始測試資料
INSERT INTO loans (customer_name, amount, status, loan_date)
VALUES
    ('張小明', 500000, 'Approved', '2024-03-01'),
    ('李阿花', 1200000, 'Pending', '2024-03-15'),
    ('王大同', 800000, 'Approved', '2024-03-20');
    ('李奕辰', 813545, 'Approved', '2024-03-20');


-- 1757. Recyclable and Low Fat Products
select product_id
FROM Products 
WHERE low_fats = 'Y' and recyclable  = 'Y'

-- 584. Find Customer Referee
Select name from Customer
where referee_id <> 2
OR referee_id IS NULL

-- 595. Big Countries
SELECT name, population, area 
from World
WHERE area >= 3000000
OR population >= 25000000

-- 1148. Article Views I 
-- 這題必須使用 DISTINCT 或 GROUP BY，否則會因為重複檢視而產生重複 ID。
-- 選 DISTINCT 的時機： 當你只想單純「過濾掉重複的雜訊」，且不涉及任何數值計算時。例如：列出所有有交易紀錄的客戶清單。
-- 選 GROUP BY 的時機： 當你需要計算「每個客戶買了多少錢」或是「某個欄位出現幾次」時。

select DISTINCT author_id as id 
from Views
where author_id = viewer_id     
order by author_id ASC
-- 或是
select author_id as id 
from Views
where author_id = viewer_id     
group by author_id
order by author_id ASC


-- 1683. Invalid Tweets
-- PostgreSQL	LENGTH()	計算字元數（一個中文算 1）。
-- MS SQL	LEN()	計算字元數，但不包含結尾的空白。
-- Oracle	LENGTH()	計算字元數。

select tweet_id 
from Tweets
where LENGTH(content) > 15



