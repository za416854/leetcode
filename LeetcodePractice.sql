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


 