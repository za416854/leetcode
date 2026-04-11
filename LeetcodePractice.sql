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

-- 1378. Replace Employee ID With The Unique Identifier
-- Oracle 在 FROM 與 JOIN 子句後的「表別名」不能加 AS（但欄位別名可以加 AS）。
-- LEFT JOIN (左外連接)： 以「左表」為主。無論右表有沒有對應的資料，左表的每一列都會出現。如果右表沒資料，就填 NULL。(左表資料全留)
-- RIGHT JOIN (右外連接)： 以「右表」為主。無論左表有沒有對應的資料，右表的每一列都會出現。如果左表沒資料，就填 NULL。(右表資料全留)

SELECT uni.unique_id , es.name from Employees es
LEFT JOIN EmployeeUNI uni on es.id = uni.id


-- 1068. Product Sales Analysis I
-- 為什麼這次用 INNER JOIN 也可以？ 因為你要的是「兩個都有」的交集。
-- 雖然你在上一題學會了 LEFT JOIN（保證左表不消失），但在這題： 
-- 主體是 Sales：我們要的是「每一筆銷售 (sale_id)」。 
-- 關係完整：因為有外鍵約束，不會出現「有銷售紀錄但沒產品名稱」的情況。 
-- 結論：使用 INNER JOIN 或 LEFT JOIN 結果會是一樣的，但實務上 INNER JOIN 效能通常更好。

SELECT product_name, year, price FROM Sales s
inner join Product p on s.product_id = p.product_id 


-- 1581. Customer Who Visited but Did Not Make Any Transactions
-- 當你使用 GROUP BY 時，SELECT 出來的欄位要麼是「分組的依據」，要麼是「計算的結果（如加總、計數）」。
-- NOT EXISTS 通常會比 NOT IN 快，特別是當 Transactions 資料表很大時。
-- 這題可以多練因為有很多觀念
SELECT customer_id, 
		count(v.visit_id) as count_no_trans 
FROM Visits v
LEFT JOIN Transactions t on v.visit_id = t.visit_id 
-- WHERE t.transaction_id is NULL
-- WHERE v.visit_id not in (select visit_id from Transactions)
WHERE not exists (select 1 from Transactions t where t.visit_id = v.visit_id )
GROUP BY v.customer_id

-- 197. Rising Temperature
select w1.id from weather w1 
join weather w2 on w1.recordDate - w2.recordDate = 1
where w1.temperature > w2.temperature 




-- 補充:
-- CONCAT 就是把 「多個字串黏在一起」 的膠水。 
-- 在 SQL 中，如果你有「姓」跟「名」兩個欄位，但你想顯示「全名」，這時候就要出動 CONCAT。
-- 假設你有這張表：
-- firstName,  lastName
-- 趙,         雷
-- 錢,         電
-- 執行 SQL：
-- SELECT CONCAT(firstName, lastName) AS fullName FROM Student;
-- fullName
-- 趙雷
-- 錢電

-- 1661. Average Time of Process per Machine
-- on 的作用是甚麼，跟where 差在哪裡
-- 在資料庫的執行順序（Execution Order）中，ON 發生在 WHERE 之前。
-- FROM + JOIN: 先確定要抓哪些表。
-- ON: 根據條件，把兩張表「橫向綑綁」在一起，產生一張暫時的大表。
-- WHERE: 對這張「已經捆好的大表」進行過濾，把不符合條件的整行刪掉。 
select a1.machine_id, ROUND(AVG(a2.timestamp - a1.timestamp), 3) as processing_time from Activity a1
join Activity a2
on a1.machine_id = a2.machine_id 
AND a1.process_id  = a2.process_id  
AND a1.activity_type  = 'start'
AND a2.activity_type  = 'end'
GROUP BY a1.machine_id


-- 577. Employee Bonus
select e.name, b.bonus  from Employee e
left join Bonus b on e.empId = b.empId
where b.bonus < 1000 or  b.bonus is null

-- 1280. Students and Examinations
-- CROSS JOIN（交叉連接）。這題最難的地方在於：即使學生「沒有參加過」某門考試，結果也要顯示該科目為 0。
-- CROSS JOIN（暴力配對）先把「所有學生」跟「所有科目」強行配對。
-- 假設有 3 個學生，3 個科目。 CROSS JOIN 會產生 3 * 3 = 9 行資料。這就是我們的「點名簿」，確保每個人在每一科都有一個位置。
-- 第二步：LEFT JOIN（填入成績）
-- 拿這張 9 行的「點名簿」去跟 Examinations 表做 LEFT JOIN。 
-- 有對到的，代表有考試，我們就算次數。 
-- 沒對到的，代表沒考試，次數就是 0。

-- 1. GROUP BY 三個欄位的意義在哪？ 
-- 簡單來說：GROUP BY 的目的是為了決定你的「小計（Summary）要算到多細」。 
-- 在這題中，我們希望最後的報表長這樣： 
-- 小明 - 數學 - 3次 
-- 小明 - 英文 - 2次 
-- 小華 - 數學 - 1次 
-- 為什麼一定要這三個？ 
-- student_id & subject_name：這是邏輯上的核心。如果你只 GROUP BY student_id，電腦會把小明所有的考試（不分科目）全部加在一起，變成「小明 - 5次」，這樣科目資訊就丟失了。所以我們必須按「人 + 科目」這對組合來打包。 
-- student_name：這是語法上的強制要求。雖然我們知道一個 ID 只會對應一個 name，但對電腦來說，name 跟 ID 是不同的欄位。 
-- SQL 鐵律：在 SELECT 裡面出現的欄位，如果它沒有被包在 COUNT()、SUM() 之類的函數裡，它就必須出現在 GROUP BY 中。  
-- 如果不寫 student_name，電腦會報錯：「我按 ID 打包好了，但這包裡面雖然名字都叫小明，你沒叫我按名字分組，我不敢隨便把它顯示出來。」

select 
s.student_id, 
s.student_name, 
sub.subject_name, 
count(e.student_id) as attended_exams 
from Students s
cross join Subjects sub -- on sub.subject_name = e.subject_name 
left join Examinations e 
on s.student_id = e.student_id 
and sub.subject_name = e.subject_name 
group by s.student_id, s.student_name, sub.subject_name
order by s.student_id, sub.subject_name

-- 620. Not Boring Movies
select id, movie, description, rating 
from Cinema 
where MOD(id, 2) = 1
and description <> 'boring'
order by rating desc






-- 1251. Average Selling Price
-- ON 的本質：相親的條件
-- 想像 LEFT JOIN 是一場相親大會： 
-- 左表 (Prices) 是「價格時段」。 
-- 右表 (UnitsSold) 是「賣出的紀錄」。 
-- 如果你只寫 ON p.product_id = u.product_id： 
-- 條件是：「只要產品 ID 一樣就配對。」
-- 結果：一個賣出紀錄會跟所有該產品的時段（不論日期）通通握手。如果產品 A 有 3 個時段，這筆銷售就會被重複算 3 次。 
-- 如果你加上 AND u.purchase_date BETWEEN ...： 
-- 條件變成：「產品 ID 要一樣，而且銷售日期必須在時段內。」
-- 結果：這筆紀錄會去檢查所有的時段，最後只有一個時段會符合條件。這就達到了你感覺到的「不重複」效果。
select p.product_id, NVL(ROUND(sum(p.price * u.units) / sum(u.units), 2), 0) as average_price 
FROM Prices p
LEFT JOIN UnitsSold u 
on p.product_id = u.product_id 
and u.purchase_date between p.start_date and p.end_date
group by p.product_id 

-- 1075. Project Employees I
SELECT e1.employee_id 
FROM Employees e1 
WHERE e1.salary < 30000
  AND e1.manager_id IS NOT NULL
    --  NOT EXISTS (select 1 from Employees e2 where e2.manager_id = e1.employee_id)，這個寫法是在找：
    --  關鍵修正：檢查「我的主管 ID」是否存在於「員工 ID」欄位中
    -- 「哪些員工沒有任何下屬（不是別人的主管）」，而不是題目要的「主管離職了」。
    --  結果： 你會抓到所有**「基層員工」**（因為沒人把他們當主管），但這跟主管有沒有離職完全無關。
  AND NOT EXISTS (
      SELECT 1 
      FROM Employees e2 
      WHERE e2.employee_id = e1.manager_id 
  )
ORDER BY e1.employee_id;




