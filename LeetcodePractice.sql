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
    -- 為什麼 1 號 (Kalel) 「不會」出現？
    -- 外層 (e1)：現在輪到 1 號 Kalel。他的 manager_id 是 11。 
    -- 內層 (e2)：電腦拿著 11 去 e2.employee_id 找。 
    -- 結果：電腦找到了 Joziah (11號)！所以 EXISTS (存在) 是 True。
    -- NOT EXISTS 就是FALSE，所以也就是 另外一個e1.主管id不在，e2.員工id卻在的11號
      SELECT 1 
      FROM Employees e2 
      WHERE e2.employee_id = e1.manager_id 
  )
ORDER BY e1.employee_id;


-- 2356. Number of Unique Subjects Taught by Each Teacher
-- Oracle：支援 COUNT(UNIQUE column)，它在 Oracle 裡跟 COUNT(DISTINCT ...) 是完全等價的（同義詞）。但為了代碼的可移植性，資深工程師通常還是會寫 DISTINCT。
-- 因為MySQL (LeetCode 預設) / PostgreSQL / MS SQL 不支援 UNIQUE，只支援 DISTINCT
SELECT teacher_id, count(DISTINCT subject_id) as cnt 
FROM Teacher
group by teacher_id

-- 1731. The Number of Employees Which Report to Each Employee
-- 解題邏輯：經理與下屬的對碰
-- 我們要再次把 Employees 表想像成兩張： 
--   - mgr (經理表)：我們想從這裡拿經理的 employee_id 和 name。 
--   - emp (下屬表)：我們想從這裡算有多少人，以及他們的平均 age。 
-- 連線條件：
-- 當下屬的 reports_to 等於 經理的 employee_id 時，這條線就連起來了。
SELECT mgr.employee_id, mgr.name, COUNT(emp.reports_to ) as reports_count, ROUND(AVG(emp.age)) as average_age
FROM Employees mgr
-- 這裡用 JOIN (也就是 INNER JOIN) 即可，因為題目定義「經理」是至少有一名下屬報告的人。沒有下屬的人，在連線時就會被自動刷掉。
INNER JOIN Employees emp on mgr.employee_id = emp.reports_to
GROUP BY mgr.employee_id, mgr.name
order by mgr.employee_id


-- 596. Classes With at Least 5 Students
SELECT class   
FROM Courses 
GROUP BY class
HAVING COUNT( class ) >= 5
-- Q: 所以HAVING也可以說是
-- 我要用聚合函數算欄位但不要SELECT 這個東西
-- 我就可以用HAVING去做?
-- GEMINI:
-- 1. 沒錯！「算而不秀」是 HAVING 的絕活
-- 在 SQL 的世界裡，SELECT 決定的是「最後誰要出現在螢幕上」，而 HAVING 決定的是「誰有資格留在結果清單裡」。這兩者是完全獨立的。

-- 以這題 596. Classes With at Least 5 Students 為例：
-- 題目只要我們回傳 class 名稱，不需要知道具體有幾個人。
-- 電腦在後台做的： 默默地把每個班級的人數算出來（例如：數學班 6 人、英文班 3 人）。 
-- HAVING 做的： 看到數學班 >= 5，給過；看到英文班 < 5，踢掉。 
-- SELECT 做的： 只把合格的 class 名字印出來，那個人數（6 或 3）雖然算過了，但因為你沒在 SELECT 寫它，它就不會出現。 
-- 2. 跟 WHERE 的「權限」差別 
-- 你可能會問：那為什麼 WHERE 不能這樣玩？ 
-- 因為 SQL 的執行順序是： 
-- WHERE：這時候電腦還在看「每一行」原始資料，它根本還沒開始「打包」，所以它不知道「這班到底有幾個人」。 
-- GROUP BY：這時候才開始打包。  
-- HAVING：打包完了，這時候電腦才知道每包的重量（COUNT, SUM 等），所以它可以根據這些重量來過濾。 
-- 白話比喻： 
--   - WHERE：像是在百貨公司門口檢查，沒帶會員卡的人不准進去（針對個人）。 
--   - HAVING：像是在百貨公司裡面檢查，消費總額沒滿 5000 的團體不准換贈品（針對分組後的結果）。



-- 補充HAVING：WHERE 是在分組前過濾「每一行」；HAVING 是在分組後過濾「那一組」。
-- 為什麼需要 HAVING？（跟 WHERE 的大決鬥）
-- 想像你正在統計學校的班級人數。 
--   - WHERE：你可以在統計前說：「我不要看補習班的學生（過濾原始資料）。」 
--   - HAVING：你不能在統計前說：「我只要看人數大於 30 人的班級。」 
-- 為什麼？因為在還沒「數」完之前，電腦根本不知道哪個班級有幾個人。所以你必須先 GROUP BY 算完 COUNT，最後再用 HAVING 把不達標的班級踢掉。
-- 特性,      WHERE,                    HAVING
-- 執行時機,  GROUP BY 之前,            GROUP BY 之後
-- 作用對象,  原始資料的「每一列」,       分組後的「統計結果」
-- 聚合函數,  "不可搭配 COUNT, SUM 等", "專門搭配 COUNT, SUM 等"

-- 1667. Fix Names in a Table
SELECT 
user_id, 
CONCAT(
  UPPER(SUBSTR(name, 1,1)), 
  LOWER(SUBSTR(name, 2, LENGTH(name))) 
  ) as name
FROM Users 
ORDER BY user_id

-- 解題邏輯：字串的外科手術
-- 想像我們要處理的名字是 aLICE： 
-- 切下第一個字：拿掉 a，用 UPPER() 變成 A。 
-- 切下剩下的字：拿掉 LICE，用 LOWER() 變成 lice。 
-- 縫合起來：用 CONCAT() 把 A 和 lice 拼成 Alice。

-- 三大資料庫的字串函數對照
-- 功能,         MySQL/Oracle,       PostgreSQL,             MS SQL
-- 拼接字串,     "CONCAT(a, b)",     a || b 或 CONCAT,       a + b 或 CONCAT
-- 切片 (第1字), "SUBSTR(n, 1, 1)",  "SUBSTRING(n, 1, 1)",   "SUBSTRING(n, 1, 1)"
-- 切片 (剩餘),  "SUBSTR(n, 2)",     "SUBSTRING(n, 2)",      "SUBSTRING(n, 2, LEN(n))"
-- 字串長度,     LENGTH(),           LENGTH(),               LEN()

-- 1527. Patients With a Condition
SELECT patient_id, patient_name, conditions
FROM Patients 
WHERE conditions like '% DIAB1%'
OR conditions like 'DIAB1%'

SELECT patient_id, patient_name, conditions
FROM Patients 
WHERE REGEXP_LIKE(conditions ,'(^| )DIAB1')
-- Reg-ex (regular expression) 快速上手：五分鐘掌握 80% 的場景
-- Regex 看起來像亂碼，是因為它把「位置」和「內容」都用符號代替了。我們把它拆成三個部分來記：
-- 位置與邊界（在哪裡找？）
-- ^：字串的開頭，強制規定文字必須從這開始。 (例如 ^A 找 A 開頭的) 
  -- - WHERE REGEXP_LIKE(code, '^AB') 找出所有以 "AB" 開頭的代碼。(等同於 LIKE 'AB%'。)
-- $：字串的結尾。 (例如 Z$ 找 Z 結尾的) 
  -- - WHERE REGEXP_LIKE(code, '99$') 找出所有以 "99" 結尾的代碼。(等同於 LIKE '%99'。)
-- |：或是 (OR)。 (例如 Apple|Banana 兩者都抓)，左右兩邊的模式，只要中一個就可以。
  -- - WHERE REGEXP_LIKE(description, 'SALE|FREE')  找出含有 "SALE" 或是 "FREE" 字樣的描述。 
-- + (至少 1 次)： 
  -- - 需求： 找出含有「至少一個數字」的代碼。 
  -- - 範例： WHERE REGEXP_LIKE(code, '[0-9]+')
-- * (0 次或多次)： 
  -- - 需求： 找 A 後面跟著任意個 B（甚至沒有 B 也可以）。
  -- - 範例： WHERE REGEXP_LIKE(code, 'AB*') (會抓到 A, AB, ABB, ABBB...)
-- [ABC]：括號內的其中一個。 (例如 [aeiou] 找母音) 
-- [0-9] 或 \d：任何一個數字。必須是數字。
-- [a-z]：任何一個小寫字母。
  -- - exp: WHERE REGEXP_LIKE(phone, '^09[0-9]{8}$')
  -- - ^09：必須 09 開頭。
  -- - [0-9]：中間必須是數字。
  -- - {8}：數字必須剛好出現 8 次。
  -- - $：後面不能再有任何字了（剛好結束）。
-- 三大資料庫「模糊比對」功能對照:
-- 功能,          Oracle,                                     MySQL,                              PostgreSQL
-- 標準比對,      LIKE,                                       LIKE,                               LIKE (大小寫敏感) / ILIKE (不敏感)
-- 正則函數,      WHERE conditions "REGEXP_LIKE(col, pat)",   WHERE conditions REGEXP '\\bDIAB1', WHERE conditions  ~ '(^| )DIAB1'
-- 單字邊界處理,  強大，支援 [[:space:]],                       支援 \\b,                           支援 \y 或 `(^

-- 1141. User Activity for the Past 30 Days I
-- Oracle 處理日期非常嚴謹，建議使用 TO_DATE 確保格式正確，並利用其強大的日期算術。
-- 在 Oracle 中，DATE 其實包含了日期和時間，當你的 NLS（環境語言設定）預設包含時間時，它就會跑出那串討人厭的 00:00:00。
-- 要在 LeetCode 的 Oracle 環境中消除它，你有幾個方法，最推薦的是 TO_CHAR。
SELECT 
	TO_CHAR(activity_date, 'YYYY-MM-DD') as day, 
	COUNT(DISTINCT user_id) as active_users
FROM Activity 
WHERE activity_date BETWEEN TO_DATE('2019-07-27', 'YYYY-MM-DD') - 29
AND TO_DATE('2019-07-27', 'YYYY-MM-DD')
GROUP BY activity_date

-- 1667. Fix Names in a Table 
SELECT user_id, CONCAT(UPPER(SUBSTR(name, 1, 1)), LOWER(SUBSTR(name, 2, LENGTH(name)))) as name FROM USERS
order by user_id

