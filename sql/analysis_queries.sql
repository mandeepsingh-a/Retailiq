-- analysis_queries.sql
-- ======================
-- Advanced SQL analyses for RetailIQ. Each block is prefixed with a
-- "name:" marker comment so database.py can run them individually and
-- the README can reference them by name. Demonstrates CTEs, window
-- functions, and cohort-style retention logic -- not just SELECT *.

-- name: monthly_revenue_trend
WITH monthly AS (
    SELECT
        order_year_month AS month,
        SUM(line_revenue) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS active_customers
    FROM transactions
    WHERE is_return = 0
    GROUP BY order_year_month
)
SELECT
    month,
    revenue,
    orders,
    active_customers,
    ROUND(revenue / orders, 2) AS avg_order_value,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0),
        2
    ) AS mom_growth_pct
FROM monthly
ORDER BY month;

-- name: top_customers_by_revenue_rank
WITH customer_revenue AS (
    SELECT
        customer_id,
        SUM(line_revenue) AS total_revenue,
        COUNT(DISTINCT order_id) AS total_orders
    FROM transactions
    WHERE is_return = 0
    GROUP BY customer_id
)
SELECT
    customer_id,
    total_revenue,
    total_orders,
    RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
    NTILE(10) OVER (ORDER BY total_revenue DESC) AS revenue_decile
FROM customer_revenue
ORDER BY total_revenue DESC
LIMIT 20;

-- name: category_share_of_revenue
SELECT
    category,
    SUM(line_revenue) AS category_revenue,
    ROUND(100.0 * SUM(line_revenue) / SUM(SUM(line_revenue)) OVER (), 2) AS pct_of_total_revenue
FROM transactions
WHERE is_return = 0
GROUP BY category
ORDER BY category_revenue DESC;

-- name: monthly_signup_cohort_retention
-- Cohort each customer by signup month, then measure what fraction of
-- each cohort was still purchasing 1, 3, and 6 months later.
WITH cohort AS (
    SELECT
        customer_id,
        strftime('%Y-%m', signup_date) AS cohort_month
    FROM customers
),
orders_with_cohort AS (
    SELECT
        t.customer_id,
        c.cohort_month,
        t.order_year_month,
        CAST(
            (strftime('%Y', t.order_date) - strftime('%Y', c.cohort_month || '-01')) * 12 +
            (strftime('%m', t.order_date) - strftime('%m', c.cohort_month || '-01'))
            AS INTEGER
        ) AS months_since_signup
    FROM transactions t
    JOIN cohort c ON t.customer_id = c.customer_id
    WHERE t.is_return = 0
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM cohort
    GROUP BY cohort_month
)
SELECT
    o.cohort_month,
    cs.cohort_size,
    COUNT(DISTINCT CASE WHEN o.months_since_signup = 1 THEN o.customer_id END) * 1.0 / cs.cohort_size AS retention_m1,
    COUNT(DISTINCT CASE WHEN o.months_since_signup = 3 THEN o.customer_id END) * 1.0 / cs.cohort_size AS retention_m3,
    COUNT(DISTINCT CASE WHEN o.months_since_signup = 6 THEN o.customer_id END) * 1.0 / cs.cohort_size AS retention_m6
FROM orders_with_cohort o
JOIN cohort_sizes cs ON o.cohort_month = cs.cohort_month
GROUP BY o.cohort_month, cs.cohort_size
ORDER BY o.cohort_month
LIMIT 18;

-- name: rfm_raw_components
-- Recency / Frequency / Monetary components computed in pure SQL,
-- consumed by src/features.py rather than recomputed in pandas.
WITH last_txn AS (
    SELECT MAX(order_date) AS max_date FROM transactions
)
SELECT
    customer_id,
    CAST(julianday((SELECT max_date FROM last_txn)) - julianday(MAX(order_date)) AS INTEGER) AS recency_days,
    COUNT(DISTINCT order_id) AS frequency,
    ROUND(SUM(line_revenue), 2) AS monetary
FROM transactions
WHERE is_return = 0
GROUP BY customer_id;
