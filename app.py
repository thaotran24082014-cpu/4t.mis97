"""
Northbridge Bank - Credit Risk Query Engine (Streamlit App)
============================================================
Streamlit conversion of "Project 3 - Credit Risk Query Engine" notebook.

Pipeline (unchanged from the notebook):
  1. Intent classification  -> verified template vs. generated SQL
  2. Query construction     -> load template OR generate fresh SQL
  3. Validation gate        -> read-only, schema, EXPLAIN, LLM relevance, template integrity
  4. Retry once             -> generated track only
  5. Escalate               -> if validation still fails
  6. Execute                -> read-only SQLite connection
  7. Response generation    -> focused business narrative + confidence score

Run locally:
    streamlit run app.py
"""

import json
import os
import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Credit Risk Query Engine | Northbridge Bank",
    page_icon="🏦",
    layout="wide",
)

# =============================================================================
# File paths (override with environment variables if needed)
# =============================================================================
DB_PATH = os.getenv("DB_PATH", "credit_risk_portfolio.db")
TEST_QUERIES_PATH = os.getenv("TEST_QUERIES_PATH", "test_queries.csv")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")


# =============================================================================
# Credentials: Streamlit secrets -> environment variables -> config.json
# =============================================================================
def load_credentials():
    """Load OpenAI credentials (same keys as the notebook's config.json)."""
    api_key, api_base = None, None

    # 1. Streamlit secrets (Streamlit Community Cloud / .streamlit/secrets.toml)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        api_base = st.secrets.get("OPENAI_API_BASE", None)
    except Exception:
        pass

    # 2. Environment variables
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    api_base = api_base or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")

    # 3. config.json (same format as the notebook)
    if not api_key and os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as file:
            config = json.load(file)
            api_key = config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    # Store API credentials in environment variables (as in the notebook)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()


# =============================================================================
# LLM setup (same models / temperature as the notebook)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_llms(api_key, api_base, model_name):
    kwargs = dict(temperature=0, model_name=model_name, openai_api_key=api_key)
    if api_base:
        kwargs["openai_api_base"] = api_base
    _llm = ChatOpenAI(**kwargs)
    _evaluator_llm = ChatOpenAI(**kwargs)
    return _llm, _evaluator_llm


llm, evaluator_llm = (None, None)
if OPENAI_API_KEY:
    llm, evaluator_llm = get_llms(OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME)


# =============================================================================
# Database: READ-ONLY connection (URI mode ?mode=ro)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_connection(db_path):
    # check_same_thread=False because Streamlit serves sessions from worker threads
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def load_test_queries(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# =============================================================================
# Database schema provided to the LLM (verbatim from the notebook)
# =============================================================================
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# =============================================================================
# Verified Query Template Library (verbatim from the notebook)
# =============================================================================
verified_query_library = {
    'VQ1': {
        'title': 'Sector-wise Outstanding and NPA Breakdown',
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': """
SELECT s.sector_name,
       ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding,
       ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                      THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding DESC
"""
    },

    'VQ2': {
        'title': 'Portfolio Outstanding by Loan Category',
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': """
SELECT loan_category,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding DESC
"""
    },

    'VQ3': {
        'title': 'IFRS 9 Stage-wise ECL Summary',
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': """
SELECT ifrs9_stage,
       COUNT(*) AS loan_count,
       ROUND(SUM(ead_amount) / 1000000.0, 2) AS ead_mn,
       ROUND(SUM(ecl_amount) / 1000000.0, 2) AS ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""
    },

    'VQ4': {
        'title': 'Provision Coverage Ratio by Sector',
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': """
SELECT s.sector_name,
       ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_pct
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_pct DESC
"""
    },

    'VQ5': {
        'title': 'Top 10 Loan Exposures',
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': """
SELECT l.borrower_name,
       s.sector_name,
       ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
       l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10
"""
    },

    'VQ6': {
        'title': 'Top 5 Business Group Exposures',
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': """
SELECT group_name,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_exposure_mn
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY SUM(total_outstanding) DESC
LIMIT 5
"""
    },

    'VQ7': {
        'title': 'All Overdue Loan Accounts',
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': """
SELECT l.loan_account_number,
       l.borrower_name,
       s.sector_name,
       ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
       l.days_past_due,
       l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC
"""
    },

    'VQ8': {
        'title': 'DPD Bucket Distribution',
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': """
SELECT CASE WHEN days_past_due = 0 THEN '0 (Current)'
            WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
            WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
            WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
            WHEN days_past_due > 90 THEN '90+'
       END AS dpd_bucket,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)
"""
    },

    'VQ9': {
        'title': 'Latest Rating Downgrades',
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': """
SELECT borrower_id,
       previous_rating,
       internal_rating AS current_rating,
       pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""
    },

    'VQ10': {
        'title': 'ECL Trend Across Reporting Quarters',
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': """
SELECT reporting_date,
       ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""
    }
}


# =============================================================================
# TOOL 1: Intent Classification
# =============================================================================
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Returns:
    - dict: 'route' (verified or generated), 'query_id' (template ID or None),
            'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

You are the intent router for a commercial-lending credit-risk query engine.
Decide whether the user's question is answered by ONE verified template ("verified")
or needs freshly generated SQL ("generated").

Rules:
- Choose "verified" only if a template's data source, metric and grouping give the data needed to
  answer the question. Templates return the full result set, so a question that focuses on one
  sector, category or stage still counts as verified when a template covers that metric.
- Choose "generated" when the question needs a metric, dimension, filter or time window that no
  template provides (for example average interest rate, restructured loans, or a single stage
  tracked across quarters).
- Do not force a near-miss. If you are not sure, choose "generated".
- query_id must be null when route is "generated".
{user_question}

{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# =============================================================================
# TOOL 2: Query Generation
# (called by run_pipeline in the notebook; implemented here following the
#  notebook's description and the conventions of retry_generation)
# =============================================================================
def generate_query(user_question, schema_context):
    '''
    Generates a read-only, SQLite-compatible SQL query for a novel question.

    Returns:
    - str: SQL query as a string.
    '''

    generation_prompt = f"""
You are a senior credit-risk data analyst writing SQL for a bank's commercial-lending portfolio.
Write ONE read-only SQLite query that answers the user's question.

Rules:
- Output only the SQL: a single statement starting with SELECT or WITH.
  No markdown, no code fences, no explanation, no trailing commentary.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, REPLACE, ATTACH or multiple statements.
- Use only tables and columns that exist in the schema below.
- NPA means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
- "Latest" / "current" quarter means reporting_date = '2025-09-30' (provisioning)
  or rating_date = '2025-09-30' (borrower_rating).
- Trend or "over quarters" questions must group by the date column and must NOT filter to a single date.
- Express USD amounts in millions using ROUND(SUM(x) / 1000000.0, 2) and alias with a clear name.
- Use ROUND(..., 2) for averages, rates and percentages.
- Use clear column aliases and sort the result in a sensible business order.

User Question:
{user_question}

Database Schema:
{schema_context}

"""

    generated_sql = llm.invoke(generation_prompt).content.strip()
    generated_sql = re.sub(r'^```sql\s*|\s*```$', '', generated_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    generated_sql = re.sub(r'^```\s*|\s*```$', '', generated_sql, flags=re.MULTILINE).strip()
    return generated_sql


# =============================================================================
# TOOL 3: Query Validation (five checks)
# =============================================================================
def _column_probe(sql):
    """Wrap a query so its output columns can be read without fetching rows.
    (The notebook appended ' LIMIT 0' directly, which is a syntax error for
    templates that already end in LIMIT, e.g. VQ5 and VQ6.)"""
    return f"SELECT * FROM ({sql.strip().rstrip(';')}) LIMIT 0"


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Returns:
    - dict: 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            'relevance_confidence' (float, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]
    # NOTE: as in the notebook, 'unknown' identifiers are informational only
    # (aliases/literals make this list noisy); hard schema errors are caught
    # by the EXPLAIN check below.

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

You are an independent reviewer of SQL for a bank's credit-risk reporting.
Judge whether the SQL correctly answers the user's question.

Check:
1. Right tables and joins for the metric asked.
2. Right metric and aggregation (SUM vs AVG vs COUNT; total_outstanding vs ead_amount, etc.).
3. NPA logic is asset_classification IN ('Substandard','Doubtful','Loss') where NPA is involved.
4. Date logic: latest quarter = 2025-09-30 for provisioning and ratings. Trend questions must
   not be filtered to a single date.
5. Filters match what was asked (sector, stage, category, threshold).

Give "yes" only if the SQL would return the data needed for a correct answer.
Confidence: 0.9-1.0 clearly correct, 0.6-0.89 probably correct with minor doubt,
below 0.6 wrong or unclear.


{track_context}

{user_question}

{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(_column_probe(expected_sql)).description]
            actual_cols = [d[0] for d in cur.execute(_column_probe(candidate_sql)).description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# =============================================================================
# TOOL 4: Retry Generation
# =============================================================================
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
Your previous SQL failed validation. Fix it. Keep the original intent of the question.
Output only the corrected SQL: one read-only SQLite statement starting with SELECT or WITH,
no markdown, no explanation. Use only tables and columns from the schema.
Change only what the validation error requires; do not rewrite parts of the query that were
already correct.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# =============================================================================
# TOOL 5: Query Execution (deterministic, no LLM)
# =============================================================================
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Returns:
    - dict: 'dataframe' (pandas DataFrame), 'reasonable' (bool), 'warnings' (list).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# =============================================================================
# TOOL 6: Response Generation
# =============================================================================
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

You are a credit-risk analyst answering a business user. Use ONLY the data table below.

Rules:
- Answer exactly what was asked, first. If the table covers more than the question (for example
  all sectors when one sector was asked, or all stages when one stage was asked), pull out only
  the relevant row(s) and do not describe the other rows.
- Quote exact figures from the table, with units (USD millions, %, counts). Do not round further
  and do not invent numbers that are not in the table.
- If you rank, compare, or sum values, base it only on what is in the table, and check that any
  total or ratio you state is consistent with the underlying rows.
- If the table does not fully answer the question, say plainly what is missing rather than
  guessing.
- 3 to 5 sentences, plain business English. No SQL, no markdown tables, no column names.

### USER QUESTION

{user_question}

{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True, trace=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    - verbose (bool): if True, prints intermediate stages to the server log.
    - trace (list, optional): collects the same stage messages for display in the UI.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    def emit(msg):
        if trace is not None:
            trace.append(msg)
        if verbose:
            print(msg)

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'execution_warnings': [],
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    emit(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    emit(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    emit(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    emit(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        emit(f"    Failed check: {gate.get('failed_check')}")
        emit(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        emit(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        emit(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            emit(f"    Retry failed check: {gate.get('failed_check')}")
            emit(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        emit(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['execution_warnings'] = exec_result['warnings']

    emit(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        emit(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    emit(f"[5] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# =============================================================================
# Evaluation against ground truth (same logic as the notebook)
# =============================================================================
def evaluate_results(ground_truth, test_results):
    evaluation_rows = []
    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        tr = test_results[i]
        evaluation_rows.append({
            'Test Case': gt['Test Case'],
            'Expected Route': gt['Expected Route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['Expected Route'],
            'Expected Query ID': gt['Expected Query ID'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (
                pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
            ) or tr['query_id'] == gt['Expected Query ID'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count'],
        })

    evaluation_df = pd.DataFrame(evaluation_rows)
    path_accuracy = evaluation_df['Route Match'].mean() * 100
    verified = evaluation_df['Expected Route'].astype(str).str.strip().str.lower() == 'verified'
    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100 if verified.any() else np.nan
    # Non-numeric values such as 'ESCALATED' are excluded from the average
    numeric_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce')
    average_confidence = numeric_confidence.mean()
    return evaluation_df, path_accuracy, query_accuracy, average_confidence


# =============================================================================
# UI helpers
# =============================================================================
def init_state():
    defaults = {
        'audit_log': [],
        'last_result': None,
        'last_trace': [],
        'test_results': {},
        'question_input': '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_audit_entry(result, source="Ask"):
    gate = result.get('gate_result') or {}
    st.session_state.audit_log.append({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'source': source,
        'user_question': result.get('user_question'),
        'route': result.get('route'),
        'query_id': result.get('query_id'),
        'match_reason': result.get('match_reason'),
        'retry_used': result.get('retry_used'),
        'escalated': result.get('escalated'),
        'gate_passed': gate.get('passed'),
        'failed_check': gate.get('failed_check'),
        'gate_details': gate.get('details'),
        'confidence': result.get('confidence'),
        'row_count': result.get('row_count'),
        'execution_warnings': "; ".join(result.get('execution_warnings') or []),
        'candidate_sql': result.get('candidate_sql'),
        'executed_sql': result.get('executed_sql'),
        'narrative': result.get('narrative'),
    })


def render_confidence(conf):
    if conf == 'ESCALATED':
        st.error("Confidence: **ESCALATED** — routed to a human analyst")
    elif conf is None:
        st.warning("Confidence: not available (relevance check returned no score)")
    else:
        try:
            c = float(conf)
        except (TypeError, ValueError):
            st.info(f"Confidence: {conf}")
            return
        if c >= 0.9:
            st.success(f"Confidence: **{c:.2f}** — high")
        elif c >= 0.6:
            st.warning(f"Confidence: **{c:.2f}** — moderate, review recommended")
        else:
            st.error(f"Confidence: **{c:.2f}** — low")


def render_chart(df):
    """Simple auto-chart: first column as label, other numeric columns as values."""
    if df is None or df.empty or len(df.columns) < 2 or len(df) > 50:
        return False
    label_col = df.columns[0]
    num_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return False
    chart_df = df.copy()
    chart_df[label_col] = chart_df[label_col].astype(str)
    chart_df = chart_df.set_index(label_col)[num_cols]
    if "date" in label_col.lower():
        st.line_chart(chart_df)
    else:
        st.bar_chart(chart_df)
    return True


def render_result(result, trace=None, key_prefix="main"):
    """Display narrative, confidence, SQL, raw data and pipeline trace."""
    route = result.get('route')
    qid = result.get('query_id')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", (route or "—").capitalize())
    c2.metric("Query ID", qid if qid else "Generated")
    c3.metric("Retry used", "Yes" if result.get('retry_used') else "No")
    c4.metric("Rows returned", result.get('row_count') if result.get('row_count') is not None else "—")

    render_confidence(result.get('confidence'))

    st.markdown("#### Answer")
    if result.get('escalated'):
        st.error(result.get('narrative'))
    else:
        st.write(result.get('narrative'))

    if result.get('match_reason'):
        st.caption(f"Routing reason: {result['match_reason']}")

    warnings = result.get('execution_warnings') or []
    for w in warnings:
        st.warning(f"Data check: {w}")

    tab_sql, tab_data, tab_chart, tab_gate, tab_trace = st.tabs(
        ["🧾 SQL", "📋 Raw data", "📊 Chart", "✅ Validation gate", "🔎 Pipeline trace"]
    )

    with tab_sql:
        if result.get('executed_sql'):
            st.markdown("**Executed SQL**")
            st.code(result['executed_sql'].strip(), language="sql")
        else:
            st.markdown("**Candidate SQL (not executed)**")
            st.code((result.get('candidate_sql') or "").strip(), language="sql")

    with tab_data:
        df = result.get('dataframe')
        if df is not None:
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download result as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
                key=f"{key_prefix}_dl_csv",
            )
        else:
            st.info("No data — the query was escalated before execution.")

    with tab_chart:
        if not render_chart(result.get('dataframe')):
            st.info("No suitable chart for this result.")

    with tab_gate:
        gate = result.get('gate_result') or {}
        st.json(gate)

    with tab_trace:
        if trace:
            st.code("\n".join(trace), language="text")
        else:
            st.info("No trace recorded.")


# =============================================================================
# APP
# =============================================================================
init_state()

st.title("🏦 Credit Risk Query Engine")
st.caption(
    "Northbridge Bank · Commercial Lending Portfolio · Read-only natural-language query engine "
    "with verified templates, validated SQL generation, confidence scoring and a full audit trail."
)

# ---- Pre-flight checks -------------------------------------------------------
if not os.path.exists(DB_PATH):
    st.error(
        f"Database file not found at `{DB_PATH}`. Place `credit_risk_portfolio.db` next to `app.py` "
        "or set the `DB_PATH` environment variable."
    )
    st.stop()

conn = get_connection(DB_PATH)
test_queries = load_test_queries(TEST_QUERIES_PATH)
llm_ready = llm is not None and evaluator_llm is not None

# ---- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System status")
    st.write(f"**Database:** `{os.path.basename(DB_PATH)}` (read-only)")
    st.write(f"**Model:** `{MODEL_NAME}` (temperature 0)")
    if llm_ready:
        st.success("OpenAI credentials loaded")
    else:
        st.error("OpenAI API key not found. Add `OPENAI_API_KEY` to Streamlit secrets, "
                 "environment variables, or `config.json`.")
    if test_queries is not None:
        st.success(f"Test cases loaded: {len(test_queries)}")
    else:
        st.info("`test_queries.csv` not found — evaluation tab disabled.")

    st.divider()
    st.header("📚 Verified templates")
    for qid, entry in verified_query_library.items():
        st.markdown(f"**{qid}** — {entry['description']}")

    st.divider()
    st.metric("Questions answered this session", len(st.session_state.audit_log))
    if st.button("Clear session audit log"):
        st.session_state.audit_log = []
        st.rerun()

tab_ask, tab_lib, tab_eval, tab_audit, tab_data = st.tabs(
    ["💬 Ask a question", "📚 Verified query library", "🧪 Test cases & evaluation",
     "🗂️ Audit trail", "🗄️ Data & schema"]
)

# =============================================================================
# TAB 1: Ask a question
# =============================================================================
with tab_ask:
    st.subheader("Ask a portfolio question in plain English")

    if test_queries is not None and 'User Query' in test_queries.columns:
        def _use_sample():
            choice = st.session_state.get("sample_choice")
            if choice and choice != "— choose a sample question —":
                st.session_state.question_input = choice

        st.selectbox(
            "Or pick a sample question from the test set",
            ["— choose a sample question —"] + test_queries['User Query'].astype(str).tolist(),
            key="sample_choice",
            on_change=_use_sample,
        )

    st.text_area(
        "Your question",
        key="question_input",
        height=90,
        placeholder="e.g. What is the NPA exposure in the Real Estate sector?",
    )

    run_clicked = st.button("Run query", type="primary", disabled=not llm_ready)

    if run_clicked:
        question = st.session_state.question_input.strip()
        if not question:
            st.warning("Please enter a question.")
        else:
            trace = []
            with st.spinner("Classifying intent, building and validating SQL, executing, and writing the answer..."):
                try:
                    result = run_pipeline(question, conn, verified_query_library, database_schema,
                                          verbose=True, trace=trace)
                    st.session_state.last_result = result
                    st.session_state.last_trace = trace
                    add_audit_entry(result, source="Ask")
                except Exception as e:
                    st.session_state.last_result = None
                    st.error(f"Pipeline error: {e}. The question should be escalated to a human analyst.")
                    if trace:
                        st.code("\n".join(trace), language="text")

    if st.session_state.last_result is not None:
        st.divider()
        st.markdown(f"**Question:** {st.session_state.last_result.get('user_question')}")
        render_result(st.session_state.last_result, st.session_state.last_trace, key_prefix="ask")

# =============================================================================
# TAB 2: Verified query library
# =============================================================================
with tab_lib:
    st.subheader("Verified Query Template Library")
    st.write(
        f"{len(verified_query_library)} pre-approved, version-controlled SQL templates. "
        "Each runs without modification and returns the complete result set."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry.get('title', '')}"):
            st.markdown(f"*{entry['description']}*")
            st.code(entry['sql'].strip(), language="sql")
            if st.button(f"Run {qid}", key=f"run_{qid}"):
                try:
                    exec_result = execute_query(entry['sql'], conn)
                    st.dataframe(exec_result['dataframe'], use_container_width=True)
                    for w in exec_result['warnings']:
                        st.warning(w)
                except Exception as e:
                    st.error(f"Execution failed: {e}")

# =============================================================================
# TAB 3: Test cases & evaluation
# =============================================================================
with tab_eval:
    st.subheader("Test cases and evaluation against ground truth")

    if test_queries is None:
        st.info(f"Add `{TEST_QUERIES_PATH}` next to `app.py` to enable this tab.")
    else:
        st.dataframe(test_queries, use_container_width=True)
        ground_truth = test_queries

        col_a, col_b = st.columns([1, 1])
        with col_a:
            run_all = st.button("Run all test cases", type="primary", disabled=not llm_ready)
        with col_b:
            if st.button("Clear test results"):
                st.session_state.test_results = {}
                st.rerun()

        if run_all:
            progress = st.progress(0.0, text="Running test cases...")
            for i, (_, row) in enumerate(ground_truth.iterrows()):
                trace = []
                try:
                    res = run_pipeline(row['User Query'], conn, verified_query_library, database_schema,
                                       verbose=True, trace=trace)
                except Exception as e:
                    res = {'user_question': row['User Query'], 'route': None, 'query_id': None,
                           'match_reason': None, 'candidate_sql': None,
                           'gate_result': {'passed': False, 'failed_check': 'pipeline_error', 'details': str(e)},
                           'retry_used': False, 'escalated': True, 'executed_sql': None,
                           'row_count': None, 'confidence': 'ESCALATED',
                           'narrative': f"Pipeline error: {e}", 'execution_warnings': [],
                           'dataframe': None}
                res['_trace'] = trace
                st.session_state.test_results[i] = res
                add_audit_entry(res, source=f"Test: {row['Test Case']}")
                progress.progress((i + 1) / len(ground_truth),
                                  text=f"Completed {i + 1} of {len(ground_truth)}")
            progress.empty()

        # Individual test case results
        for i, (_, row) in enumerate(ground_truth.iterrows()):
            with st.expander(f"{row['Test Case']}: {row['User Query']}", expanded=False):
                st.markdown(
                    f"**Expected route:** {row.get('Expected Route')} · "
                    f"**Expected query ID:** {row.get('Expected Query ID') if pd.notna(row.get('Expected Query ID')) else '—'}"
                )
                if 'Expected Answer' in row and pd.notna(row.get('Expected Answer')):
                    st.markdown(f"**Expected answer:** {row['Expected Answer']}")

                if st.button("Run this test case", key=f"run_tc_{i}", disabled=not llm_ready):
                    trace = []
                    try:
                        res = run_pipeline(row['User Query'], conn, verified_query_library, database_schema,
                                           verbose=True, trace=trace)
                        res['_trace'] = trace
                        st.session_state.test_results[i] = res
                        add_audit_entry(res, source=f"Test: {row['Test Case']}")
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")

                if i in st.session_state.test_results:
                    st.divider()
                    res = st.session_state.test_results[i]
                    render_result(res, res.get('_trace'), key_prefix=f"tc{i}")
                else:
                    st.info("Not run yet.")

        # Aggregate evaluation (only once every test case has a result)
        st.divider()
        st.markdown("### Evaluation summary")
        if len(st.session_state.test_results) == len(ground_truth) and len(ground_truth) > 0:
            test_results = [st.session_state.test_results[i] for i in range(len(ground_truth))]
            evaluation_df, path_accuracy, query_accuracy, average_confidence = evaluate_results(
                ground_truth, test_results
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            m2.metric("Selected Query Accuracy",
                      f"{query_accuracy:.1f}%" if pd.notna(query_accuracy) else "—")
            m3.metric("Average Confidence Score",
                      f"{average_confidence:.2f}" if pd.notna(average_confidence) else "—")
            st.dataframe(evaluation_df, use_container_width=True)
            st.download_button(
                "Download evaluation as CSV",
                evaluation_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_results.csv",
                mime="text/csv",
            )
        else:
            st.info(
                f"Run all {len(ground_truth)} test cases to compute the evaluation "
                f"({len(st.session_state.test_results)} completed)."
            )

# =============================================================================
# TAB 4: Audit trail
# =============================================================================
with tab_audit:
    st.subheader("Audit trail")
    st.write(
        "Every system-generated answer in this session is logged with its route, template, "
        "validation outcome, SQL, confidence and narrative."
    )
    if st.session_state.audit_log:
        audit_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(audit_df, use_container_width=True)
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download audit log (CSV)",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name=f"audit_log_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )
        d2.download_button(
            "Download audit log (JSON)",
            json.dumps(st.session_state.audit_log, indent=2, default=str).encode("utf-8"),
            file_name=f"audit_log_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
        )
        st.caption("Note: the in-app audit log lives for the browser session. Download it to retain a permanent record.")
    else:
        st.info("No queries have been run yet in this session.")

# =============================================================================
# TAB 5: Data & schema
# =============================================================================
with tab_data:
    st.subheader("Database preview")
    tables = ['sector_master', 'loan_master', 'borrower_rating', 'provisioning']
    for table in tables:
        with st.expander(f"Table: {table}"):
            try:
                df_preview = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5;", conn)
                st.dataframe(df_preview, use_container_width=True)
            except Exception as e:
                st.error(f"Could not preview {table}: {e}")

    st.subheader("Schema provided to the LLM")
    st.code(database_schema.strip(), language="text")
