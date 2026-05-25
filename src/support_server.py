import sqlite3

from mcp.server.fastmcp import FastMCP

# 1. Initialize the FastMCP Container
# NOTE: MCP CLI tools (e.g. `mcp dev file.py`) auto-detect a global named
# `mcp`, `server`, or `app`. Keep this as `mcp` for compatibility.
mcp = FastMCP("Support-And-Sales-Data-Bridge")


# ==========================================
# DATA INFRASTRUCTURE & AUTOMATIC SEEDING
# ==========================================
def initialize_and_seed_data() -> sqlite3.Connection:
    """Simulates a production DB server instance and auto-seeds sample records."""

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_name TEXT,
            issue_details TEXT,
            status TEXT,
            priority TEXT
        );
        """
    )

    sample_tickets = [
        (
            "TKT-101",
            "Rajesh Iyer",
            "Payment gateway timeout during checkout on the Chennai Metro booking app. Money debited but no ticket generated.",
            "OPEN",
            "HIGH",
        ),
        (
            "TKT-102",
            "Priyan S.",
            "Model drift detected on our deployed production sales prediction endpoint. Predictions skewed by over 35%.",
            "OPEN",
            "MEDIUM",
        ),
        (
            "TKT-103",
            "Meera Nair",
            "Cannot log into the analytics portal dashboard. Getting a 500 internal server error on loading charts.",
            "RESOLVED",
            "LOW",
        ),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO support_tickets VALUES (?, ?, ?, ?, ?);",
        sample_tickets,
    )
    conn.commit()
    return conn


# Boot up the database client and hold the connection open
_db_connection = initialize_and_seed_data()


# Simulated Elasticsearch Document Index data payload
MOCK_ELASTICSEARCH_CATALOG = [
    {
        "sku": "PROD-AIDA-01",
        "title": "Nexus Traffic-Vision Monitoring Suite",
        "description": "Computer-vision vehicle classification engine optimized for urban traffic hubs. Analyzes raw video feeds, counts vehicles, and predicts congestions using deep learning models.",
        "price": "₹25,000/month",
    },
    {
        "sku": "PROD-FIN-02",
        "title": "Smart-Spend Predictive Ledger",
        "description": "Time-series prediction engine for corporate expense analytics. Automatically groups expenses, maps seasonal trends, and forecasts quarterly institutional savings using advanced regression models.",
        "price": "₹12,000/month",
    },
]


# ==========================================
# MCP tools exposing data to the LLM agent
# ==========================================


@mcp.tool()
def query_support_database(ticket_id: str) -> str:
    """Query the relational support DB for a specific ticket."""

    cursor = _db_connection.cursor()
    cursor.execute(
        "SELECT customer_name, issue_details, status, priority FROM support_tickets WHERE ticket_id = ?;",
        (ticket_id,),
    )
    row = cursor.fetchone()

    if row:
        return (
            "🔍 [DATABASE MATCH FOUND]\n"
            f"Customer: {row[0]}\n"
            f"Issue: {row[1]}\n"
            f"Status: {row[2]}\n"
            f"Priority: {row[3]}"
        )

    return f"❌ Database Query Failed: Ticket ID '{ticket_id}' does not exist in relational records."


@mcp.tool()
def read_ticket_record(ticket_id: str) -> str:
    """Alias for query_support_database (used by host guardrails)."""

    return query_support_database(ticket_id)


@mcp.tool()
def search_elasticsearch_catalog(search_intent: str) -> str:
    """Perform a keyword lookup over the mock product catalog."""

    matches: list[str] = []
    keywords = search_intent.lower().split()

    for product in MOCK_ELASTICSEARCH_CATALOG:
        searchable_block = (product["title"] + " " + product["description"]).lower()
        if any(word in searchable_block for word in keywords):
            matches.append(
                f"📦 Product: {product['title']} [{product['sku']}]\n"
                f"Price: {product['price']}\n"
                f"Summary: {product['description']}"
            )

    if matches:
        return "\n\n".join(matches)

    return "⚠️ Elasticsearch Notice: 0 matching indices found for that requirement pattern."


@mcp.tool()
def execute_database_escalation(ticket_id: str, reason: str) -> str:
    """Escalate a ticket: mark it CRITICAL and routed to human queue."""

    cursor = _db_connection.cursor()
    cursor.execute("SELECT ticket_id FROM support_tickets WHERE ticket_id = ?;", (ticket_id,))

    if not cursor.fetchone():
        return f"Escalation Rejected: Ticket {ticket_id} does not exist."

    cursor.execute(
        """
        UPDATE support_tickets
        SET priority = 'CRITICAL', status = 'ESCALATED_TO_HUMAN'
        WHERE ticket_id = ?;
        """,
        (ticket_id,),
    )
    _db_connection.commit()

    return (
        f"🚨 SUCCESS: Ticket {ticket_id} has been modified in the database. "
        "Status: CRITICAL. Human queue updated. "
        f"Reason: {reason}"
    )


@mcp.tool()
def escalate_to_human_queue(ticket_id: str, reason: str) -> str:
    """Alias for execute_database_escalation (used by host guardrails)."""

    return execute_database_escalation(ticket_id, reason)


if __name__ == "__main__":
    mcp.run(transport="stdio")
