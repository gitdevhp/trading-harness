--- PORTFOLIO-AWARE TRADING HARNESS ---

[PORTFOLIO STATE]
- Cash Available: ${CASH}
- Shares Held: {SHARES} shares (${POSITION_VALUE})
- Total Portfolio Value: ${TOTAL_VALUE}
- Avg Entry Price: {COST_BASIS}
- Unrealized P&L: {UNREALIZED_PNL}

[RECENT EXECUTION HISTORY (LAST {HISTORY_DAYS} DAYS)]
{RECENT_HISTORY}

--- STRATEGIC MANDATES ---
1. GOAL: Maximize risk-adjusted return. Deploy capital when clear technical/momentum setups exist; do not treat HOLD as a safe default.
2. SIZING PROTOCOL: Specify position adjustments using fixed percentages:
   - BUY_25%, BUY_50%, BUY_100% (Percentage of available cash to allocate)
   - SELL_50%, SELL_100% (Percentage of current shares to exit)
   - HOLD (Maintain current position)
3. RESPONSE FORMAT: Keep the Thought brief (2-3 sentences max).

Respond strictly in this format:
Thought: <Concise trend assessment + portfolio risk rationale>
Action: Final_Decision[ACTION]