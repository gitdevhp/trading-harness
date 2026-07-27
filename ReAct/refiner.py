import json
import re
from vllm import LLM, SamplingParams

class StockRefiner:
    
    def __init__(self, llm: LLM, max_refinements: int = 3, temperature: float = 0.3):
        """
        Args:
            llm: vLLM model instance
            max_refinements: Number of refinement iterations
            temperature: Sampling temperature for generation
        """
        self.llm = llm
        self.max_refinements = max_refinements
        self.temperature = temperature
        self.sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=2048, 
            stop=["[END_REFINEMENT]"]
        )
    
    def refine_decision(
        self,
        ticker: str,
        current_date: str,
        current_price: float,
        portfolio_state: dict,
        initial_decision: str,
        market_context: str,
        price_data: str
    ) -> dict:
        """
        Iteratively refine a trading decision through multiple passes.
        
        Args:
            ticker: Stock ticker symbol
            current_date: Trading date
            current_price: Current stock price
            portfolio_state: Dict with cash, shares, cost_basis, total_value
            initial_decision: Initial BUY/SELL/HOLD decision
            market_context: Context about market conditions
            price_data: Historical price data summary
        
        Returns:
            Dict with refined decision, rationale, and refinement history
        """
        
        refinement_history = []
        current_decision = initial_decision
        current_rationale = ""
        
        for iteration in range(self.max_refinements):
            system_prompt = self._build_system_prompt(
                ticker, current_date, portfolio_state
            )
            
            user_prompt = self._build_user_prompt(
                iteration,
                current_decision,
                current_rationale,
                market_context,
                price_data,
                portfolio_state,
                current_price
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            outputs = self.llm.chat(
                messages=messages,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            
            response = outputs[0].outputs[0].text.strip()
            
            # Parse refined decision and rationale
            decision_match = re.search(
                r"REFINED_DECISION:\s*(BUY|SELL|HOLD)(?:_(\d+)%)?",
                response,
                re.IGNORECASE
            )
            
            rationale_match = re.search(
                r"RATIONALE:\s*(.+?)(?:\n|$)",
                response,
                re.DOTALL
            )
            
            confidence_match = re.search(
                r"CONFIDENCE:\s*(\d+)%",
                response
            )
            
            if decision_match:
                action = decision_match.group(1).upper()
                pct = decision_match.group(2)
                
                if pct:
                    current_decision = f"{action}_{int(pct)/100.0}"
                else:
                    current_decision = action
                
                current_rationale = (
                    rationale_match.group(1).strip() 
                    if rationale_match else ""
                )
                confidence = (
                    int(confidence_match.group(1)) 
                    if confidence_match else 50
                )
                
                refinement_history.append({
                    "iteration": iteration + 1,
                    "decision": current_decision,
                    "rationale": current_rationale,
                    "confidence": confidence,
                    "full_response": response
                })
                
                # Exit early if high confidence
                if confidence > 85:
                    print(f"[Refiner] High confidence ({confidence}%) reached at iteration {iteration + 1}")
                    break
            else:
                refinement_history.append({
                    "iteration": iteration + 1,
                    "decision": current_decision,
                    "rationale": "Failed to parse refined decision",
                    "confidence": 0,
                    "full_response": response
                })
        
        return {
            "final_decision": current_decision,
            "final_rationale": current_rationale,
            "refinement_iterations": len(refinement_history),
            "refinement_history": refinement_history,
            "timestamp": current_date
        }
    
    def _build_system_prompt(self, ticker: str, current_date: str, portfolio_state: dict) -> str:
        """Build system prompt with portfolio context."""
        return f"""You are an expert stock trading analyst using continual refinement.
Your role is to iteratively improve trading decisions for {ticker} on {current_date}.

Current Portfolio State:
- Cash Available: ${portfolio_state.get('cash', 0):,.2f}
- Shares Held: {portfolio_state.get('shares', 0):.2f} shares
- Total Portfolio Value: ${portfolio_state.get('total_value', 0):,.2f}
- Cost Basis: ${portfolio_state.get('cost_basis', 0):.2f}
- Unrealized P&L: ${portfolio_state.get('unrealized_pnl', 0):+,.2f}

Rules:
1. Respond with structured format: REFINED_DECISION, RATIONALE, CONFIDENCE
2. Consider portfolio risk, cash constraints, and position sizing
3. Trade sizing: BUY_25%, BUY_50%, BUY_100%, SELL_50%, SELL_100%, HOLD
4. Each iteration should address concerns from the previous decision
5. Provide confidence percentage (0-100)"""
    
    def _build_user_prompt(
        self,
        iteration: int,
        current_decision: str,
        current_rationale: str,
        market_context: str,
        price_data: str,
        portfolio_state: dict,
        current_price: float
    ) -> str:
        """Build user prompt for refinement iteration."""
        
        if iteration == 0:
            # First iteration: analyze initial decision
            return f"""ITERATION {iteration + 1}: INITIAL DECISION ANALYSIS

Current Decision: {current_decision}

Market Context:
{market_context}

Price Data:
{price_data}

Analyze this initial decision. Consider:
1. Is the direction (BUY/SELL/HOLD) appropriate for current market conditions?
2. Is the position sizing optimal given portfolio constraints?
3. What are the key risks?
4. What would increase or decrease your confidence?

Provide refined decision in this format:
REFINED_DECISION: [ACTION]_[%] or HOLD
RATIONALE: [Brief 1-2 sentence explanation]
CONFIDENCE: [0-100]%"""
        
        else:
            # Subsequent iterations: refine based on previous iteration
            return f"""ITERATION {iteration + 1}: REFINEMENT PASS

Previous Decision: {current_decision}
Previous Rationale: {current_rationale}

Price Data:
{price_data}

Portfolio State:
- Cash: ${portfolio_state.get('cash', 0):,.2f}
- Shares: {portfolio_state.get('shares', 0):.2f}
- Current Price: ${current_price:.2f}

Further refinement considerations:
1. Address any concerns from the previous iteration
2. Validate sizing against current portfolio capacity
3. Adjust confidence based on conviction strength
4. Consider market microstructure and execution risk

Provide refined decision in this format:
REFINED_DECISION: [ACTION]_[%] or HOLD
RATIONALE: [Brief 1-2 sentence explanation]
CONFIDENCE: [0-100]%"""

    def batch_refine_decisions(
        self,
        decisions_batch: list,
        portfolio_state: dict
    ) -> list:
        """
        Refine multiple trading decisions in batch.
        
        Args:
            decisions_batch: List of decision dicts with ticker, date, price, etc.
            portfolio_state: Current portfolio state
        
        Returns:
            List of refined decision results
        """
        refined_results = []
        
        for decision_input in decisions_batch:
            result = self.refine_decision(
                ticker=decision_input["ticker"],
                current_date=decision_input["date"],
                current_price=decision_input["price"],
                portfolio_state=portfolio_state,
                initial_decision=decision_input["decision"],
                market_context=decision_input.get("market_context", ""),
                price_data=decision_input.get("price_data", "")
            )
            refined_results.append(result)
        
        return refined_results


# ==========================================
# Convenience Function for Integration
# ==========================================

def refine_trading_decision(
    llm: LLM,
    ticker: str,
    current_date: str,
    current_price: float,
    portfolio_state: dict,
    initial_decision: str,
    market_context: str,
    price_data: str,
    max_refinements: int = 3
) -> dict:
    """
    Standalone function to refine a single trading decision.
    
    Args:
        llm: vLLM model instance
        ticker: Stock ticker
        current_date: Trading date
        current_price: Current price
        portfolio_state: Portfolio info dict
        initial_decision: Initial BUY/SELL/HOLD decision
        market_context: Market context string
        price_data: Historical price data
        max_refinements: Number of refinement passes
    
    Returns:
        Refined decision dict with history
    """
    refiner = StockRefiner(llm, max_refinements=max_refinements)
    return refiner.refine_decision(
        ticker=ticker,
        current_date=current_date,
        current_price=current_price,
        portfolio_state=portfolio_state,
        initial_decision=initial_decision,
        market_context=market_context,
        price_data=price_data
    )
