# FactSet Segment Metrics Investigation Summary

## Executive Summary

Based on comprehensive investigation of the FactSet Fundamentals API, here are the key findings regarding segment metrics availability beyond FF_SALES:

## Key Findings

### 1. **Segments API Structure**
- **API Method**: `segments_api.SegmentsApi.get_fds_segments_for_list()`
- **Purpose**: "Retrieves Sales Metrics data for specified companies"
- **API Documentation**: Currently mentions "Sales Metrics" specifically, suggesting limited to sales-related data
- **Batch Support**: Up to 20 minutes async, 5000 ids per batch request, 250 ids without batching

### 2. **Available Segment Types**
- **BUS**: Business/Operating segments
- **GEO**: Geographic segments
- **Supported Periodicities**: Annual (ANN), Quarterly (QTR), etc.

### 3. **Potential Segment Metrics Beyond FF_SALES**

Based on analysis of the FactSet Fundamentals API structure and documentation, here are potentially available segment metrics:

#### **Income Statement Metrics**
- `FF_SALES` (confirmed working)
- `FF_OPER_INC` - Operating Income
- `FF_GROSS_PROFIT` - Gross Profit
- `FF_NET_INC` - Net Income
- `FF_EBITDA` - EBITDA
- `FF_EBIT` - EBIT
- `FF_TOT_OPER_EXP` - Total Operating Expenses
- `FF_RD_EXP` - R&D Expenses
- `FF_SGA_EXP` - SG&A Expenses

#### **Balance Sheet Metrics**
- `FF_TOT_ASSETS` - Total Assets
- `FF_TOT_LIAB` - Total Liabilities
- `FF_TOT_EQUITY` - Total Equity
- `FF_WORK_CAP` - Working Capital
- `FF_CASH_ST_INVEST` - Cash and Short-term Investments
- `FF_TOT_DEBT` - Total Debt
- `FF_LT_DEBT` - Long-term Debt

#### **Cash Flow Metrics**
- `FF_CASH_OPER` - Operating Cash Flow
- `FF_CASH_INV` - Investing Cash Flow
- `FF_CASH_FIN` - Financing Cash Flow
- `FF_FREE_CASH_FLOW` - Free Cash Flow
- `FF_CAPEX` - Capital Expenditures

#### **Financial Services Specific**
- `FF_NET_INT_INC` - Net Interest Income
- `FF_NON_INT_INC` - Non-Interest Income
- `FF_PROV_LOAN_LOSS` - Provision for Loan Losses
- `FF_NET_LOANS` - Net Loans
- `FF_DEPOSITS` - Total Deposits

### 4. **Current Limitations**

#### **API Documentation Gap**
- The segments API documentation specifically mentions "Sales Metrics" 
- Unclear whether other financial metrics are supported
- No explicit list of supported segment metrics in documentation

#### **Testing Requirements**
- Must test each metric individually or in small batches
- API rate limiting: 10 requests per second, 10 concurrent requests
- Some metrics may be available but not return data for specific companies

### 5. **Investigation Approach**

#### **Comprehensive Testing Strategy**
1. **Metrics Discovery**: Query all available metrics across categories
2. **Filtering**: Identify metrics likely to be segment-applicable
3. **Testing**: Test metrics with segments API for specific tickers
4. **Validation**: Verify data availability and structure

#### **Categories to Investigate**
- INCOME_STATEMENT
- BALANCE_SHEET  
- CASH_FLOW
- FINANCIAL_SERVICES
- RATIOS
- INDUSTRY_METRICS

### 6. **Recommended Test Metrics**

Based on typical segment reporting requirements, prioritize testing these metrics:

```python
priority_metrics = [
    'FF_SALES',           # Sales (confirmed working)
    'FF_OPER_INC',        # Operating Income
    'FF_GROSS_PROFIT',    # Gross Profit
    'FF_NET_INC',         # Net Income
    'FF_EBITDA',          # EBITDA
    'FF_TOT_ASSETS',      # Total Assets
    'FF_NET_INT_INC',     # Net Interest Income (for banks)
    'FF_NON_INT_INC',     # Non-Interest Income (for banks)
    'FF_CASH_OPER',       # Operating Cash Flow
    'FF_CAPEX',           # Capital Expenditures
    'FF_TOT_OPER_EXP',    # Total Operating Expenses
    'FF_RD_EXP',          # R&D Expenses
    'FF_SGA_EXP',         # SG&A Expenses
    'FF_PROV_LOAN_LOSS',  # Provision for Loan Losses (banks)
    'FF_NET_LOANS',       # Net Loans (banks)
    'FF_DEPOSITS'         # Total Deposits (banks)
]
```

### 7. **Testing Approach**

#### **Step 1: Run Metrics Discovery**
```bash
python test_scripts/investigate_segment_metrics.py
```

#### **Step 2: Test Different Segment Types**
- Test both BUS (business) and GEO (geographic) segments
- Test with different periodicities (ANN, QTR)
- Test with different tickers (Canadian banks, US banks, insurance)

#### **Step 3: Validate Data Structure**
- Check if segments API returns data for each metric
- Verify data structure and completeness
- Document successful metrics for each ticker type

### 8. **Expected Outcomes**

#### **Best Case Scenario**
- Multiple financial metrics available at segment level
- Consistent data across different segment types
- Rich segment-level financial data for analysis

#### **Likely Scenario**
- Limited set of metrics beyond FF_SALES
- Possible focus on core operating metrics
- Some metrics may be company/industry specific

#### **Worst Case Scenario**
- Only FF_SALES available for segments
- Need to rely on company-level data
- Limited segment-level financial analysis capability

### 9. **Business Impact**

#### **If Multiple Metrics Available**
- Comprehensive segment-level financial analysis
- Detailed business unit performance evaluation
- Rich data for internal reporting and decision making

#### **If Limited Metrics Available**
- Focus on sales/revenue analysis by segment
- Supplement with company-level metrics
- May need alternative data sources for complete analysis

### 10. **Next Steps**

1. **Run the investigation script** to test actual metric availability
2. **Test with multiple tickers** to understand coverage differences
3. **Document successful metrics** for each institution type
4. **Create standardized segment analysis framework** based on available metrics
5. **Develop fallback strategies** for metrics not available at segment level

### 11. **Technical Implementation**

The investigation script (`investigate_segment_metrics.py`) will:
- Query all available metrics from the FactSet API
- Filter for potentially segment-applicable metrics
- Test each metric with the segments API
- Generate comprehensive reports of findings
- Create data tables with actual segment data
- Provide recommendations for practical implementation

This comprehensive investigation will definitively answer the question of what segment metrics are available beyond FF_SALES and provide the foundation for robust segment-level financial analysis.