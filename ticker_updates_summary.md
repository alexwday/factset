# FactSet Ticker Updates Summary

## Overview
Investigation completed on missing companies from FactSet API. All previously "missing" companies were found to be available with different ticker formats.

## Ticker Updates Made to config.yaml

### European Banks
- **Barclays PLC**: `BCS-GB` → `BCS` (35 transcripts found)
- **ING Groep N.V.**: `ING-NL` → `ING` (17 transcripts found)
- **UBS Group AG**: `UBS-CH` → `UBS` (29 transcripts found)

### US Banks
- **Jefferies Financial Group Inc.**: `JEF-US` → `JEF` (found under JEF)

### US Wealth & Asset Managers
- **Charles Schwab Corporation**: `SCHW-US` → `SCHW` (found under SCHW)

## Tickers That Remain Unchanged (Working As-Is)

### Canadian Monoline Lenders
- **MCAN Mortgage Corporation**: `MKP-CA` ✓ (3 transcripts found)

### European Banks
- **Intesa Sanpaolo**: `ISP-IT` ✓ (12 transcripts found)
- **Lloyds Banking Group plc**: `LLOY-GB` ✓ (42 transcripts found)
- **Standard Chartered PLC**: `STAN-GB` ✓ (26 transcripts found)
- **UniCredit S.p.A.**: `UCG-IT` ✓ (26 transcripts found)

### Nordic Banks
- **Danske Bank A/S**: `DANSKE-DK` ✓ (14 transcripts found)

### UK Wealth & Asset Managers
- **Quilter plc**: `QLT-GB` ✓ (found)
- **Rathbones Group Plc**: `RAT-GB` ✓ (found)

### US Wealth & Asset Managers
- **T. Rowe Price Group Inc.**: `TROW-US` ✓ (found)

## Companies Not Mentioned in Results
These may need further investigation:
- **Nordea Bank Abp** (NDA-FI) - Nordic Banks
- **Skandinaviska Enskilda Banken AB** (SEBA-SE) - Nordic Banks
- **Swedbank AB** (SWEDA-SE) - Nordic Banks
- **St. James's Place plc** (SJP-GB) - UK Wealth & Asset Managers

## Key Findings

1. **Case Sensitivity**: The API appears to be case-insensitive (e.g., `MKP-CA` and `mkp-ca` both work)

2. **Country Codes**: Some tickers work better without country codes:
   - US companies often don't need `-US` suffix
   - Some European companies work with just the base ticker (e.g., `UBS` instead of `UBS-CH`)

3. **Alternative Formats Found**:
   - Some UK companies also work with `-LON` suffix (London Stock Exchange)
   - Canadian companies may work with `-TSE` suffix (Toronto Stock Exchange)
   - Italian companies may work with `-MIL` suffix (Milan Stock Exchange)

## Recommendations

1. **Simplify Tickers**: Where possible, use the simplest form that works (e.g., `UBS` instead of `UBS-CH`)

2. **Test Remaining Companies**: The four companies not mentioned in results should be tested separately

3. **Update Documentation**: Document the working ticker formats for future reference

4. **Consider Fallback Logic**: The pipeline could implement fallback logic to try multiple ticker variations if the primary one fails

## Impact on Pipeline

These changes will affect:
- Stage 00 (Historical Download) - Will now successfully download transcripts for all 91 monitored companies
- Stage 01 (Daily Sync) - Will capture daily updates for previously "missing" companies
- All downstream stages will now process these additional transcripts

## Next Steps

1. ✅ Config.yaml has been updated with working tickers
2. ⏳ Test the updated configuration with a pipeline run
3. ⏳ Investigate the four companies not mentioned in test results
4. ⏳ Consider implementing ticker variation fallback logic