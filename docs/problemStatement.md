# Dietary Guidance RAG Chatbot

## Brief

A chatbot that answers questions about food, nutrition and food safety. A retrieval layer sits under it, so it answers only from official public dietary guidance documents. Every claim carries a citation. When the guidance does not cover a question, the assistant says so.

## Requirements

1. **Corpus:** 5 to 7 public guidance documents from recognised authorities (national nutrition institutes, food safety regulators, international health bodies). Written prose only. Store publisher, year, source URL and retrieval date for every document.
2. **Chunking:** every chunk carries document name, publisher, year and section heading. Do not cut tables or numbered recommendations in half. The README states what was chosen and what it cost.
3. **Retrieval:** vector index (ChromaDB) supporting retrieval across all documents and retrieval filtered to one named document.
4. **Answer layer:** answer only from retrieved chunks. Every claim cites document name, publisher, year and a link.
5. **Cross-document questions:** when two documents cover a topic (for example cooking oil), answer per document with separate citations. Never blend two sources into one claim.
6. **Two refusals, both enforced in code, not only in the prompt:**
   - **Not in corpus:** say the guidance does not cover it and name what was searched.
   - **Out of scope by design:** no medical advice, no calorie or weight targets, nothing about what anyone should weigh. Decline and point to a qualified professional.

## Out of scope

Nutrient numbers for individual foods (Milestone 3).

## Stack

Python, ChromaDB, embedding model chosen after chunk analysis, Groq `openai/gpt-oss-120b`, Streamlit, deployed on Streamlit Community Cloud.

## Sources

Retrieval date for all: **2026-10-05**

1. Promotion of "My Plate for the Day" and physical activity among the population to prevent all forms of malnutrition and NCDs in the country (policy brief, 4-page PDF)  
   Publisher: ICMR-National Institute of Nutrition, India | Year: 2024  
   URL: https://www.nin.res.in/brief/Policy%20Brief%20My%20Plate%20J18%2024.pdf

2. Reusing Cooking Oils  
   Publisher: Singapore Food Agency | Year: 2024  
   URL: https://www.sfa.gov.sg/food-safety-tips/food-risk-concerns/risk-at-a-glance/reusing-cooking-oils

3. Healthy diet (fact sheet)  
   Publisher: World Health Organization | Year: 2026  
   URL: https://www.who.int/news-room/fact-sheets/detail/healthy-diet

4. Saturated fatty acid and trans fatty acid intake for adults and children: WHO guideline  
   Publisher: World Health Organization | Year: 2023  
   URL: https://www.who.int/publications/i/item/9789240073630

5. The Eatwell Guide  
   Publisher: Office for Health Improvement and Disparities (OHID), GOV.UK | Year: 2024  
   URL: https://www.gov.uk/government/publications/the-eatwell-guide

6. Healthy eating recommendations  
   Publisher: Health Canada | Year: 2019  
   URL: https://www.canada.ca/en/health-canada/services/food-guide/explore/healthy-eating-recommendations.html

7. Cold Food Storage Charts  
   Publisher: FoodSafety.gov (U.S. government) | Year: 2023  
   URL: https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts
