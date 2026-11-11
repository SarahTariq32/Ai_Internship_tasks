from llm_helper import generate_multi_queries

query = "How does Cassie meet Michael?"

results = generate_multi_queries(query)

for r in results:

    print(r)