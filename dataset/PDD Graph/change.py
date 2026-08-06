from rdflib import Graph

input_file = "drug_patients_expansion.nt"
output_file = "drug_patients_expansion.ttl"

g = Graph()
g.parse(input_file, format="nt")  

print(f"Graph loaded: {len(g)} triples")

g.serialize(destination=output_file, format="turtle")
print(f"Turtle file saved to {output_file}")
