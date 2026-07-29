import json
from build_indexes import load_existing_index, MODULE_PERSIST, PARAMS_PERSIST

def inspect_storage(persist_dir: str, label: str):
    print(f"\n{'='*30} {label} {'='*30}")
    try:
        index, nodes = load_existing_index(persist_dir)
        print(f"Total chunks found: {len(nodes)}")
        
        for i, node in enumerate(nodes):
            print(f"\n--- [NODE {i}] ID: {node.id_} ---")
            
            # Show all metadata to see what LlamaIndex actually found
            print(f"METADATA:\n{json.dumps(node.metadata, indent=2)}")
            print("-" * 20)
            
            # Print the FULL content
            print(f"FULL TEXT:\n{node.get_content()}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Could not load {label}: {e}")

if __name__ == "__main__":

    # here write the file whose chunks need to be checked
    inspect_storage(MODULE_PERSIST["evaluator"], "Evaluator")