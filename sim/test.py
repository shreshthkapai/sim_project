# test.py (in sim/ folder)
import sys
import os
# Add kg folder to Python path (so pickle can find knowledge_graph module)
kg_path = os.path.join(os.path.dirname(__file__), '..', 'kg')
sys.path.insert(0, os.path.abspath(kg_path))

print("Loading checkpoint_FINAL.pkl...")
try:
    # Checkpoint is in models/ folder (one level up from sim/)
    checkpoint_path = os.path.join('..', 'models', 'checkpoint_FINAL.pkl')
    
    import pickle
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    kg = checkpoint['kg']
    graph = kg.G  # UserKnowledgeGraph uses .G not .graph
    
    # Count nodes by type
    user_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'user']
    category_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'category']
    entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'entity']
    
    print("\n" + "="*60)
    print("GRAPH LOADED SUCCESSFULLY")
    print("="*60)
    print(f"Total nodes: {graph.number_of_nodes()}")
    print(f"  - USER: {len(user_nodes)}")
    print(f"  - Categories: {len(category_nodes)}")
    print(f"  - Entities: {len(entity_nodes)}")
    print(f"Total edges: {graph.number_of_edges()}")
    print(f"Search history: {len(kg.search_history)} events")
    print("="*60)
    
    print("\nCategories:")
    for cat in sorted(category_nodes):
        print(f"  - {cat}")
    
    print("\nSample entities (first 10):")
    for ent in sorted(entity_nodes)[:10]:
        clean_name = ent.replace('entity_', '').replace('_', ' ').title()
        print(f"  - {clean_name}")
    
    print("\n✓ Graph is ready for simulation!")
    print("\nNow you need to update graph_setup.py to use kg.G instead of kg.graph")
    
except Exception as e:
    print(f"\n❌ Error loading graph: {e}")
    import traceback
    traceback.print_exc()