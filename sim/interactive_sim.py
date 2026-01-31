# interactive_sim.py - Interactive simulation interface
import sys
import os

# DON'T add kg to path - it conflicts with sim/config.py
# The kg module will be found by pickle when loading checkpoint

from simulator import Simulator

print("\n" + "="*60)
print("🧠 PERSONAL KNOWLEDGE GRAPH SIMULATOR")
print("="*60)
print("Predicting search behavior based on 7 years of history...")
print("="*60)

# Initialize simulator
checkpoint_path = os.path.join('..', 'models', 'checkpoint_FINAL.pkl')
print("\nInitializing simulator...")
sim = Simulator(checkpoint_path, verbose=False)

print("\n✅ Ready! Enter life situations and see what you'd search for.")
print("Type 'quit' to exit.\n")

while True:
    query = input("🔍 Enter situation/query: ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not query:
        continue
    
    print("\n" + "-"*60)
    print("Running neural dynamics simulation...")
    print("-"*60)
    
    try:
        result = sim.run_query(
            query=query,
            format='text',
            top_categories=5,
            top_entities_per_category=10
        )
        
        print("\n" + result)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60 + "\n")