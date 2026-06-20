import networkx as nx
import pytest
from src.models.simulation_engine import (
    get_live_state,
    simulate_event_impact,
    simulate_with_diversion
)

@pytest.fixture
def mock_graph():
    G = nx.Graph()
    # Simple line graph: A - B - C - D - E
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('C', 'D')
    G.add_edge('D', 'E')
    for node in G.nodes:
        G.nodes[node]['baseline_demand'] = 100.0
    return G

def test_get_live_state(mock_graph):
    # Peak hour (e.g., 8 AM)
    state_peak = get_live_state(8, mock_graph)
    assert 'A' in state_peak
    # Off-peak hour (e.g., 2 AM)
    state_offpeak = get_live_state(2, mock_graph)
    
    # Peak demand should be higher than off-peak
    assert state_peak['A'] > state_offpeak['A']

def test_simulate_event_impact(mock_graph):
    # Simulate event at 'C'
    impact_state = simulate_event_impact('C', 1.0, 8, mock_graph)
    live_state = get_live_state(8, mock_graph)
    
    # Direct impact at C
    assert impact_state['C'] > live_state['C']
    # 1-hop spillover at B and D
    assert impact_state['B'] > live_state['B']
    assert impact_state['D'] > live_state['D']
    # 2-hop spillover at A and E
    assert impact_state['A'] > live_state['A']
    assert impact_state['E'] > live_state['E']

def test_simulate_with_diversion(mock_graph):
    # Diversion route avoids C
    div_route = ['B', 'D']
    div_state, improvement = simulate_with_diversion('C', 1.0, 8, div_route, mock_graph)
    impact_state = simulate_event_impact('C', 1.0, 8, mock_graph)
    
    # Diversion should reduce congestion at C
    assert div_state['C'] < impact_state['C']
    assert improvement > 0
    # Diversion cells should experience increased load
    assert div_state['B'] > impact_state['B']
    assert div_state['D'] > impact_state['D']
