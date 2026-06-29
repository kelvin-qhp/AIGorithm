from agent.subagents.loader import load_subagent_configs

def test_load_subagent_configs():
    config = load_subagent_configs()
    print(config)

if __name__ == '__main__':
    test_load_subagent_configs()