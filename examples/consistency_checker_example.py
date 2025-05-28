from agents.consistency_checker_agent import ConsistencyCheckerAgent
from agents.environments.consistency_checker_env import ConsistencyCheckEnvironment

def main():
    # Example documents
    documents = {
        'course_reports': [...],  # Your course reports
        'clo_assessments': [...],  # Your CLO assessments
        'syllabus': [...],  # Your syllabus data
    }
    
    # Create environment and agent
    env = ConsistencyCheckEnvironment(documents)
    agent = ConsistencyCheckerAgent(env)
    
    # Train agent
    agent.train(num_episodes=1000)
    
    # Use trained agent for consistency checking
    state, _ = env.reset()
    inconsistencies = []
    
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        if action['flag'] == 1:
            inconsistencies.append({
                'clo_index': action['clo_index'],
                'issue_type': action['issue_type']
            })
            
        if done:
            break
        state = next_state
        
    print("Found inconsistencies:", inconsistencies)

if __name__ == "__main__":
    main()