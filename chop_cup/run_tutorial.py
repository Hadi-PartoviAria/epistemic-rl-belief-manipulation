import time
from chop_cup_env import ChopCupEnv

def print_step(title):
    print(f"\n{'='*60}")
    print(f"🪄  {title}")
    print(f"{'='*60}")

def main():
    print("Welcome to the Chop Cup Code Tutorial!")
    print("We will run the optimal 'magic trick' sequence line by line.")
    print("This will show you exactly how the environment state changes.\n")
    
    # ---------------------------------------------------------
    print_step("STEP 1: INITIALIZATION")
    print("> Code: env = ChopCupEnv(enable_limes=False)")
    env = ChopCupEnv(enable_limes=False)
    
    print("> Code: state, rm_state = env.reset()")
    state, rm_state = env.reset()
    
    print("\n[Initial State]")
    print(env.get_compact_state_str())
    print("Notice: The observer explicitly believes (△) the ball is on the table.")
    time.sleep(2)
    
    # ---------------------------------------------------------
    print_step("STEP 2: THE DECEPTIVE ACTION (PALM BALL)")
    print("The magician secretly takes the ball from the table into their hand.")
    print("> Code: state, rm_state, reward, done, info = env.step('palm_ball')")
    
    state, rm_state, reward, done, info = env.step('palm_ball')
    
    print("\n[Internal Info]")
    print(f"Info: {info}")
    print(f"Reward: {reward}")
    print(f"RM State (Trick Phase): {rm_state.value}")
    
    print("\n[New State]")
    print(env.get_compact_state_str())
    print("Notice: The observer's belief transferred! They now explicitly believe")
    print("the ball is under the cup (△(under)), and the SAT solver deduced")
    print("that it MUST be under the cup (□(under)). But physically, it's in the HAND!")
    time.sleep(2)

    # ---------------------------------------------------------
    print_step("STEP 3: REVEAL EMPTY HAND")
    print("The magician shows their hand... but wait, they actually DO have the ball!")
    print("> Code: state, rm_state, reward, done, info = env.step('reveal_hand')")
    
    state, rm_state, reward, done, info = env.step('reveal_hand')
    
    print("\n[Internal Info]")
    print(f"Info: {info}")
    print(f"RM State: {rm_state.value}")
    
    print("\n[New State]")
    print(env.get_compact_state_str())
    print("Notice: The observer now explicitly believes the ball is in the hand (△(hand)).")
    time.sleep(2)

    # ---------------------------------------------------------
    print_step("STEP 4: THE BIG REVEAL (LIFT CUP)")
    print("The magician lifts the cup. The magnetic ball, which was stuck, is now visible!")
    print("> Code: state, rm_state, reward, done, info = env.step('lift_cup')")
    
    state, rm_state, reward, done, info = env.step('lift_cup')
    
    print("\n[Internal Info]")
    print(f"Info: {info}")
    print(f"Reward: {reward} <--- HUGE SUCCESS!")
    print(f"RM State: {rm_state.value} (uT = Milestone reached)")
    
    print("\n[New State]")
    print(env.get_compact_state_str())
    print("Notice: The ball is visible (ball:VIS). The observer is shocked!")
    print("The trick is complete.")

if __name__ == '__main__':
    main()
