import numpy as np

# Transition matrix
P = np.array([
    [5/6, 1/6, 0, 0],
    [5/6, 0, 1/6, 0],
    [5/6, 0, 0, 1/6],
    [0, 0, 0, 1]
])
R = np.array([
    [-1000, 0, 0, 0],
    [-7800, 0, 0, 0],
    [-49500, 0, 0, 100000],
    [0, 0, 0, 0]
])

def find_m_and_Q(probability: float = 0.95,
                start_state: int = 0,
                end_state: int = 3,
                P = np.array([
                [5/6, 1/6, 0, 0],
                [5/6, 0, 1/6, 0],
                [5/6, 0, 0, 1/6],
                [0, 0, 0, 1]
        ])):
    # The number of steps for computation 
    i = 3
    while True:
        # The probability of transitioning from one state to another after i steps in a Markov chain
        Q = np.linalg.matrix_power(P, i) 

        # The probability for the Markov chain to transition from state 0 to state 3 after i tosses
        p = float(Q[start_state][end_state]) 

        # Break while loop if The probability from state 0 to state 3 is >= 0.95
        if p >= probability:
            break
        i = i+1
    
    return Q, i

def cal_cost(r = None,
            P = np.array([
                [5/6, 1/6, 0, 0],
                [5/6, 0, 1/6, 0],
                [5/6, 0, 0, 1/6],
                [0, 0, 0, 1]
            ]),
            R = np.array([
                [-1000, 0, 0, 0],
                [-7800, 0, 0, 0],
                [-49500, 0, 0, 100000],
                [0, 0, 0, 0]
            ])
            ):
    if not r is None:
        R[2][3] = r
    # The cost list is used to store the amount of money gained or lost after each dice roll.
    cost = []
    # Pi is the state probability vector
    pi = np.array([1, 0, 0, 0]) # At begin, we always at state 0
    while True:
        # Calculate expected reward at the first step
        # Formula: Expected Reward = sum(pi_0[i] * P[i, j] * R[i, j] for all i, j)
        expected_reward = np.sum(pi[:, None] * P * R)
        cost.append(float(expected_reward))
        if pi[3]>=0.95:
            break
        pi = np.dot(pi,P)
    return cost

def prize_money(r = None):
    gap = 10000
    r = 110000
    while True:
        cost = cal_cost(r)
        ex = sum(cost)/len(cost)
        if ex > 0:
            break
        r = r + gap
    return r, ex

def two_dice_find_m_and_Q():
    P = np.array([
    [8/9, 1/9, 0, 0],
    [8/9, 0, 1/9, 0],
    [8/9, 0, 0, 1/9],
    [0, 0, 0, 1]
])
    Q, m = find_m_and_Q(P = P)
    return m, Q

def two_dice_cal_cost():
    P = np.array([
    [8/9, 1/9, 0, 0],
    [8/9, 0, 1/9, 0],
    [8/9, 0, 0, 1/9],
    [0, 0, 0, 1]
])
    R = np.array([
                [-1000, 0, 0, 0],
                [-7800, 0, 0, 0],
                [-49500, 0, 0, 3000000],
                [0, 0, 0, 0]
            ])
    cost = cal_cost(P = P, R = R)
    ex = sum(cost)/len(cost)
    return cost

def two_dice_prize_money(r = None):
    P = np.array([
    [8/9, 1/9, 0, 0],
    [8/9, 0, 1/9, 0],
    [8/9, 0, 0, 1/9],
    [0, 0, 0, 1]
])
    R = np.array([
                [-1000, 0, 0, 0],
                [-7800, 0, 0, 0],
                [-49500, 0, 0, 3000000],
                [0, 0, 0, 0]
            ])
    gap = -10000
    r = 3000000
    
    while True:
        cost = cal_cost(r, P = P, R = R)
        ex = sum(cost)/len(cost)
        if ex < 0:
            break
        last_r, last_cost = r, cost
        r = r + gap

    return last_r, last_cost

def main():
    Q, i = find_m_and_Q()
    print(f"The dice tosses do you need to get to 95% win confidence is: {i}")
    print(f"The transition matrix after {i} tosses:\n {Q}")
    cost = cal_cost()
    EX = sum(cost)/len(cost)
    # for index in range(len(cost)):
    #     print(f"Gain/Loss per toss: \n{i}: {cost[i]}")
    print(f"EX of Gain/Loss: {EX}")
    min_reward, ex_min = prize_money()
    print(f"Min reward: {min_reward} --- EX gain/loss for min_rewward: {ex_min}")
    relevant_premium = min_reward - 100000
    print(f"Relevant Premium: {relevant_premium}")
    print(f"========== For 2 dices ==========")
    m_2,  Q_2 = two_dice_find_m_and_Q()
    print(f"The dice tosses do you need to get to 95% win confidence is: {m_2}")
    print(f"The transition matrix after {m_2} tosses:\n {Q_2}")
    cost_2 = two_dice_cal_cost()
    EX_2 = sum(cost_2)/len(cost_2)
    # for index in range(len(cost_2)):
    #     print(f"Gain/Loss per (2 dices): \n{index}: {cost_2[i]}")
    print(f"EX of Gain/Loss: {EX_2}")
    min_reward_2, cost_min_2 = two_dice_prize_money()
    ex_min_2 = sum(cost_min_2)/len(cost_min_2)
    print(f"Min reward: {min_reward_2} --- EX gain/loss for min_rewward: {ex_min_2}")
    relevant_premium_2 = min_reward_2 - 3000000
    print(f"Relevant Premium: {relevant_premium_2}")


if __name__ == "__main__":
    main()






