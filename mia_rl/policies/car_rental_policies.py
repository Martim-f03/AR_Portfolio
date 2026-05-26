def zero_move_policy(mdp):
    return {
        s: 0
        for s in mdp.states()
    }