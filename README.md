# Vanilla Policy Gradient

The implementation of this algorithm required some prior knowledge of some key components of RL

Not each was followed to the later but the same effect was achieved.

## Trajectory

The trajectory was implemented using a deque object. It sort of acted like the group and its order was the total number of elements in it.

Each element consisted of a MDP implementation (S, A, R, S', p)

## Reward Function

Both Finite-Horizon Discounted and Infinite-Horizon Undiscounted were tried upon. The gamma valu instantiated with the agent acts as our discount factor. However, the rewards to go approach was used to reduce variance and to focus more on the recent rewards.

## Value Function

Both the on-policy value and on-policy action value functions were tried upon and different ways of implementing them to calculate the advantage were tried. Eventually, since the results yielded less for the episodes done a much simpler approach was used.

## Minimalistic Approach

The model predicted the probs of trajectory states using current weights, they were then sampled using the Categorical distribution and the result was passed upon the log_prob() function to create log-likelihoods.

The log-likelihoods were then multiplied to the rewards-to-go and the all the result was summed up using the torch.sum() function.
(# loss that when differentiated with autograd gives the gradient of J(θ))

The policy weights are then updated and the policy keeps training for set number of episodes.
