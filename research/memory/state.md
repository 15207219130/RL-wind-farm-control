# Research State

## Current Phase: Phase 1 - Foundation Environment & Baselines

## Status: In Progress

### Completed
- [x] Feasibility assessment: CONFIRMED viable with high novelty
- [x] Literature review: Key gap identified (RL + wake + lifetime integration)
- [x] Project structure created
- [x] FLORIS v4.6.4 environment working (3x3 farm, NREL 5MW turbines)
- [x] Single-agent WindFarmEnv (Gymnasium) — SB3 compatible
- [x] Multi-agent WindFarmMAEnv (PettingZoo) — SB3 compatible via wrapper
- [x] FLORIS gradient optimizer baseline: **+10.2% avg improvement** over greedy

### In Progress
- [ ] PPO single-agent training (100K steps, ~1hr)
- [ ] MAPPO parameter-sharing training (100K steps, ~1hr)

### Next Steps
- Compare RL results vs FLORIS optimizer vs greedy
- Scale training to 500K+ steps if results are promising
- Proceed to Phase 2: Fatigue integration (DEL, CMDP)

## Key Numbers
- Rated farm power (3x3, 12m/s): 27.39 MW
- FLORIS optimizer improvement: +10.2% avg (up to +27% at low wind + aligned wake)
- Training speed: ~27 fps per env (FLORIS bottleneck)

## Last Updated: 2026-04-03
