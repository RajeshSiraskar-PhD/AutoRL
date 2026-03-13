
# Article notes
------------------------------------------------------------

### Results folder : 
- paper\SIT_Results
- Set-1: SIT training file 15-17-19
- Set-2: SIT training file 7-8-9-19-20

## Hypothesis
1. Primary hypothesis: The computationally light REINFORCE algo performs better than more advanced A2C, DQN and PPO
2. Scondary hypothesis: The same RL environment design and reward function design works perfectly well for different milling machine settings - proved by using two very different data sets IEEE and SIT 
3. Secondary hypothesis: Attention mechanism can add to REINFORCE whenver it lacks and perform better that PPO the most advanced algorithm

## Result Artefacts - 4 artefacts for supporting 3 hypotheses

Artefact 1: 
_Analysis_Report_SIT_Set-N 
- Consolidated analysis dashboard. 

Parts of the Analysis Report components:
----------------------------------------
1. OVERALL PERFORMANCE -- Eval score in a 4x4 matrix. Algo by Attention
2. MODEL PERFORMANCE -- Attention & Algo, but as a bar chart and importantly "CI based confidence error bars" 
3. ALGORITHM PERFORMANCE - Focussed on algorithm (Avg ± CI)
4. LAMBDA METRIC - focussed on Lambda - which is a prognostic specific metric Lambda i.e how close to threhold is the first prediction made. Too big +ve number means its either too early and too big a -ve number means too late. A small +/- Lambda is good - we are close enough to threshold that we maximize tool life as well as save work piece quality

Artefact 2: 
_Heatmap_SIT_Set-N 
- How each model trained on data set X, did when evaluated on dataset Y 

Artefact 3:
_Statistical_p-value_SIT_Set-1 > p value heat map (blue is best, red is worst)
_Statistical_t_SIT_Set-1 > t-test t-stat values heat map (green is best, red is worst)

Artefact 4:
_Hypothesis_Tests_SIT_Set-1.csv > Hypthesis values

