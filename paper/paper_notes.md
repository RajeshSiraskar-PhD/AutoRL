
# Article notes
------------------------------------------------------------

### Results folder : 
- figures\SIT_Results and figures\IEEE_Results 

SIT_Results folder contains evaluation results of 7 models trained (trained over two rounds of training) on 7 SIT data sets, and each trained model is then evaluated on 14 unseen dataset  
- Set-1: SIT training file 15-17-19
- Set-2: SIT training file 7-8-9-19-20

Similarly IEEE_Results folder contains eval results on 5 trained models, each evaluated on the other 4 unseen datasets   

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

=========================================================================
# P.E. AREA > Prisim, VSCode and UnderMind 
------------------------------------------------------------------------
## Thoughts for starting research article
------------------------------------------------------------------------

Write an article suitable for publication in a high level scientific journal. 

Research Theme: Reinforcement Learning using REINFORCE algorithm for predictive mainteance
Research Topic: Study and implemention of AutoRL for creating Reinforcement Learning agents for predictive mainteance of a milling machine. 

Research Objective: Industry 4.0, Edge computing requires computationally light weight models. Determine if REINFORCE models are as good as more advanced models such as PPO for this task 

You are provided the following:
1. Code base - two files. train_agent.py - that runs the autoRL and evaluation pipelines. rl_pdm.py that contains the attention mechanisms implemtations
2. Results of the trained models - evaluated on unseen data
3. Two slides about evaluation metrics

Instructions:

1. Overall length of paper, less than 15 pages
2. Scientific PhD audience, think Springer Nature or IEEE Xplore level.
3. Formulas in LateX formats
4. Cite relevant and as far as possible **recent** or **seminal** research

5. Create sections and write about the following:
 - Setting the context: Industry 4.0 + predictive maintenance + edge computing + computationaly light and light weight AI-ML models + lesser carbon footprint when training lighter models + carbon credits
 - AutoRL
 - Attention mechanism in general
 - Attention mechanism for predictive mainteance
 - All basic RL algorithms - A2C, DQN, PPO and REINFORCE
 - Why REINFORCE could be best suited
 
6. My own previous research for creating a sound segway for this research. Use content from this and cite these without fail.

	Ref. 1. An empirical study of the naïve REINFORCE algorithm for predictive maintenance - https://link.springer.com/article/10.1007/s42452-025-06613-1
	For: Study of automated RL frameworks (AutoRL) as platforms to encourage industrial practitioners to apply RL to their problems. The empirical study demonstrated that, in the untuned state, simpler algorithms like the REINFORCE perform reasonably well. For AutoRL frameworks, this research encourages seeking new design approaches to automatically identify optimum algorithm-hyperparameter combinations.

	Ref. 2. Application of the Nadaraya-Watson estimator based attention mechanism to the field of predictive maintenance -  https://www.sciencedirect.com/science/article/pii/S2215016124002073
	For: I applied attention mechanism to predictive maintenance.

	Ref. 3. Reinforcement learning for predictive maintenance: a systematic technical review -
	https://link.springer.com/article/10.1007/s10462-023-10468-6
	For: The section on prognostic specific evaluation metrics. Lambda is what I used. Lambda = time from first prediction of tool replacement to wear threshold. Lower Lambda is better - zero is ideal. 

7. Attention mechanisms used:
   - NW - NadarayaWatson
   - TP - Temporal attention
   - MH - MultiHead
   - SA - Self Attention 
   
8. RESULTS:
 - Models were trained using 3 data files
 - 4 algorithms and 4 attention mechanisms were used
 - 3 x 4 x 4 i.e.. 48 models trained
 - Tested on 6 other **unseen** data files

Metrics for evaluation: 
- A model weighted score "Evaluation Score"
- Tool use % (higher the better)
- Lambda metric is a prognostic specific metric from = time from first prediction of tool replacement to wear threshold. Lower Lambda is better - zero is ideal. 
- Evalaution was based on running 20 rounds of testing on UNSEEN data
- Mention hypothesis testing showed REINFORCE was better than all other algos. using alpha = 0.05
- Then show REINFORCE with multi-head and temporal is the overall best suggestion.

Expand Section on Attention Mechanisms Considered - 
1. Study the rl_pdm.py, look at how each attention mechanism is implemented - write about that.
2. Write in MORE detail about each attention mechanism, how it is implemented in theory - how it was implemented
3. What is special about - pros/cons
4. Finally make Multi-head stand out
5. Why is Temporal showing so much promise too

------------------------------------------------
12-Mar-2026: PRISM PE 
------------------------------------------------

Take a critical look at Artcile_VSC.tex. It was essentially written on the same subject. 
However it had access to the code I actually implemented for AuoRL in Python. 
Given the code - that artcile (Artcile_VSC) - has similar sections like say Temporal Attention - 
but is written around how it was actually implemented.

The article you authored i.e. 'Main_Article.tex' was based on information on the internet.

Your task:
1. First change all citations to the authorname-year format (e.g. Siraskar et al (2023))
2. Review what you have written is factually right
3. Merge contents from Artcile_VSC and expand Main_Article.tex
4. Ensure final article is logical and consistent
5. Important: Ensure implement details are retained with subheading "Design and Implemention"
6. Ensure article is more detailed - for eg if there are MDP related details in the Article_VSC that are 
missed in Main_Artcile, include them
7. If additional references are need to support the Article_VSC content, add that - 
again insure that it is factually correct with references and citations

=========================================================================
# V.2. 14-Mar-2026 > 
=========================================================================

You are an assistant to a PhD Professor helping him write a scientific artcile on his research. 

# ABOUT THE RESEARCH
- Theme: Reinforcement Learning (RL) for Industry 4.0 based Predictive Mainteance (PdM)
- Specific industrial domain: Predictive maintenance (PdM) of a **milling machine**
- Research objectives: 
	RO-1. Industry 4.0: Study various RL algos suitable for PdM
	RO-2. AutoRL: Implement an AutoRL pipeline for creating Reinforcement Learning agents for predictive mainteance
	RO-3. Designed for industrial practioners: The AutoRL allows **industrial practioners** to implement RL - without having to understand RL and hyperparameter tuning
	RO-4. RL environment: Design a RL environment using open source design standards - Open AI Gymnassioum - for a milling machine
	RO-5. RL environment: The RL environment design follows an abstract class OOPS concept - should work for different schemas - ie. different machines and settings and sensor data
	RO-6. Edge compute and Carbon credits: Edge computing requires computationally light weight models. Study scientifically if REINFORCE performs better - since it is a computationally light model
	RO-7. Attention mechanism: Where REINFORCE and other advanced algos are almost at par - can implementing attention mechanism for REINFORCE improve its performance to match say PPO

	## Hypothesis
	1. Primary hypothesis: The computationally light REINFORCE algo performs better than more advanced A2C, DQN and PPO
	2. Scondary hypothesis: The same RL environment design and reward function design works perfectly well for different milling machine settings - proved by using two very different data sets IEEE and SIT 
	3. Secondary hypothesis: Attention mechanism can add to REINFORCE whenver it lacks and perform better that PPO the most advanced algorithm


# SPECIFICATIONS:
1. Referencing style: APA 7
2. Tables: Light grey heading row and bold titles. Numbers always 3 decimal places. For results use averages and the CI mentioned
3. Figures: If you are unable to paste figures - give a URL link. However create the sub-figure or figure arrays correctly
4. Always ensure content references concepts that are explained previously in the paper for flow and provide Section references


# ARTEFACTS
This is a list of all artefacts you can use to write the contents of the paper. I will reference this again in the Instructions section and explain which article sections you can use these in.ß

## Code artefacts - 2 python code files

1. batch_train.py - that runs the autoRL and evaluation pipelines. 
2. rl_pdm.py that contains the environment MT_Env, REINFORCE from scratch implementation, other algo wrapper functions based on Stablebaselines-3 library and from scratch implemtations of attention mechanisms
3. Attention mechanisms implemented:
   - NW - NadarayaWatson
   - TP - Temporal attention
   - MH - MultiHead
   - SA - Self Attention 

## Result Artefacts - 4 artefacts for supporting 3 hypotheses

Note: All analysis and results are of a trained model evaluated on unseen data (within that Schema - so either SIT dataset or IEEE dataset)

### Artefact 1: 
_Analysis_Report_SIT_Set-N.pdf 
- Consolidated analysis dashboard. 

Parts of the Analysis Report components:
1. OVERALL PERFORMANCE -- Eval score in a 4x4 matrix. Algo by Attention
2. MODEL PERFORMANCE -- Attention & Algo, but as a bar chart and importantly "CI based confidence error bars" 
3. ALGORITHM PERFORMANCE - Focussed on algorithm (Avg ± CI)
4. LAMBDA METRIC - focussed on Lambda - which is a prognostic specific metric Lambda i.e how close to threhold is the first prediction made. Too big +ve number means its either too early and too big a -ve number means too late. A small +/- Lambda is good - we are close enough to threshold that we maximize tool life as well as save work piece quality

### Artefact 2: 
_Heatmap_SIT_Set-N.pdf 
- How each model trained on data set X, did when evaluated on dataset Y 

### Artefact 3:
_Statistical_p-value_SIT_Set-1.pdf > p value heat map (blue is best, red is worst)
_Statistical_t_SIT_Set-1.pdf > t-test t-stat values heat map (green is best, red is worst)

### Artefact 4:
_Hypothesis_Tests_SIT_Set-1.csv > Hypthesis values


# INSTRUCTIONS FOR WRITING THE CONTENT:
- Important: Given below is the structure for the article and the ** artefacts that I already have ** so you know ** where to look first for information, and go to web only if necessary. 

## Structure of article:
1. Background
	- Use: UM_Background.md
	- Section "Most important conclusions"
	- Use: UM_Literature_Review.md
	- Set background related to Industry 4.0 + PdM + edge computing + computationaly light and light weight AI-ML models + lesser carbon footprint when training lighter models + carbon credits

2. Core foundational concepts 
	- Use: Use content from these Article_VSC.tex and Article_PRISM.tex, however verify it once again based on good and standard Web knowledge take. If require gather new and fresh inputs from the Web 
	- Write about the following, with formulas, and ** always ** connect with the predictive maintenance angle. 
		- Predictive maintenance (PdM)
		- MDP, POMPD and relation to PdM
		- RL
		- AutoRL
		- All basic RL algorithms - A2C, DQN, PPO and REINFORCE
		- Important: Provide pros and cons in general, as well as the suitability or non-suitability to PdM
		- Why REINFORCE could be best suited?
		- Attention mechanism (AM) in general, pros and cons
 		- Attention mechanism for PdM, pros/cons

3. Linking foundational concepts to how they were implemented
	- Use: Code files from folder 'code' - (1) batch_train.py and (2) rl_pdm.py

4. Experimental design
	- batch_train.py
	- SIT and IEEE datasets. Train on one, evaluate on rest i.e. one-vs-all 
	- Metrics for evaluation: 
	- A model weighted score "Evaluation Score"
	- Tool use % (higher the better)
	- Lambda metric is a prognostic specific metric from = time from first prediction of tool replacement to wear threshold. Lower Lambda is better - zero is ideal. 
	- Write about Lambda as an implemented variant from Prognostics specific metrics from the seminal research by Saxena et al, of NASA. Reference this "Metrics for Offline Evaluation of Prognostic Performance" - link https://c3.ndc.nasa.gov/dashlink/static/media/publication/PHM_2008_Metrics.pdf
	- Evaluationtion was based on running 20 rounds of testing on UNSEEN data
	
5. Experimental results
	- Use information from the 'figures and results' folder. See SIT and IEEE folders for individual results
	- You are going to need some good study of the results, go over it couple of times and then write the Results
	- Mention hypothesis testing showed REINFORCE was better than all other algos. using alpha = 0.05
	- Then show REINFORCE with multi-head is the overall best combination across SIT and IEEE.

6. Further research directions and Conclusions: Use your judgement
 

# REFERENCES  
 
In general use seminal and recent references. Wherever you can try to use my previous research work for creating a ramp for this research. U

	Ref. 1. An empirical study of the naïve REINFORCE algorithm for predictive maintenance - https://link.springer.com/article/10.1007/s42452-025-06613-1
	For: Study of automated RL frameworks (AutoRL) as platforms to encourage industrial practitioners to apply RL to their problems. The empirical study demonstrated that, in the untuned state, simpler algorithms like the REINFORCE perform reasonably well. For AutoRL frameworks, this research encourages seeking new design approaches to automatically identify optimum algorithm-hyperparameter combinations.

	Ref. 2. Application of the Nadaraya-Watson estimator based attention mechanism to the field of predictive maintenance -  https://www.sciencedirect.com/science/article/pii/S2215016124002073
	For: I applied attention mechanism to predictive maintenance.

	Ref. 3. Reinforcement learning for predictive maintenance: a systematic technical review -
	https://link.springer.com/article/10.1007/s10462-023-10468-6
	For: The section on prognostic specific evaluation metrics. Lambda is what I used. Lambda = time from first prediction of tool replacement to wear threshold. Lower Lambda is better - zero is ideal. 


==================================
# MODIFICATIONS
==================================
You are an assistant to a PhD Professor helping him write a scientific artcile using LaTeX, on his research. Title it "AutoRL for Predictive Maintenance".

# ABOUT THE RESEARCH
- Theme: Reinforcement Learning (RL) for Industry 4.0 based Predictive Mainteance (PdM)
- Specific industrial domain: Predictive maintenance (PdM) of a **milling machine**
- Research objectives: 
    RO-1. Industry 4.0: Study various RL algos suitable for PdM
    RO-2. AutoRL: Implement an AutoRL pipeline for creating Reinforcement Learning agents for predictive mainteance
    RO-3. Designed for industrial practioners: The AutoRL allows **industrial practioners** to implement RL - without having to understand RL and hyperparameter tuning
    RO-4. RL environment: Design a RL environment using open source design standards - Open AI Gymnassioum - for a milling machine
    RO-5. RL environment: The RL environment design follows an abstract class OOPS concept - should work for different schemas - ie. different machines and settings and sensor data
    RO-6. Edge compute and Carbon credits: Edge computing requires computationally light weight models. Study scientifically if REINFORCE performs better - since it is a computationally light model
    RO-7. Attention mechanism: Where REINFORCE and other advanced algos are almost at par - can implementing attention mechanism for REINFORCE improve its performance to match say PPO

    ## Hypothesis
    1. Primary hypothesis: The computationally light REINFORCE algo performs better than more advanced A2C, DQN and PPO
    2. Scondary hypothesis: The same RL environment design and reward function design works perfectly well for different milling machine settings - proved by using two very different data sets IEEE and SIT 
    3. Secondary hypothesis: Attention mechanism can add to REINFORCE whenver it lacks and perform better that PPO the most advanced algorithm


# SPECIFICATIONS:
1. Minimum length 20 pages
2. Referencing style: APA 7
3. Tables: Light grey heading row and bold titles. Numbers always 3 decimal places. For results use averages and the CI mentioned
4. Figures: If you are unable to paste figures - give a URL link. However create the sub-figure or figure arrays correctly
5. Always ensure content references concepts that are explained previously in the paper for flow and provide Section references


# ARTEFACTS
This is a list of all artefacts you can use to write the contents of the paper. I will reference this again in the Instructions section and explain which article sections you can use these in.ß

STRICT INSTRUCTIONS: Do NOT read any other folder other than the ones listed here. Limit yourself within the 'AutoRL\paper' folder ** ONLY **. Within that ONLY use content in the main 'paper' folder , followed by 'paper\code' and 'paper\figures and results' folder. Do NOT search any other folder.

## Code artefacts - 2 python code files
Folder: 'paper\code'
1. batch_train.py - that runs the autoRL and evaluation pipelines. 
2. rl_pdm.py that contains the environment MT_Env, REINFORCE from scratch implementation, other algo wrapper functions based on Stablebaselines-3 library and from scratch implemtations of attention mechanisms
3. Attention mechanisms implemented:
   - NW - NadarayaWatson
   - TP - Temporal attention
   - MH - MultiHead
   - SA - Self Attention 

## Result Artefacts - 4 artefacts for supporting 3 hypotheses
Folder: 'paper\figures and results\SIT_Results' and 'paper\figures and results\IEEE_Results'

Note: All analysis and results are of a trained model evaluated on unseen data (within that Schema - so either SIT dataset or IEEE dataset)

### Artefact 1: 
_Analysis_Report_SIT_Set-N.jpg
- Consolidated analysis dashboard. 

Parts of the Analysis Report components:
1. OVERALL PERFORMANCE -- Eval score in a 4x4 matrix. Algo by Attention
2. MODEL PERFORMANCE -- Attention & Algo, but as a bar chart and importantly "CI based confidence error bars" 
3. ALGORITHM PERFORMANCE - Focussed on algorithm (Avg ± CI)
4. LAMBDA METRIC - focussed on Lambda - which is a prognostic specific metric Lambda i.e how close to threhold is the first prediction made. Too big +ve number means its either too early and too big a -ve number means too late. A small +/- Lambda is good - we are close enough to threshold that we maximize tool life as well as save work piece quality

### Artefact 2: 
_Heatmap_SIT_Set-N.jpg 
- How each model trained on data set X, did when evaluated on dataset Y 

### Artefact 3:
_Statistical_p-value_SIT_Set-1.jpg > p value heat map (blue is best, red is worst)
_Statistical_t_SIT_Set-1.jpg > t-test t-stat values heat map (green is best, red is worst)

### Artefact 4:
_Hypothesis_Tests_SIT_Set-1.csv > Hypthesis values


# INSTRUCTIONS FOR WRITING THE CONTENT:
- Important: Given below is the structure for the article and the ** artefacts that I already have ** so you know ** where to look first for information, and go to web only if necessary. 

## Structure of article:
1. Background
    - Use: UM_Background.md
    - Section "Most important conclusions"
    - Use: UM_Literature_Review.md
    - Set background related to Industry 4.0 + PdM + edge computing + computationaly light and light weight AI-ML models + lesser carbon footprint when training lighter models + carbon credits

2. Core foundational concepts 
    - Use: Use content from these Article_VSC.tex and Article_PRISM.tex, however verify it once again based on good and standard Web knowledge take. If require gather new and fresh inputs from the Web 
    - Write about the following, with formulas, and ** always ** connect with the predictive maintenance angle. 
        - Predictive maintenance (PdM)
        - MDP, POMPD and relation to PdM
        - RL
        - AutoRL
        - All basic RL algorithms - A2C, DQN, PPO and REINFORCE
        - Important: Provide pros and cons in general, as well as the suitability or non-suitability to PdM
        - Why REINFORCE could be best suited?
        - Attention mechanism (AM) in general, pros and cons
        - Attention mechanism for PdM, pros/cons

3. Linking foundational concepts to how they were implemented
    - Use: Code files from folder 'code' - (1) batch_train.py and (2) rl_pdm.py

4. Experimental design
    - batch_train.py
    - SIT and IEEE datasets. Train on one, evaluate on rest i.e. one-vs-all 
    - Metrics for evaluation: 
    - A model weighted score "Evaluation Score"
    - Tool use % (higher the better)
    - Lambda metric is a prognostic specific metric from = time from first prediction of tool replacement to wear threshold. Lower Lambda is better - zero is ideal. 
    - Write about Lambda as an implemented variant from Prognostics specific metrics from the seminal research by Saxena et al, of NASA. Reference this "Metrics for Offline Evaluation of Prognostic Performance" - link https://c3.ndc.nasa.gov/dashlink/static/media/publication/PHM_2008_Metrics.pdf
    - Evaluationtion was based on running 20 rounds of testing on UNSEEN data
    
5. Experimental results
    - Use information from the 'figures and results' folder. See SIT and IEEE folders for individual results
    - You are going to need some good study of the results, go over it couple of times and then write the Results
    - Mention hypothesis testing showed REINFORCE was better than all other algos. using alpha = 0.05
    - Then show REINFORCE with multi-head is the overall best combination across SIT and IEEE.

6. Further research directions and Conclusions: Use your judgement
 

# REFERENCES  
 
In general use seminal and recent references. Wherever you can try to use my previous research work for creating a ramp for this research. U

    Ref. 1. An empirical study of the naïve REINFORCE algorithm for predictive maintenance - https://link.springer.com/article/10.1007/s42452-025-06613-1
    For: Study of automated RL frameworks (AutoRL) as platforms to encourage industrial practitioners to apply RL to their problems. The empirical study demonstrated that, in the untuned state, simpler algorithms like the REINFORCE perform reasonably well. For AutoRL frameworks, this research encourages seeking new design approaches to automatically identify optimum algorithm-hyperparameter combinations.

    Ref. 2. Application of the Nadaraya-Watson estimator based attention mechanism to the field of predictive maintenance -  https://www.sciencedirect.com/science/article/pii/S2215016124002073
    For: I applied attention mechanism to predictive maintenance.

    Ref. 3. Reinforcement learning for predictive maintenance: a systematic technical review -
    https://link.springer.com/article/10.1007/s10462-023-10468-6
    For: The section on prognostic specific evaluation metrics. Lambda is what I used. Lambda = time from first prediction of tool replacement to wear threshold. Lower Lambda is better - zero is ideal. 