# AutoRL for Predictive Maintenance Agents

- Ver. 1.0: Mac Version

- 27-Feb: Multi-round
- Store each round results
- Lets NOT eval against REINFORCE. - just impose simple constraint A2C/DQN > 0.7 then retry for RETRY number till < 0.7 
- for PPO, if > 0.92 retry


PROMPT:

Lets extend the -V CLI parameters. -V 1 runs one round of eval and saves only the three main reports.

The theme of my research is REINFORCE, as a light weigth model, is most suitable for edge computing in Industry 4.0.s

Instructions:
1. Add a RETRY_EVAL global constant and set to 5
2. CLI -V option followed by any number > 1, we want to run the eval for N rounds and **save the results of each round** in the '' csv file
3. To differentiate between a single eval when saving, save as "Evaluation_Results_Multiround_xxxxx" 
4. Add two columnns "Round" and "Retry Index" - this will hold the eval. round number and retries attempted
5. If option "-V" is 0 or 1, Round should contain 1, and for any rounds > 1, mention the eval round currently bring run

When is Re-try needed: We are trying to obtain the best REINFORCE model as it is light weight and most suitable for edge computing
6. When evaluatinng any other algo except REINFORCE, if the eval performance score is high - we will retry and see if a lower performance evaluation is evident
7. For A2C and DQN, if the performance eval score is > 0.7, retry upto RETRY_EVAL times, until one is < 0.7. Same thing for PPO, but this time if it is better than 0.85, then retry. If RETRY_EVAL rounds is exhausted, save the one with lowest score
8. The "Retry Index" should contain the index 'i' (for i from 1 to RETRY_EVAL),  will tell us at what try we got the lowest one
9. Note in the 'valuation_Results_Multiround_xxxxx' file, we save only the lowest one with the 'Retry Index', we are not interested in the other rows
10. For REINFORCE, don't retry just keep the first one. We will later see if we want to use the same RETRY_EVAL loop to find the HIGHEST one. For now we are okay with what we get.

Now - about the Heatmap and the Analysis plots. 
11. Make sure you do this AFTER the Eval_Results file (data) is created. 
12. Create these two plots only from the FINAL results (i.e. REINFORCE and the retried lowest other algos.)