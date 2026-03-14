# Literature review on lightweight RL for edge predictive maintenance

##### [**Undermind**](https://undermind.ai)

---


## Table of Contents

- [Literature review](#literature-review)
- [Key comparison table](#key-comparison-table)
- [Main literature conclusion for the thesis](#main-literature-conclusion-for-the-thesis)
- [Bridge section for the empirical results](#bridge-section-for-the-empirical-results)
- [Ready-to-use transition text](#ready-to-use-transition-text)
- [Claims the literature can and cannot support](#claims-the-literature-can-and-cannot-support)
- [References](#references)

## Literature review

Predictive maintenance has become a core Industry 4.0 use case because it joins condition monitoring, machine connectivity, and maintenance decision support inside cyber-physical production systems (Ansari et al., 2019; Compare et al., 2020; Ran et al., 2019). Within this setting, reinforcement learning is attractive because maintenance is not only a prediction problem. It is a sequential control problem in which the system must choose when to inspect, defer, repair, or replace under uncertainty, cost, and downtime tradeoffs (Ghobadi et al., 2021; Siraskar et al., 2023; Zhang et al., 2024).

The literature shows two broad streams. The first stream uses lightweight or non-deep reinforcement learning, often with tabular Q-learning, SARSA, policy iteration, or related Markov decision formulations (Adsule et al., 2020; Dadash & Björsell, 2024; Rasay et al., 2024; Wang et al., 2014; Yousefi et al., 2020). The second stream uses deep reinforcement learning, especially DQN, PPO, actor-critic variants, and related methods, to cope with larger state spaces, richer sensor inputs, coupled resource constraints, or more complex maintenance scheduling environments (Aglogallos et al., 2026; Chen et al., 2023; Lee & Mitici, 2022; Liu et al., 2020; Ong, Wenbo, et al., 2022; Ong, Wang, et al., 2022). This split matters for the present research goal because it shows that deep RL is not the default answer to predictive maintenance. It is one answer, often chosen when the problem formulation becomes too rich for simpler methods.

A useful reading of the literature is that many predictive maintenance tasks do not obviously require deep function approximation. Several studies show that good maintenance policies can be learned with simpler RL formulations when the state representation is structured and the decision space is moderate (Adsule et al., 2020; Rasay et al., 2024; Wang et al., 2014; Yousefi et al., 2020). In this sense, the burden of justification lies with algorithmic complexity. A deep method should earn its place by delivering material gains that matter in the target deployment setting, not simply by being newer.

This point becomes stronger under edge deployment constraints. Industry 4.0 predictive maintenance increasingly aims to move analytics closer to equipment through IIoT and edge architectures, but the deployment literature is clear that memory limits, compute budgets, latency, and system integration burdens are real barriers (Compare et al., 2020; Njor et al., 2024; Pandey et al., 2023). The same pattern appears in embedded RL more broadly. When deep RL is pushed toward constrained hardware, it often needs extra support in the form of transfer learning, policy distillation, pruning, or compression before it becomes practical (Jang et al., 2020; *Resource Efﬁcient Deep Reinforcement Learning for Acutely Constrained TinyML Devices*, 2020; Szydlo et al., 2022; Xu et al., 2022). Even inside predictive maintenance, transfer learning is introduced to cut deep RL training wall time by 58 percent, which is evidence that training burden is a first-order concern rather than a small engineering detail (Ong, Wang, et al., 2022).

Against that backdrop, a lightweight method such as REINFORCE becomes attractive for reasons that are architectural as much as algorithmic. A simpler method may be easier to train, easier to port, easier to inspect, and easier to fit within edge resource budgets. The literature found here contains a direct empirical anchor in a 2025 study of naïve REINFORCE for predictive maintenance (Siraskar et al., 2025). More importantly, the wider PdM literature already establishes that useful maintenance policies can often be learned without deep policy networks (Adsule et al., 2020; Dadash & Björsell, 2024; Rasay et al., 2024; Wang et al., 2014; Yousefi et al., 2020). The case for REINFORCE therefore does not need to begin from scratch. It can be framed as part of a larger and already credible tradition of lightweight maintenance control.

That said, the literature does not support a broad claim that REINFORCE is universally superior to DQN, A2C, or PPO. The strongest direct comparative evidence in predictive maintenance still favors more advanced methods in several high-complexity settings. PPO performs strongly in IIoT resource allocation and maintenance coordination tasks (Ong, Wenbo, et al., 2022), PPO and SAC perform well in health-state prediction settings with richer environment structure (Aglogallos et al., 2026), and deep RL methods are repeatedly adopted for large multi-component or long-horizon maintenance optimization problems (Chen et al., 2023; Lee & Mitici, 2022; Liu et al., 2020; Valet et al., 2022). These papers do not invalidate the lightweight argument. They show where deep methods are most likely to earn their complexity.

The literature therefore supports a narrower but stronger thesis-ready conclusion. Lightweight RL is not best defended as universally better. It is best defended as more suitable when predictive maintenance must run near the edge, when deployment simplicity and hardware feasibility are part of the objective, and when the PdM problem is structured enough that advanced deep function approximation offers limited extra value (Jang et al., 2020; Njor et al., 2024; Pandey et al., 2023). In such cases, a computationally light method may be the more scientifically and industrially responsible choice.

## Key comparison table

| Theme | Literature conclusion | Implication for REINFORCE-focused research | Key papers |
|:---|:---|:---|:---|
| Core PdM fit | RL is well matched to maintenance because maintenance is a sequential decision problem under uncertainty, not only a prediction task | REINFORCE enters a well-motivated problem class rather than an artificial benchmark choice | (Ghobadi et al., 2021; Siraskar et al., 2023; Zhang et al., 2024) |
| Need for deep models | Many PdM studies succeed with non-deep RL when the state and action spaces are manageable | A lightweight method is defensible if the experimental setup does not require very high-dimensional representation learning | (Adsule et al., 2020; Dadash & Björsell, 2024; Rasay et al., 2024; Wang et al., 2014; Yousefi et al., 2020) |
| Where deep RL helps | DQN, PPO, A2C-style, and related deep methods help most in richer IIoT, multi-resource, or high-complexity maintenance settings | Deep baselines are important, but they should be treated as complexity-costly tools rather than automatic winners | (Aglogallos et al., 2026; Chen et al., 2023; Lee & Mitici, 2022; Liu et al., 2020; Ong, Wenbo, et al., 2022; Ong, Wang, et al., 2022) |
| Edge feasibility | Edge PdM and embedded RL literatures repeatedly report compute, memory, latency, and deployment constraints | This is the strongest literature basis for preferring REINFORCE if performance is competitive | (Jang et al., 2020; Njor et al., 2024; Pandey et al., 2023; *Resource Efﬁcient Deep Reinforcement Learning for Acutely Constrained TinyML Devices*, 2020; Szydlo et al., 2022; Xu et al., 2022) |
| Sustainability claim | Maintenance optimization can reduce energy use and emissions, but direct algorithm-level carbon comparisons are rare | Carbon benefits should be framed as a downstream implication of lightweight deployment, not as a directly proven result of REINFORCE itself | (Bányai, 2021; Bányai & Bányai, 2022; Jasiulewicz-Kaczmarek, 2024; Pandey et al., 2023) |

## Main literature conclusion for the thesis

A thesis-ready position from the literature is as follows. Predictive maintenance in Industry 4.0 requires not only accurate decision policies but also deployable ones. The literature shows that many maintenance control problems can be solved effectively with lightweight RL methods, while deep RL becomes most useful in larger and more entangled environments (Adsule et al., 2020; Liu et al., 2020; Ong, Wenbo, et al., 2022; Wang et al., 2014; Yousefi et al., 2020). Because edge deployment introduces strict limits on compute, memory, latency, and engineering overhead, algorithmic simplicity becomes part of performance rather than an afterthought (Jang et al., 2020; Njor et al., 2024; Pandey et al., 2023). Under these conditions, REINFORCE is defensible not because the literature proves it always beats DQN, A2C, or PPO, but because the literature supports the broader principle that simpler RL can be the better fit when deployment realism is included in the evaluation criterion (Siraskar et al., 2023; Siraskar et al., 2025; Zhang et al., 2024).

## Bridge section for the empirical results

The literature and the empirical study make different kinds of claims and should be kept separate. The literature establishes the problem logic: predictive maintenance is a sequential decision task, lightweight RL is already credible in maintenance optimization, and edge deployment makes computational efficiency a central concern rather than a secondary implementation issue (Adsule et al., 2020; Njor et al., 2024; Pandey et al., 2023; Siraskar et al., 2023; Wang et al., 2014). The empirical study then adds a narrower claim: under a common predictive maintenance setup and a shared set of evaluation conditions, REINFORCE can be compared directly with DQN, A2C, and PPO.

A clean bridge into the experiments can therefore be written in this way. The literature suggests that deep RL methods are not always necessary for predictive maintenance, especially when deployment on edge devices imposes tight resource constraints (Jang et al., 2020; Njor et al., 2024; Pandey et al., 2023). It also suggests that deep methods tend to justify themselves mainly in more complex maintenance environments with richer state spaces or stronger coupling across decisions (Aglogallos et al., 2026; Liu et al., 2020; Ong, Wenbo, et al., 2022). Against this background, the present empirical comparison tests whether a computationally light policy-gradient method can match or exceed more complex deep RL baselines in a predictive maintenance setting while offering a more practical path to edge deployment.

A second bridge is needed for the statistics. The t-tests should be used to support the claim that the observed differences are unlikely to be due to chance under the chosen experimental design. They should not be used to claim universal superiority across all predictive maintenance problems. The clean interpretation is local and conditional: within the studied predictive maintenance environment, and under the selected metrics, REINFORCE achieved outcomes that were statistically distinguishable from or at least competitive with DQN, A2C, and PPO. If the experiments also show lower training cost, lower inference cost, lower memory use, or simpler deployment, then the contribution becomes stronger because it joins statistical performance evidence with deployment relevance.

## Ready-to-use transition text

The predictive maintenance literature does not show that increasingly complex reinforcement learning algorithms are always the best choice. Rather, it shows a split landscape. Simpler RL methods remain effective across many maintenance-control formulations (Adsule et al., 2020; Wang et al., 2014; Yousefi et al., 2020), while deep RL methods are mainly favored in settings with richer state spaces, coupled resources, or more demanding optimization structure (Lee & Mitici, 2022; Liu et al., 2020; Ong, Wenbo, et al., 2022). At the same time, edge deployment in Industry 4.0 places strict limits on compute, memory, and latency, which raises the practical cost of algorithmic complexity (Compare et al., 2020; Njor et al., 2024; Pandey et al., 2023). This motivates a direct empirical test of whether a lightweight method such as REINFORCE can offer equal or better predictive maintenance performance than DQN, A2C, and PPO under a deployment-relevant evaluation setup.

The experimental results should therefore be read as an extension of the literature rather than a contradiction of it. If REINFORCE achieves superior or statistically comparable maintenance performance while using less computational effort, the result supports a strong applied conclusion: for edge-oriented predictive maintenance, the most suitable algorithm is not necessarily the most sophisticated one. It may instead be the method that achieves the best tradeoff between decision quality, computational burden, and deployment feasibility. Such a conclusion is well aligned with Industry 4.0 priorities around edge intelligence and operational efficiency (Ansari et al., 2019; Compare et al., 2020), while any carbon-emissions benefit should be presented as a plausible downstream consequence of lower computational demand rather than a directly measured property unless the experiments explicitly quantify it (Bányai, 2021; Bányai & Bányai, 2022).

## Claims the literature can and cannot support

| Claim type | Supported by current literature | How to phrase it safely |
|:---|:---|:---|
| REINFORCE is a legitimate PdM method | Yes | REINFORCE is a credible lightweight RL candidate for predictive maintenance, with direct though still limited empirical support (Siraskar et al., 2025). |
| Simpler RL can be preferable in edge PdM | Yes | Lightweight RL may be more suitable when deployment constraints are treated as part of the optimization objective (Jang et al., 2020; Njor et al., 2024; Pandey et al., 2023). |
| REINFORCE is always better than DQN, A2C, and PPO | No | The present study evaluates whether REINFORCE is better suited within the specific predictive maintenance environment considered here. |
| Lower compute implies lower carbon impact | Partly | Lower computational demand plausibly supports lower energy use and emissions, but direct PdM algorithm-level carbon evidence remains sparse (Bányai, 2021; Bányai & Bányai, 2022; Jasiulewicz-Kaczmarek, 2024). |
| Industry 5.0 directly selects REINFORCE over deep RL | No | Industry 5.0 is better used as a secondary framing around sustainability and human-centered deployment than as a direct algorithm-selection rule (Jasiulewicz-Kaczmarek, 2024). |

---

## References

Adsule, A., Kulkarni, M., & Tewari, A. (2020). Reinforcement learning for optimal policy learning in condition‐based maintenance. In *IET Collaborative Intelligent Manufacturing*. <https://doi.org/10.1049/iet-cim.2020.0022>

Aglogallos, A., Bousdekis, A., Kontos, S., & Mentzas, G. (2026). Health state prediction with reinforcement learning for predictive maintenance. *Frontiers in Artificial Intelligence*, *8*. <https://doi.org/10.3389/frai.2025.1720140>

Ansari, F., Glawar, R., & Nemeth, T. (2019). PriMa: a prescriptive maintenance model for cyber-physical production systems. *International Journal of Computer Integrated Manufacturing*, *32*, 482–503. <https://doi.org/10.1080/0951192X.2019.1571236>

Bányai, Á. (2021). Energy Consumption-Based Maintenance Policy Optimization. In *Energies*. <https://doi.org/10.3390/en14185674>

Bányai, Á., & Bányai, T. (2022). Real-Time Maintenance Policy Optimization in Manufacturing Systems: An Energy Efficiency and Emission-Based Approach. In *Sustainability*. <https://doi.org/10.3390/su141710725>

Chen, Y., Liu, Y., & Xiahou, T. (2023). Dynamic inspection and maintenance scheduling for multi-state systems under time-varying demand: Proximal policy optimization. *IISE Transactions*, *56*, 1245–1262. <https://doi.org/10.1080/24725854.2023.2259949>

Compare, M., Baraldi, P., & Zio, E. (2020). Challenges to IoT-Enabled Predictive Maintenance for Industry 4.0. *IEEE Internet of Things Journal*, *7*, 4585–4597. <https://doi.org/10.1109/JIOT.2019.2957029>

Dadash, A. H., & Björsell, N. (2024). Effective machine lifespan management using determined state–action cost estimation for multi-dimensional cost function optimization. *Production & Manufacturing Research*, *12*. <https://doi.org/10.1080/21693277.2024.2383656>

Ghobadi, Z. D., Haghighi, F., & Safari, A. (2021). An overview of reinforcement learning and deep reinforcement learning for condition-based maintenance. *International Journal of Reliability, Risk and Safety: Theory and Application*. <https://doi.org/10.30699/ijrrs.4.2.9>

Jang, I., Kim, H., Lee, D., Son, Y.-S., & Kim, S. (2020). Knowledge Transfer for On-Device Deep Reinforcement Learning in Resource Constrained Edge Computing Systems. *IEEE Access*, *8*, 146588–146597. <https://doi.org/10.1109/ACCESS.2020.3014922>

Jasiulewicz-Kaczmarek, M. (2024). Maintenance 4.0 Technologies for Sustainable Manufacturing. *Applied Sciences*. <https://doi.org/10.3390/app14167360>

Lee, J., & Mitici, M. (2022). Deep reinforcement learning for predictive aircraft maintenance using probabilistic Remaining-Useful-Life prognostics. *Reliab. Eng. Syst. Saf.*, *230*, 108908. <https://doi.org/10.1016/j.ress.2022.108908>

Liu, Y., Chen, Y., & Jiang, T. (2020). Dynamic selective maintenance optimization for multi-state systems over a finite horizon: A deep reinforcement learning approach. *Eur. J. Oper. Res.*, *283*, 166–181. <https://doi.org/10.1016/j.ejor.2019.10.049>

Njor, E., Hasanpour, M. A., Madsen, J., & Fafoutis, X. (2024). A Holistic Review of the TinyML Stack for Predictive Maintenance. *IEEE Access*, *12*, 184861–184882. <https://doi.org/10.1109/ACCESS.2024.3512860>

Ong, K. S.-H., Wang, W., Hieu, N. Q., Niyato, D., & Friedrichs, T. (2022). Predictive Maintenance Model for IIoT-Based Manufacturing: A Transferable Deep Reinforcement Learning Approach. *IEEE Internet of Things Journal*, *9*, 15725–15741. <https://doi.org/10.1109/JIOT.2022.3151862>

Ong, K. S.-H., Wenbo, W., Niyato, D., & Friedrichs, T. (2022). Deep-Reinforcement-Learning-Based Predictive Maintenance Model for Effective Resource Management in Industrial IoT. *IEEE Internet of Things Journal*, *9*, 5173–5188. <https://doi.org/10.1109/jiot.2021.3109955>

Pandey, R., Uziel, S., Hutschenreuther, T., & Krug, S. (2023). Towards Deploying DNN Models on Edge for Predictive Maintenance Applications. *Electronics*. <https://doi.org/10.3390/electronics12030639>

Ran, Y., Zhou, X., Lin, P., Wen, Y., & Deng, R. (2019). A Survey on Intelligent Predictive Maintenance (IPdM) in the Era of Fully Connected Intelligence. *IEEE Communications Surveys & Tutorials*, *28*, 633–671. <https://doi.org/10.1109/COMST.2025.3567802>

Rasay, H., Azizi, F., Salmani, M., & Naderkhani, F. (2024). Joint Optimization of Condition‐Based Maintenance and Production Rate Using Reinforcement Learning Algorithms. *Quality and Reliability Engineering International*, *41*. <https://doi.org/10.1002/qre.3714>

*Resource Efﬁcient Deep Reinforcement Learning for Acutely Constrained TinyML Devices*. (2020).

Siraskar, R., Kumar, S., Patil, S., Bongale, A., Kotecha, K., & Kulkarni, A. (2025). An empirical study of the naïve REINFORCE algorithm for predictive maintenance. *Discover Applied Sciences*, *7*. <https://doi.org/10.1007/s42452-025-06613-1>

Siraskar, R., Kumar, S., Patil, S., Bongale, A., & Kotecha, K. (2023). Reinforcement learning for predictive maintenance: a systematic technical review. *Artificial Intelligence Review*, 1–63. <https://doi.org/10.1007/s10462-023-10468-6>

Szydlo, T., Jayaraman, P., Li, Y., Morgan, G., & Ranjan, R. (2022). TinyRL: Towards Reinforcement Learning on Tiny Embedded Devices. *Proceedings of the 31st ACM International Conference on Information & Knowledge Management*. <https://doi.org/10.1145/3511808.3557206>

Valet, A., Altenmüller, T., Waschneck, B., May, M., Kuhnle, A., & Lanza, G. (2022). Opportunistic maintenance scheduling with deep reinforcement learning. In *Journal of Manufacturing Systems*. <https://doi.org/10.1016/j.jmsy.2022.07.016>

Wang, X., Wang, H., Qi, C., & Sivakumar, A. I. (2014). *Reinforcement Learning Based Predictive Maintenance for a Machine with Multiple Deteriorating Yield Levels*.

Xu, R., Luan, S., Gu, Z., Zhao, Q., & Chen, G. (2022). LRP-based Policy Pruning and Distillation of Reinforcement Learning Agents for Embedded Systems. *2022 IEEE 25th International Symposium On Real-Time Distributed Computing (ISORC)*, 1–8. <https://doi.org/10.1109/ISORC52572.2022.9812837>

Yousefi, N., Tsianikas, S., & Coit, D. (2020). Reinforcement learning for dynamic condition-based maintenance of a system with individually repairable components. In *Quality Engineering* (Vol. 32, pp. 388–408). <https://doi.org/10.1080/08982112.2020.1766692>

Zhang, Q., Liu, Y., Xiang, Y., & Xiahou, T. (2024). Reinforcement learning in reliability and maintenance optimization: A tutorial. *Reliab. Eng. Syst. Saf.*, *251*, 110401. <https://doi.org/10.1016/j.ress.2024.110401>
