# Compte rendu de réunion

*Source : `dicte_audio_3.normalized.txt`*

## 1. Executive Summary

Le groupe discute autour de la feuille de route Smart Cockpit, mettant en avant l'intérêt des assistants opérateurs avec IA générative. Ils abordent les briques de calcul et études, ainsi que les compétences d'industrielsisation de contenus complexes. SPEAKER_01 propose automatiquement la création et rédaction des rapports EOD rentabilité, en intégrant plusieurs briques de traitement automatisé. La discussion évolue vers le coût des projets, l'interface utilisateur, la génération de rapports par IA, ainsi que la robustesse des outils. Les participants discutent également de l'automatisation de la rédaction et de l'orchestrage de réunions, soulignant les projets en cours sur le développement d'outils d'orchestration (N8) et de génération de synthèses. Ils abordent ensuite les cas d'utilisation MCP, avancées des travaux et la nécessité de clarifier les besoins.

## 2. Sujets abordés

### 1. Définition de la feuille de route Smart Cockpit  _(00:00:00 → 00:00:38)_

Le groupe discute autour du besoin d'IA générative, avec les participants principalement Mathieu (DataIa manager), Maya (consultante en IA génératifs), Bruno (RD sur le Smart Cockpit), Jérôme Picot (grillerier à la RD), et Dinka. Ils se concentrent sur l'IA générative pour les assistants opérateurs, avec une échelle de maturité depuis 10 ans.

### 2. Discussion des briques de calcul et études  _(00:00:38 → 00:00:45)_

Le speaker 02 a évoqué le besoin d'explorer différentes briques de calcul, d'études et réseaux. Il a mentionné que Mike avait l'occasion de travailler sur ces sujets et qu'il connaissait bien les aspects à explorer. Le speaker 01 a proposé de présenter ses projets.

**Actions :**

- Présentation des projets par le speaker 01 _(resp. Speaker 01)_ — échéance : A définir

### 3. Présentation des compétences et projets IHM  _(00:00:45 → 00:01:00)_

SPEAKER_01 a présenté ses compétences en matière d'industrielsisation de contenus complexes, notamment la transformation en multimodalité pour faciliter la prise de décision. Il a également mentionné son travail sur l'industrialisation des modèles hébergés localement et leur fine-tuning, ainsi que l'utilisation de méthodes comme QNTubinefiel et le rag (ranking). Il a également évoqué ses expériences en matière d'orchestration et de déploiement de modèles.

### 4. Automatisation du rapport sur les études EOD rentabilité  _(00:01:00 → 00:01:37)_

SPEAKER_01 propose d'automatiser le processus de création et rédaction des rapports basés sur les études EOD rentabilité, développées par l'équipe équilibreoffre. Il présente une chaîne de traitement automatisée qui permet d'accélérer la production du rapport sans compromettre sa rigueur. Le processus comprend plusieurs briques : traitements des fichiers PPT et création d'artefacts, assignement des sections, etc.

**Décisions :**

- Automatisation de la chaîne de traitement pour accélérer la production du rapport
- Définition d'une structure préétablie pour le rapport

**Actions :**

- Étudier et développer une chaîne de traitement automatisée _(resp. SPEAKER_01)_ — échéance : Non spécifiée
- Définir la structure préétablie du rapport _(resp. SPEAKER_01)_ — échéance : Non spécifiée

### 5. Analyse du coût et structure des rapports  _(00:01:37 → 00:02:08)_

La discussion a porté sur le coût d'un projet, la description des hypothèses et règles d'assignement, ainsi que les différentes parties de la sélection et catégorisation des slides. Le groupe a également abordé l'interface utilisateur, la génération de rapports par IA, l'assignation des sections, la rédaction section par section, l'intégration des sections, et la correction du rapport.

**Décisions :**

- Aucune décision prise

**Actions :**

- Développer un prototype d’interface utilisateur simple pour démontrer la capacité de l'IA à générer des rapports _(resp. SPEAKER_00)_ — échéance : Vingt-cinq jours incluant les ateliers
- Rédiger une section par section et rédiger uniquement au niveau des bulles points _(resp. SPEAKER_01)_ — échéance : Aucune échéance mentionnée
- Intégrer toutes les sections entre elles après la rédaction _(resp. SPEAKER_01)_ — échéance : Aucune échéance mentionnée
- Corriger le rapport en utilisant un agent qui prend en compte les commentaires sur les bulles points _(resp. SPEAKER_01)_ — échéance : Aucune échéance mentionnée

### 6. Discussion sur l'automatisation de la rédaction et robustesse des outils  _(00:02:08 → 00:02:33)_

Le groupe discute de l'automatisation d'un outil pour la rédaction. L'hypothèse est que plusieurs templates peuvent fonctionner, avec un agent capable d'orienter les chapitres, la signation et le contenu. La robustesse est discutée en termes de gestion des erreurs et traçabilité. Les participants mentionnent également des limites non implémentées pour cette première itération.

### 7. Synthèse et orchestration de réunions  _(00:02:33 → 00:02:55)_

La discussion concerne la synthèse d'événements et données, ainsi que l'orchestration de plusieurs outils pour augmenter les consultants et automatiser la génération de synthèses. Les participants discutent également des projets en cours sur le développement d'outils d'orchestration (N8) et de génération de résumés de réunions.

### 8. Discussion sur les outils et projets  _(00:02:55 → 00:03:05)_

Le débat a porté sur la synthèse des propositions, la souveraineté liée aux outils, le projet de planification Pok et l'orchestration. Les participants ont également abordé la création d'un agent qui peut accéder à plusieurs outils pour la bibliographie et la simulation.

### 9. Analyse de projets  _(00:03:05 → 00:03:25)_

La réunion a principalement abordé les projets en cours, avec une discussion sur le projet de génération de modèles pré-étalés par l'un des participants (SPEAKER_01). Il a également mentionné un projet lié à la détection de froide et au fichiering de documents manuscrits.

### 10. Discussion sur les cas d'usage, MCP et orchestrage  _(00:03:25 → 00:03:55)_

La réunion aborde la nécessité des cas d'usage pour le développement de rapports et synthèses, ainsi que l'intérêt du travail en NLP. Les participants discutent également des serveurs MCP et des simulations, soulignant les connexions entre ces domaines.

### 11. Discussion sur le cas d'usage MCP et avancées des travaux  _(00:03:55 → 00:04:26)_

Le SPEAKER_02 propose une réflexion sur un cas d'usage MCP, soulignant la nécessité de clarifier les besoins. Il mentionne également qu'ils ont déjà des actions prévues pour explorer cette question. Le SPEAKER_00 suggère d'avancer vers le design et d'organiser un atelier design. Les participants approuvent ces suggestions.

**Décisions :**

- Clarifier les besoins sur le cas d'usage MCP
- Organiser un atelier design

**Actions :**

- Clarifier les besoins sur le cas d'usage MCP _(resp. SPEAKER_02)_ — échéance : À définir
- Organiser un atelier design _(resp. SPEAKER_00)_ — échéance : À définir

## 3. Décisions

| # | Sujet | Décision |
|---|-------|----------|
| 1 | Automatisation du rapport sur les études EOD rentabilité | Automatisation de la chaîne de traitement pour accélérer la production du rapport |
| 2 | Automatisation du rapport sur les études EOD rentabilité | Définition d'une structure préétablie pour le rapport |
| 3 | Analyse du coût et structure des rapports | Aucune décision prise |
| 4 | Discussion sur le cas d'usage MCP et avancées des travaux | Clarifier les besoins sur le cas d'usage MCP |
| 5 | Discussion sur le cas d'usage MCP et avancées des travaux | Organiser un atelier design |

## 4. Plan d'attaque — Prochaines actions

| # | Sujet | Action | Responsable | Échéance |
|---|-------|--------|-------------|----------|
| 1 | Discussion des briques de calcul et études | Présentation des projets par le speaker 01 | Speaker 01 | A définir |
| 2 | Automatisation du rapport sur les études EOD rentabilité | Étudier et développer une chaîne de traitement automatisée | SPEAKER_01 | Non spécifiée |
| 3 | Automatisation du rapport sur les études EOD rentabilité | Définir la structure préétablie du rapport | SPEAKER_01 | Non spécifiée |
| 4 | Analyse du coût et structure des rapports | Développer un prototype d’interface utilisateur simple pour démontrer la capacité de l'IA à générer des rapports | SPEAKER_00 | Vingt-cinq jours incluant les ateliers |
| 5 | Analyse du coût et structure des rapports | Rédiger une section par section et rédiger uniquement au niveau des bulles points | SPEAKER_01 | Aucune échéance mentionnée |
| 6 | Analyse du coût et structure des rapports | Intégrer toutes les sections entre elles après la rédaction | SPEAKER_01 | Aucune échéance mentionnée |
| 7 | Analyse du coût et structure des rapports | Corriger le rapport en utilisant un agent qui prend en compte les commentaires sur les bulles points | SPEAKER_01 | Aucune échéance mentionnée |
| 8 | Discussion sur le cas d'usage MCP et avancées des travaux | Clarifier les besoins sur le cas d'usage MCP | SPEAKER_02 | À définir |
| 9 | Discussion sur le cas d'usage MCP et avancées des travaux | Organiser un atelier design | SPEAKER_00 | À définir |
