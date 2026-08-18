# Compte rendu de réunion

*Source : `dicte_audio_3.normalized.txt`*

## 1. Executive Summary

La réunion a débuté par un tour de table initié par **Nordine**, mettant en lumière les besoins communs en IA générative entre **Élé Consulting**, **IVB** (via **Maya**) et **RTE/Smart Cockpit**, avec des priorités sur des assistants pour la conduite ou le soin, ainsi que l’intérêt pour des interfaces conversationnelles comme le MCP. Les échanges ont ensuite exploré les potentialités d’une future mission en identifiant des éléments techniques à structurer, sans préciser de cadre défini.

Les participants ont présenté leurs expertises : **SPÉAKER_01** chez Élé a détaillé ses projets d’industrialisation et de fine-tuning de modèles locaux (comme QNTubine) pour transformer des documents complexes en multimodaux actionnables, tandis que **deRTE** a exposé l’automatisation partielle des rapports EOD via un pipeline nécessitant une clarification des sections préétablies. Le prototype d’outil de génération (POG) pour la catégorisation des slides, validé par **Basile** et **Gérald**, reste en phase itérative avec une boucle de correction humaine.

L’accent a été mis sur l’automatisation robuste des rapports, avec des garde-fous comme la traçabilité des artefacts (commentaires par section) et une validation progressive, tout en explorant des cas d’usage transversaux : synthèse de données manuscrites via NLP, orchestration centralisée via le MCP pour les études énergétiques, ou encore l’extraction structurée de documents administratifs (comme ceux de **Maya**). Aucune décision formelle n’a été prise sur la feuille de route, mais des pistes comme un atelier UX ou une collaboration avec des cabinets spécialisés ont été évoquées pour affiner les priorités.

## 2. Sujets abordés

### 1. Tour de table sur les besoins en IA générative et échanges collaboratifs entre participants  _(00:00:00 → 00:00:38)_

Lors d’un tour de table initié par **Nordine**, expert DataIa Manager chez Élé Consulting, plusieurs participants se sont présentés : **Mathieu**, **Maya** (consultante confirmée chez IVB spécialisée en IA générative), et **Bruno Mélière** (RD pilote la feuille de route Smart Cockpit) ainsi que **Jérôme Picot** (intelligence artificielle et innovation). Les échanges ont porté sur les besoins spécifiques en matière d’IA générative, notamment autour des assistants pour opérateurs dans le domaine de la conduite ou du soin. **Jérôme** a évoqué l’intérêt de leur société pour des interfaces conversationnelles basées sur des modèles comme le MCP (Machine Conversation Platform), visant à améliorer l’interaction fluide avec les outils.

### 2. Prise de contact et exploration des besoins pour une future mission  _(00:00:38 → 00:00:45)_

Dans cet extrait, **SPEAKER_02** propose d’explorer des éléments comme les calculs, études ou réseaux en tant que briques potentielles pour une future mission. Il souligne qu’aucune mission claire n’a été définie à ce stade et suggère de faire connaissance avec **Dandine**, qui aurait déjà travaillé sur ces sujets. **SPEAKER_01** propose ensuite de présenter les projets sur lesquels il a travaillé, sans préciser de détails supplémentaires dans cet extrait.

### 3. Présentation des compétences en IA générative et projets chez Ely  _(00:00:45 → 00:01:00)_

SPÉAKER_01 présente les projets de transformation de contenus complexes (documents hétérogènes) en multimodaux actionnables, notamment pour faciliter la prise de décision ou créer des synthèses. Il évoque son approche globale adoptée depuis son arrivée chez Yaley : industrialisation, monitoring et fine-tuning de modèles génératifs locaux (ex. QNTubine). Il détaille aussi les outils utilisés comme Mistral, VLHRM, Vicorne et Streamlit pour le déploiement, ainsi que des techniques comme RAG (Retrieval-Augmented Generation), vectorisation et orchestration. Le focus porte sur des projets variés allant de l’IAE à des missions précédentes.

### 4. Automatisation de la rédaction des rapports d'études EOD – Pipeline et structure des sections  _(00:01:00 → 00:01:37)_

Le projet consiste en l’automatisation de la création de rapports complets à partir des fichiers PPT générés par l’équipe équilibreoffre et demande. **deRTE** explique que ces fichiers contiennent principalement des graphiques avec peu de texte, nécessitant une analyse fine pour les classer dans des sections prédéfinies (ex : hypothèses, comparaisons France/Europe). L’objectif est d’accélérer la production tout en garantissant la rigueur et l’ancrage des affirmations générées dans le rapport. **PPDT** souligne la nécessité de préciser davantage la structure des sections pour améliorer les règles d’assignation des slides aux parties du document. Les descriptions détaillées des sections (ex : hypothèses, comparaisons géographiques) ont déjà été établies pour guider l’agent chargé de cette étape.

**Actions :**

- Établir et affiner les règles d’assignation des slides aux sections du rapport en fonction des descriptions préétablies (ex : exclusion des données non pertinentes pour l’introduction ou la section hypothèses de coût) _(resp. PPDT)_ — échéance : non spécifiée
- Utiliser les fichiers PPT comme entrée principale pour générer une chaîne de traitement automatisée (OCR + analyse graphique + classification des slides par section) via un pipeline en cinq étapes avec versionning des artefacts intermédiaires _(resp. deRTE)_ — échéance : non spécifiée
- Tester l’option d’ajouter une transcription vocale (format TXT) issue de l’enregistrement d’un modèle comme Spitch pour enrichir le contexte des slides non textuelles _(resp. deRTE)_ — échéance : non spécifiée

### 5. Validation du POG (Prototype d'Outil de Génération) pour la catégorisation et l’assignation des slides dans les rapports  _(00:01:37 → 00:02:08)_

Basile et Gérald discutent des hypothèses et règles d’assignation définies pour structurer les chapitres des rapports en utilisant deux templates : un pour le contenu des chapitres et un autre pour les catégories à signer. Le POG, réalisé en 25 jours (incluant ateliers), a démontré la valeur de l’IA dans la génération de rapports simples via un prototype fonctionnel. Basile précise que ce POG est utilisé actuellement pour rédiger le rapport humainement, avec une boucle de correction itérative basée sur des commentaires et des artefacts comme les bullet points. Il souligne aussi la nécessité d’une vue globale sur toutes les slides pour éviter les erreurs d’assignation dans les sections. Gérald aborde ensuite la possibilité d’adapter l’outil à des cas sans template initial, en intégrant une logique de sélection manuelle de templates par un agent.

**Actions :**

- évaluer avec Gérald si des suites sont nécessaires pour généraliser cette approche de génération de rapports _(resp. Gérald)_ — échéance : non spécifiée
- Basile propose d’explorer l’adaptation du POG pour gérer plusieurs présentations sans template initial, en intégrant une logique de sélection manuelle des templates par un agent _(resp. Basile (à discuter avec les équipes concernées))_ — échéance : non spécifiée

### 6. Analyse des approches techniques pour l'automatisation de la rédaction de rapports avec garde-fous robustes  _(00:02:08 → 00:02:33)_

Lors de cette discussion, **SPEAKER_00** et **SPEAKER_01** expliquent les principes sous-jacents à un outil d’automatisation de rédaction de rapports. Ils soulignent que l’outil peut fonctionner avec plusieurs templates distincts pour générer des sorties variées, en adaptant la sélection des chapitres ou le contenu via une approche orientée agent. **SPEAKER_01** insiste sur les aspects robustesse et traçabilité : séparation de la rédaction par sections, gestion des erreurs, artefacts intermédiaires (comme les commentaires par section), et validation humaine itérative pour corriger les hallucinations. Bien que certaines fonctionnalités comme une vérification chiffrée ou une reconstruction graphique ne soient pas dans le scope initial, ces éléments sont mentionnés comme potentiellement futurs. **SPEAKER_02** confirme la durée du projet (entre septembre et janvier 2025) sans autre précision sur les projets ultérieurs présentés ensuite.

### 7. Analyse des cas d’usage de synthèse générative et orchestration de données internes  _(00:02:33 → 00:02:55)_

Lors de cette discussion, **Speaker_02** aborde les deux axes principaux de la synthèse générative : l’intégration de sources hétérogènes (rapports, événements) pour faciliter leur interprétation. Il souligne ensuite la question réciproque liée à l’orchestration des données via un système centralisé comme le MCP (*Modular Composition Platform*), en évoquant des travaux manquants sur cette dimension. **Speaker_01** présente deux projets internes :

**Actions :**

- développer une orchestrateur N8N pour un outil de rédaction assistée (réponses d’appels d’offres) avec plusieurs agents spécialisés (CV, références, réponses AO), modulaire et interrogeable via des règles _(resp. Speaker_01)_ — échéance : non spécifiée
- finaliser l’implémentation d’un outil de génération de synthèse de réunion (chatbot interactif), en intégrant des outils internes comme le MCP pour accéder aux notes du Sherpa et exporter les synthèses _(resp. Speaker_01)_ — échéance : non spécifiée

### 8. Amélioration des synthèses, souveraineté technique et intégration multi-outils pour la planification  _(00:02:55 → 00:03:05)_

SPARKER_01 propose d'augmenter les synthèses existantes et de les adapter à un contexte spécifique en intégrant davantage de propositions. SPARKER_00 souligne l'importance de cloisonner l'utilisation d'un outil pour préserver une souveraineté technique, notamment au sein de RTE, afin qu'il reste limité à cet environnement. SPARKER_01 évoque deux missions d'orchestration distinctes (orchestration et planification via un agent de POG) et mentionne un projet en cours impliquant un outil web Sir pour la récupération bibliographique. Il suggère aussi développer une O*Dmap automatisée pour proposer des PoC (Proof of Concepts) rapides, combinant plusieurs outils de simulation dans une pipeline ou workflow type « twend ».

**Actions :**

- Récupérer de la bibliographie à partir d’un sujet donné en utilisant l’outil web Sir _(resp. SPARKER_01)_ — échéance : non précisée
- Développer une O*Dmap automatisée pour proposer des PoC minimaux sur un sujet donné, combinant plusieurs outils de simulation _(resp. SPARKER_01 (ou équipe à déterminer))_ — échéance : non précisée

### 9. Présentation des projets de recherche en traitement automatique de documents et modélisation VLM  _(00:03:05 → 00:03:25)_

Lors de cette discussion, **SPEAKER_00** explique l’intérêt d’intégrer une veille sur les avancées en recherche pour alimenter des démonstrateurs rapides, notamment via des sources comme duRKEVX. **Maya (SPEAKER_01)** présente un projet spécifique : extraction structurée de données administratives scannées (salaires, métiers) à partir de documents, avec une approche basée sur un modèle VLM pour détecter la conformité et les drifts de performance. Ce projet, initialement mené en local, inclut des étapes clés comme le prétraitement, l’amélioration fine du modèle VLM, le contrôle d’anomalies et le déploiement via un dashboard UVCorn. **SPEAKER_02** demande si les documents étaient manuscrits ou non, précisant qu’une classification initiale était nécessaire entre écriture manuscrite ou non.

### 10. Analyse des cas d’usage NLP, MCP et orchestration dans le cadre des études énergétiques  _(00:03:25 → 00:03:55)_

Lors de cette discussion, **Speaker_01**, **Speaker_00** et **Speaker_02** ont exploré les liens entre les cas d’usage liés à la synthèse de données manuscrites (NLP), l’architecture MCP (Moyens de Production Centralisés) et l’orchestration des études énergétiques. Les participants ont souligné que ces domaines partagent des besoins communs, notamment dans la gestion de scénarios complexes, la récupération d’hypothèses ou de données fragmentées (blocs d’alarme, messages), et l’interaction avec des outils internes de simulation ou de génération de rapports. **Speaker_02** a évoqué des pistes spécifiques comme la synthèse de notes manuscrites en blocs structurés et le rôle du MCP dans l’amont des études pour préparer les simulations avant leur validation humaine. **Speaker_01** a rappelé que ces travaux s’inscrivent dans une feuille de route incluant un POI (Prototype d’Outils Intégrés) pour brancher directement sur des outils internes via le MCP, tandis que **Speaker_00** a mentionné l’importance d’une matrice d’études à mener avant la finalisation du pré-rapport.

### 11. Orientation des travaux sur le cas d’usage MCP et proposition d’un atelier UX  _(00:03:55 → 00:04:26)_

SPÉAKER_00 propose de réfléchir à un cas d’usage centré sur le **MCP** (probablement une technologie ou un module spécifique) en s’appuyant sur la présentation des besoins du partenaire externe. SPÉAKER_02 souligne l’absence actuelle de besoins précis et suggère de revenir vers ce dernier pour affiner les attentes avant toute discussion approfondie. Par ailleurs, SPÉAKER_00 partage une information interne concernant un partenariat avec des cabinets UX pour la **contrôle room** du futur projet, incluant une proposition d’un atelier UX pour prioriser des cas d’usage et partager leur vision.

**Actions :**

- Tenir informé SPÉAKER_02 si des besoins plus précis émergent concernant le MCP ou les travaux en NLP, sans action immédiate _(resp. SPÉAKER_02 (ou l’équipe concernée))_ — échéance : À déterminer ultérieurement
- Préparer une réflexion sur un cas d’usage MCP en collaboration avec SPÉAKER_00, si ce dernier présente des besoins clairs _(resp. SPÉAKER_02 (ou son équipe))_ — échéance : Non précisée
- Organiser un atelier UX pour prioriser les cas d’usage avec SPÉAKER_00 et ses partenaires _(resp. SPÉAKER_00 (ou l’équipe GALËT))_ — échéance : Non précisée

## 3. Décisions

_Aucune décision formellement prise._

## 4. Plan d'attaque — Prochaines actions

| # | Sujet | Action | Responsable | Échéance |
|---|-------|--------|-------------|----------|
| 1 | Automatisation de la rédaction des rapports d'études EOD – Pipeline et structure des sections | Établir et affiner les règles d’assignation des slides aux sections du rapport en fonction des descriptions préétablies (ex : exclusion des données non pertinentes pour l’introduction ou la section hypothèses de coût) | PPDT | non spécifiée |
| 2 | Automatisation de la rédaction des rapports d'études EOD – Pipeline et structure des sections | Utiliser les fichiers PPT comme entrée principale pour générer une chaîne de traitement automatisée (OCR + analyse graphique + classification des slides par section) via un pipeline en cinq étapes avec versionning des artefacts intermédiaires | deRTE | non spécifiée |
| 3 | Automatisation de la rédaction des rapports d'études EOD – Pipeline et structure des sections | Tester l’option d’ajouter une transcription vocale (format TXT) issue de l’enregistrement d’un modèle comme Spitch pour enrichir le contexte des slides non textuelles | deRTE | non spécifiée |
| 4 | Validation du POG (Prototype d'Outil de Génération) pour la catégorisation et l’assignation des slides dans les rapports | évaluer avec Gérald si des suites sont nécessaires pour généraliser cette approche de génération de rapports | Gérald | non spécifiée |
| 5 | Validation du POG (Prototype d'Outil de Génération) pour la catégorisation et l’assignation des slides dans les rapports | Basile propose d’explorer l’adaptation du POG pour gérer plusieurs présentations sans template initial, en intégrant une logique de sélection manuelle des templates par un agent | Basile (à discuter avec les équipes concernées) | non spécifiée |
| 6 | Analyse des cas d’usage de synthèse générative et orchestration de données internes | développer une orchestrateur N8N pour un outil de rédaction assistée (réponses d’appels d’offres) avec plusieurs agents spécialisés (CV, références, réponses AO), modulaire et interrogeable via des règles | Speaker_01 | non spécifiée |
| 7 | Analyse des cas d’usage de synthèse générative et orchestration de données internes | finaliser l’implémentation d’un outil de génération de synthèse de réunion (chatbot interactif), en intégrant des outils internes comme le MCP pour accéder aux notes du Sherpa et exporter les synthèses | Speaker_01 | non spécifiée |
| 8 | Amélioration des synthèses, souveraineté technique et intégration multi-outils pour la planification | Récupérer de la bibliographie à partir d’un sujet donné en utilisant l’outil web Sir | SPARKER_01 | non précisée |
| 9 | Amélioration des synthèses, souveraineté technique et intégration multi-outils pour la planification | Développer une O*Dmap automatisée pour proposer des PoC minimaux sur un sujet donné, combinant plusieurs outils de simulation | SPARKER_01 (ou équipe à déterminer) | non précisée |
| 10 | Orientation des travaux sur le cas d’usage MCP et proposition d’un atelier UX | Tenir informé SPÉAKER_02 si des besoins plus précis émergent concernant le MCP ou les travaux en NLP, sans action immédiate | SPÉAKER_02 (ou l’équipe concernée) | À déterminer ultérieurement |
| 11 | Orientation des travaux sur le cas d’usage MCP et proposition d’un atelier UX | Préparer une réflexion sur un cas d’usage MCP en collaboration avec SPÉAKER_00, si ce dernier présente des besoins clairs | SPÉAKER_02 (ou son équipe) | Non précisée |
| 12 | Orientation des travaux sur le cas d’usage MCP et proposition d’un atelier UX | Organiser un atelier UX pour prioriser les cas d’usage avec SPÉAKER_00 et ses partenaires | SPÉAKER_00 (ou l’équipe GALËT) | Non précisée |
