# Compte rendu de réunion

*Source : `dicte_audio_3.normalized.txt`*

## 1. Executive Summary

La réunion a débuté par un tour de table entre **SPEAKER_00** (DataIa Manager) et ses collaborateurs (**Maya**, **Della**, **Dinka**) pour identifier les besoins en IA générative au sein d’Élé Consulting, tandis que **SPEAKER_02** (Bruno), pilote de la feuille de route Smart Cockpit, a souligné l’intérêt croissant pour des interfaces conversationnelles fluides et des solutions techniques comme les LLM ou MCP. Après une exploration informelle des compétences en IA générative (**SPEAKER_01**) – incluant la transformation multimodale, le fine-tuning local et l’orchestration de modèles –, les échanges se sont concentrés sur l’automatisation de rapports (ex : génération de fiches synthétiques à partir de PowerPoint) et des ajustements méthodologiques pour une meilleure assignation des slides. La discussion a aussi abordé la validation d’un outil multi-templates avec garde-fous, ainsi que des projets comme la détection de fraudes via des VLM ou l’orchestration de données énergétiques à travers le MCP (Machine Processed Content), soulignant la nécessité d’une souveraineté technique et d’une traçabilité robuste. Aucune décision formelle n’a été prise pour avancer sur ces pistes, mais des pistes techniques et collaboratives ont émergé sans engagement explicite.

## 2. Sujets abordés

### 1. Tour de table sur les besoins en IA générative et échanges entre R&D et pratique DataIa

SPEAKER_00 propose un tour de table pour échanger sur les besoins spécifiques en matière d’IA générative au sein de l’Élé Consulting. Il introduit ses rôles (DataIa Manager) et ses collaborations avec Maya, Della et Dinka, notamment sur des projets liés à la gouvernance, la régulation et les analyses métiers. SPEAKER_01 présente Maya comme consultante confirmée spécialisée en IA générative, ayant travaillé chez IVB et RTE, avec un doctorat en traitement automatique des langues. SPEAKER_02 (Bruno) explique son rôle à la RD : il pilote la feuille de route Smart Cockpit pour les assistants interactifs, notamment pour les opérateurs en santé, et travaille sur l’implémentation de réseaux de neurones. Il souligne aussi sa transition vers l’intelligence artificielle et l’innovation, avec une attention particulière aux LLM et MCP (modèles conversationnels par plateforme). Enfin, il évoque un intérêt croissant pour les interfaces conversationnelles fluides entre opérateurs et outils.

### 2. Prise de contact et exploration des besoins sans décision formelle

Dans cet extrait, **SPEAKER_02** évoque l'idée d'appeler ou consulter les services appropriés pour identifier des briques techniques (calculs, études, réseaux) afin d'explorer des pistes. Il souligne qu'actuellement, il n'existe pas de mission claire définie et que le but serait de faire connaissance avec **Mike** en lui présentant les travaux réalisés par **SPEAKER_01**. **SPEAKER_01** propose alors de présenter ses projets pour établir un échange informel. Aucune décision concrète ni engagement explicite n'a été pris.

### 3. Présentation des compétences en IA générative chez Ely

SPEAKER_01 présente les projets et compétences techniques sur lesquels elle travaille ou a travaillé dans le domaine de l'intelligence artificielle générative, notamment la transformation de contenus complexes (documents multimodaux) en informations actionnables, comme des fiches synthétiques. Elle évoque aussi des aspects d'industrialisation, de monitoring, et des outils spécifiques utilisés (ex : QNTubine, Mistral). Les compétences incluent le fine-tuning local de modèles, la gestion de prétraitements, le RAG (Retrieval-Augmented Generation), la vectorisation, ainsi que l'orchestration et les déploiements d'IA (ex : VLHRM, Vicorne, MCP).

### 4. Automatisation de la rédaction des rapports d'études EOD – Pipeline et structure des fichiers PPT

SPEAKER_01 décrit le projet réalisé dans l’équipe équilibreoffre et demande, consistant à automatiser la génération de rapports exhaustifs sur la rentabilité des moyens de production. L’outil développé, inspiré d’Antares mais spécialisé en analyse de rentabilité, traite les fichiers PPT (PowerPoint) pour extraire des descriptions détaillées des slides via une pipeline en cinq étapes : analyse OCR des graphiques et textes, classification des slides par sections prédéfinies du rapport, assignement précis des éléments aux sections, standardisation de la structure et intégration d’une boucle de correction collaborative. SPEAKER_02 aborde ensuite les besoins complémentaires pour affiner la structure du sommaire (ex : descriptions détaillées des sections comme « France vs autres pays ») afin d’améliorer l’algorithme d’assignation des slides.

### 5. Validation et ajustements du processus de catégorisation et d’assignation dans les rapports générés par IA

Les participants (SPEAKER_00, SPEAKER_01, SPEAKER_02) discutent des hypothèses et règles définies pour la sélection et l’assignation des slides aux chapitres du rapport. SPEAKER_00 présente un prototype fonctionnel en 25 jours, incluant une boucle de correction itérative pour les bullet points et une validation avec Gérald. SPEAKER_01 détaille la méthodologie : assignation par sections, séparation front/arrière-plan dans la rédaction, et intégration des commentaires via un agent dédié. SPEAKER_02 pose une question sur l’adaptabilité du système sans template initial pour d’autres présentations.

### 6. Validation de l'approche technique pour un outil d'agent multi-templates avec garde-fous robustes

SPEAKER_01 et SPEAKER_00 discutent des adaptations possibles de l'outil initial pour intégrer plusieurs templates ou orienter dynamiquement la sélection des chapitres et du contenu. SPEAKER_01 souligne l'importance d'ajouter des garde-fous (robustesse, traçabilité des erreurs par section, artefacts intermédiaires) pour un POC automatisant une tâche de rédaction, même si ce n'est pas une mission longue. Il mentionne aussi les limites hors-scopes comme le *check-in chiffré* ou la reconstruction graphique depuis des données brutes, mais évoque leur potentiel futur avec des outils anti-hallucination. Aucune décision ni action explicite ne semble avoir été prise sur ces points lors de cet extrait.

### 7. Analyse des cas d’usage pour une synthèse avancée de données et événements via orchestration MCP

SPEAKER_02 souligne l’intérêt de se concentrer sur deux axes principaux : la synthèse générative (front-end) et la synthèse d’événements ou de données (back-end). Il interroge SPEAKER_01 sur les travaux existants autour du MCP pour une orchestration multi-briques, notamment dans des projets internes comme l’outil N8N pour la rédaction assistée. SPEAKER_01 détaille deux projets : un outil de génération de réponses à appels d’offres (orchestré via N8N avec plusieurs agents spécialisés) et un projet en cours de synthèse de réunions, où il explore l’intégration d’un MCP pour interroger des outils internes ou externes (comme IPI). SPEAKER_02 évoque la souveraineté technique et les gains potentiels en confidentialité/précision avec ces solutions internes spécialisées.

### 8. Amélioration des synthèses et intégration d’une pipeline automatisée pour projets ID

Le SPEAKER_01 propose d’améliorer les synthèses existantes en les rendant plus complètes et mieux adaptées au contexte, avec une attention particulière à la souveraineté technique de l’outil par rapport aux environnements comme celui de RTE. Il évoque aussi un projet parallèle (agent de planification de PoC) où l’orchestration n’est pas encore centralisée. Pour les missions d’orchestration et de planification, il suggère développer une pipeline automatisée permettant de récupérer des bibliothèques à partir d’un sujet donné et de proposer une roadmap (MCP) pour un projet, en intégrant plusieurs outils de simulation. Cela vise notamment à accélérer la mise en œuvre lors d’événements comme les hackathons au sein de l’ID.

### 9. Présentation de projets liés à l'extraction d'informations et au contrôle de conformité via des modèles VLM

SPEAKER_01 présente un projet spécifique sur la détection de fraudes dans les documents administratifs scannés (salaire, métier) en extraisant des données structurées à partir de documents potentiellement manuscrits. Ce projet repose sur une chaîne de prétraitement, apprentissage d’un modèle visuo-linguistique (VLM), contrôle d’anomalies et déploiement avec un dashboard pour surveiller les drifts de performance. SPEAKER_01 évoque aussi des travaux en filtre-toning pour améliorer la qualité des données entrées, notamment dans le contexte multilingue (arabe). SPEAKER_00 demande si des questions surgissent après ces présentations.

### 10. Analyse des cas d’usage NLP, MCP et orchestration dans le cadre des rapports énergétiques

Les participants (SPEAKER_01, SPEAKER_00, SPEAKER_02) ont discuté des connexions entre les travaux existants en traitement du langage naturel (NLP), la gestion de cas d’usage spécifiques à la MCP (Machine Processed Content), et l’orchestration des études énergétiques. SPEAKER_02 a évoqué des pistes comme la synthèse de données fragmentées (notes manuscrites, blocs d’alarme) ou la navigation entre hypothèses et outils de simulation pour préparer des pré-rapports robustes. SPEAKER_01 a confirmé l’importance du MCP dans leur feuille de route pour orchestrer les simulations et relier agents et outils internes. SPEAKER_00 a rappelé le besoin d’une matrice d’études à finaliser, avec un rôle centralisé pour le MCP dans la préparation des scénarios avant validation humaine.

### 11. Orientation des échanges sur le cas d’usage MCP et proposition d’atelier UX

Les participants, **SPEAKER_00** et **SPEAKER_02**, discutent de deux pistes : la réflexion sur un cas d’usage centré sur l’intégration du *MCP* (si présenté par le partenaire externe), avec une suggestion de retour vers eux en cas de besoin précis. **SPEAKER_00** propose également un atelier UX pour prioriser des cas d’usage, s’appuyant sur les collaborations existantes avec des cabinets comme celui d’Olivier Maserol. Aucune décision formelle n’est prise, et aucune action concrète n’est engagée.

## 3. Décisions

_Aucune décision formellement prise._

## 4. Plan d'attaque — Prochaines actions

_Aucune action définie._
