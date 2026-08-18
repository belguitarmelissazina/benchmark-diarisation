# Compte rendu de réunion

*Source : `dicte_audio_3.normalized.txt`*

## 1. Executive Summary

Cette réunion vise à établir une collaboration structurée autour des enjeux liés à l’intelligence artificielle générative et ses applications opérationnelles au sein de RTE. Les participants, experts en Data & IA (comme Nordine ou Maya) ainsi que spécialistes des systèmes interactifs pour les opérateurs (Bruno Mélière, Dinka), explorent d’abord un besoin flou de prise de contact pour mieux identifier leurs attentes et compétences communes, sans encore définir de mission précise. L’accent est mis sur l’exploration des interfaces conversationnelles fluides entre humains et outils, notamment via des modèles comme les LLM ou MCP, afin d’améliorer la gestion des tâches techniques (études énergétiques, orchestration de projets) ou la production automatisée de documents synthétiques (rapports de rentabilité).

Si aucune décision claire n’a été prise pour orienter une mission spécifique, l’objectif immédiat reste de clarifier les attentes et d’affiner un cadre commun pour des échanges futurs, notamment en croisant les expertises sur des cas concrets comme la gestion des congestions ou l’automatisation partielle des synthèses de données. La réunion marque ainsi une étape préliminaire vers une collaboration plus ciblée, sans engagement immédiat sur des livrables définis.

## 2. Sujets abordés

### 1. Présentation des participants et discussion préliminaire sur les besoins en IA générative

Lors de cette réunion, **Nordine** (DataIa Manager chez Élé Consulting) propose un tour de table pour présenter ses rôles : expert Data & IA, gestion de la feuille de route équilibre offre/demande et analyses métiers. Il introduit ensuite **Mathieu**, **Maya** (consultante confirmée chez IVB spécialisée en IA générative avec un doctorat en traitement automatique des langues), puis **Bruno Mélière** (RD à RTE) qui pilote la feuille de route *Smart Cockpit* pour les assistants interactifs aux opérateurs de soins et de conduite, ainsi que **Dinka** (implémentation de réseaux de neurones). Enfin, **Jérôme Picot**, anciennement au département d’exploitation puis de marche chez RTE, partage son expérience en recherche appliquée sur l’interface homme-machine, notamment autour des LLM et MCP pour des interactions quasi conversationnelles fluides.

### 2. Prise de contact et exploration des besoins pour une collaboration future

Dans cet extrait, **SPEAKER_02** évoque l'idée d'appeler ou de consulter les services appropriés pour étudier des éléments comme les calculs, les réseaux ou encore les études en tant que briques à explorer. Il souligne qu’à ce stade, aucune mission claire n’a été définie et propose une approche basée sur la prise de connaissance mutuelle avec **Dandine**, qui aurait mentionné travailler dans un contexte similaire et maîtriser ces sujets. **SPEAKER_02** précise que l’objectif actuel est simplement d’échanger pour mieux comprendre les compétences ou expériences potentielles de **Mike** avant d’envisager des actions concrètes. Enfin, il annonce revenir avec une formulation plus précise du besoin en cours de réflexion.

### 3. Présentation des projets d'intelligence artificielle générative chez Ely et compétences clés

SPEAKER_01 a présenté les projets d’intelligence artificielle générative menés chez **Ely**, en se concentrant sur ses expériences passées et celles depuis son arrivée. Il a détaillé une approche globale visant à transformer des contenus complexes (documents hétérogènes) en formats actionnables, comme des fiches synthétiques ou extraits pertinents pour faciliter les décisions. Ses compétences incluent notamment la gestion de modèles multimodaux, l’industrialisation avec des outils de prétraitement et de monitoring, ainsi que le déploiement localisé de modèles finetunés (ex : QNTubine). Il a aussi évoqué l’utilisation de plateformes comme **Mistral** et des techniques comme le RAG (Retrieval-Augmented Generation), la vectorisation et l’orchestration. Les projets couvrent des phases allant du NLP pur à des solutions plus opérationnelles comme MCP ou VLHRM, avec des livraisons en temps réel via des outils de streaming.

### 4. Automatisation de la rédaction des rapports d'études de rentabilité des moyens de production

Lors de cette réunion, **deRTE** a présenté le projet mené dans l’équipe équilibreoffre et demande pour automatiser la génération de rapports exhaustifs à partir des fichiers PPT contenant des analyses de rentabilité. L’objectif était d’accélérer leur production sans altérer la rigueur en ancrant les affirmations générées dans les slides correspondantes, tout en standardisant la structure des sections préétablies (ex : hypothèses, comparaisons France/Europe). **Général** a souligné l’importance de préciser davantage cette structure de rapport pour améliorer le fonctionnement du modèle. La solution repose sur une pipeline composée de cinq briques : analyse légère des slides via OCR et traitement graphique, assignement automatique des slides aux sections prédéfinies (avec règles spécifiques comme exclure les comparaisons France/Europe hors hypothèses), et intégration d’une transcription vocale pour enrichir le contexte. Aucune décision prise concernant la validation ou l’extension de ces éléments.

### 5. Stratégie de catégorisation et d’assignation pour les rapports via un POG (Proof of Concept)

Coût a présenté une méthodologie où la sélection et l’assignation des éléments sont regroupées sous la catégorie de la catégorisation, en s’appuyant sur deux templates identiques : l’un pour la structure des chapitres et l’autre pour les catégories à assigner. Basile a détaillé que cette phase repose sur des règles d’assignement précises incluant des descriptions métiers des sections et une analyse globale des slides pour éviter des erreurs de classification (ex. : placer des résultats dans l’introduction). L’outil, développé en 25 jours y compris les ateliers, vise à démontrer la valeur ajoutée de l’IA par rapport à la génération de rapports. Basile a expliqué que la rédaction se fait ensuite section par section, séparant d’abord le fond (bullet points) de la forme avant intégration finale. Une boucle de correction itérative permet d’ajuster les contenus en fonction des commentaires, et Gérald sera consulté pour évaluer les suites à ce travail.

### 6. Robustesse et gestion des artefacts dans un outil d'automatisation de rédaction multi-templates

Lors du échange entre **SPÉAKER_01** (équipe ED) et **SPÉAKER_02**, il est question de la robustesse technique d’un outil conçu pour générer des rapports à partir de différents templates. **SPÉAKER_00** souligne que l’outil permet de tester plusieurs hypothèses avec des templates distincts, offrant ainsi une flexibilité dans le choix des chapitres et leur contenu. **SPÉAKER_01** insiste sur la nécessité d’intégrer des garde-fous pour gérer les erreurs en séparant la rédaction par sections et en attribuant chaque slide à une section spécifique, afin de garantir une traçabilité claire des corrections. Il mentionne également l’utilisation d’artéfacts intermédiaires (comme des commentaires ou versions corrigées) pour faciliter la validation humaine et itérative du rapport.

### 7. Cas d’usage de synthèse générative et orchestration d’agents pour la gestion de réunions

Lors de cette discussion, **Speaker_02** aborde les deux cas d’usage principaux autour de la synthèse générative : l’intégration de données hétérogènes (comme des rapports ou événements) pour faciliter leur analyse. Il souligne aussi la nécessité d’explorer davantage le sens inverse de la synthèse, notamment en orchestrant plusieurs outils spécialisés (**MCP**), sans préciser si ces travaux ont déjà été menés explicitement dans ce contexte.

### 8. Amélioration des synthèses et propositions pour l’orchestration et la planification des projets

Lors du échange, **SPEAKER_01** propose d’améliorer les synthèses existantes en les rendant plus complètes et mieux adaptées au contexte actuel. **SPEAKER_00** insiste sur l’importance de cloisonner l’utilisation d’un outil pour préserver la souveraineté des données côté RTE, afin qu’elles ne fuient pas vers d’autres environnements comme Gaillé. Concernant les missions d’orchestration et de planification (notamment dans le cadre de projets comme celui de planification des PoK), **SPEAKER_01** évoque un projet en cours où l’accès à un outil web permet déjà une récupération automatique de bibliographie, mais sans implémentation complète du MCP. Il suggère d’étendre cette approche pour intégrer plusieurs outils de simulation et créer une pipeline automatisée (type *twind*), facilitant ainsi la génération rapide d’un PoC minimal sur des sujets spécifiques, notamment lors d’événements comme les hackathons au sein de l’ID.

### 9. Présentation de projets liés à l'état de l'art, au traitement automatisé des documents administratifs et à la détection de conformité

Le **SPEAKER_00** explique que cette réunion vise à illustrer comment exploiter rapidement les avancées en recherche et développement (R&D) via des sources comme DRKEVX pour alimenter des démonstrations rapides, notamment dans le cadre d'agents codant avec un effet quasi instantané. Il aborde aussi brièvement la nécessité de réaliser l'ingénierie logicielle sous-jacente à ces outils.

### 10. Échanges sur les cas d’usage NLP, MCP et orchestration des études énergétiques

Lors de cette discussion, **Noën** (SPEAKER_01) et **Gérald** (SPEAKER_00) évoquent les pistes explorées pour intégrer des cas d’usage liés à la synthèse de données fragmentées (notes manuscrites, blocs d’alarme, messages) via des outils de traitement du langage naturel (NLP). Ils soulignent notamment l’intérêt potentiel pour la partie **MCP** (Management of Complex Projects), en particulier pour orchestrer les simulations énergétiques et préparer des pré-rapports validés par des comités. **Noën** précise que cette approche s’inscrit dans leur feuille de route, visant à connecter un premier agent générant des directives avec des outils internes de simulation via le MCP, afin d’automatiser partiellement la gestion des études et scénarios avant validation humaine. **Gérald** ajoute que ce travail pourrait aussi s’appliquer à d’autres contextes comme l’orchestration de données techniques (consumption, production, programmation) pour ajuster dynamiquement les hypothèses ou scénarios en temps réel.

### 11. Décisions et pistes de réflexion sur les cas d’usage MCP et NLP

Lors du échange, **SPARKER_00** propose une réflexion approfondie sur un cas d’usage orienté **MCP**, en sollicitant une présentation des besoins spécifiques par le partenaire concerné. **SPARKER_02** suggère de revenir vers ce dernier si des éléments précis émergent pour éviter une discussion sans fondement, tout en rappelant que des actions préalables sont déjà prévues pour explorer le MCP. Par ailleurs, **SPARKER_00** informe sur un partenariat avec l’UX design (notamment Olivier Maserol) pour la *contrôle room* future et propose un atelier dédié à la priorisation de cas d’usage, sans attente immédiate de réponse.

## 3. Décisions

_Aucune décision formellement prise._

## 4. Plan d'attaque — Prochaines actions

_Aucune action définie._

## 5. Recommandations consultant

_Ces recommandations sont générées par IA à partir de la matière de la réunion. Elles ne reflètent pas des engagements pris._

| # | Recommandation | Sujet lié | Horizon |
|---|----------------|-----------|---------|
| 1 | Lancer une **phase exploratoire rapide** (quick win) avec **Mathieu** et **Maya** pour identifier des templates ou workflows existants chez IVB en IA générative, puis les adapter à la structure des rapports d’études de rentabilité de RTE. Cela permettrait d’évaluer l’applicabilité directe des solutions NLP/IA déjà déployées (ex : QNTubine, RAG) pour structurer automatiquement ces documents sans réinventer la roue. | Présentation des projets d'IA générative chez Ely et compétences clés | court terme |
| 2 | Créer un **POG (Proof of Concept) prioritaire** pour valider l’assignation automatique des slides aux sections prédéfinies, en s’appuyant sur les règles de catégorisation déjà définies par Basile. En parallèle, tester la robustesse de la pipeline d’OCR et de traitement graphique avec des lots de données réels (ex : 10 rapports PPT) pour identifier les artefacts critiques à corriger avant déploiement. Cela permettrait de réduire les risques lors du passage à l’échelle. | Stratégie de catégorisation et d’assignation pour les rapports via un POG | court terme |
| 3 | Développer une **solution hybride** combinant synthèse générative (via outils comme Mistral) et orchestration d’agents spécialisés (MCP) pour automatiser la planification des projets énergétiques. En s’appuyant sur l’expérience de **Jérôme Picot**, prioriser les cas où les données sont déjà structurées (ex : PoK, événements hackathons) pour générer des PoC minimaux en temps réel, puis étendre progressivement à des scénarios plus complexes (consumption/production dynamique). | Amélioration des synthèses et orchestration/planification des projets | moyen terme |
| 4 | Organiser un **atelier collaboratif** avec l’équipe UX (Olivier Maserol) pour identifier les besoins en interface homme-machine (contrôle room) autour de la gestion des artefacts générés par l’IA. En parallèle, tester l’intégration d’un outil de transcription vocale dans la pipeline actuelle pour enrichir les données contextuelles (ex : commentaires audio des slides), ce qui pourrait améliorer la traçabilité et réduire les itérations manuelles. | Robustesse et gestion des artefacts dans un outil d’automatisation multi-templates | moyen terme |
| 5 | Lancer une **étude comparative** avec **Dandine** (ou Mike) pour évaluer les compétences en traitement de données techniques (réseaux, calculs) chez des partenaires externes. Identifier des briques technologiques clés à externaliser ou mutualiser (ex : modules RAG vectorisés), tout en sécurisant la souveraineté des données via des environnements dédiés (ex : Gaillé). Cela permettrait d’alimenter une feuille de route priorisée pour les prochaines collaborations. | Prise de contact et exploration des besoins | moyen terme |
| 6 | Former un **comité projet dédié** (avec **Gérald**, **Noën** et **Bruno Mélière**) pour prioriser les cas d’usage MCP/NLP en fonction de leur impact opérationnel (ex : gestion des alarmes, simulations énergétiques). En parallèle, explorer l’intégration progressive de plateformes comme **Mistral** ou des outils open-source (ex : Twind) pour automatiser la génération de pré-rapports validés par comité, tout en gardant une phase manuelle itérative pour les corrections critiques. | Cas d’usage NLP, MCP et orchestration des études énergétiques | moyen terme |
