# PER2023--003
Ce dépôt contient l'ensemble des fichiers nécessaires pour faire rouler de manière autonome un véhicule de type MUSHR.

## Installation de l'environnement MUSHR
Afin d'installer le container nécessaire pour faire rouler une voiture de type MUSHR, suivez ce [tutoriel](https://anr-multitrans.github.io/Robot_MuSHR/).
Une fois le container créé, remplacer le fichier catkin/src par celui présent dans ce dépôt.

## Les fonctionnalités
Pour faire fonctionner l'arrêt d'urgence et le changement de modèle il faut lancer au préalable un broker MQTT.  
Pour ce faire, lancer sur un terminal au niveau du dossier mosquitto, la commande: `mosquitto.exe -c mosquitto.conf -v  `
Attention à écrire `allow_anonymous=true` dans le fichier mosquitto.conf. Il est possible qu'il faille désactiver la protection réseau dans le pare-feu.  
Ensuite il suffit d'écrire l'adresse IP du broker MQTT dans la création du client dans le fichier Follow_Road.py et dans la méthode publish_mqtt du fichier joy_teleop.py.
Pour lancer le robot, il suffit de lancer les commandes `roslaunch mushr_base teleop.launch` et `roslaunch imredd_pkg follow_road.launch` dans deux terminaux distincts.

A partir de ce moment la vous pourrez controler le robot en restant appuyer sur la gachette L1 de la manette et en bougeant les joysticks. Pour la conduite autonome il suffira de rester appuyer sur la gachette R1.
En appuyant sur les touches rond, triangle et carré vous pourrez changer entre les modèles 1, 2 et 3 (voir chargement des modèles). En appuyant sur la touche croix vous arrêterez toutes les voitures écoutant le broker MQTT.

## Charger les modèles
Pour charger les modèles il suffit d'aller dans le fichier src/Imreddpkg/followRoadNode.py, de rajouter la classe décrivant l'architecture du réseau de neurone et de charger les poids du réseau dans la classe FollowRoad. De base trois réseaux sont déjà chargés sur le robot et il suffit de changer la variable model au début du fichier pour changer rapidement de modèle. Les trois modèles d'origines sont ceux associés aux touches rond triangle et carré.
