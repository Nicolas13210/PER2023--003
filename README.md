# PER2023--003
Ce dépôt contient l'ensemble des fichiers nécessaires pour faire rouler de manière autonome un véhicule de type MUSHR.
## Installation de l'environnement MUSHR
Afin d'installer le container nécessaire pour faire rouler une voiture de type MUSHR, suivez ce [tutoriel](https://anr-multitrans.github.io/Robot_MuSHR/).
Une fois le container créé, remplacer le fichier catkin/src par celui présent dans ce dépôt.
## Les fonctionnalités
Pour faire fonctionner l'arrêt d'urgence et le changement de modèle il faut lancer au préalable un broker MQTT.
Pour ce faire, nous avons lancé sur un terminal au niveau du dossier mosquitto, la commande: mosquitto.exe -c mosquitto.conf -v 
Ensuite il suffit d'écrire l'adresse IP du broker MQTT dans le client de Follow_Road.py et dans la méthode publish_mqtt de joy_teleop.py.
