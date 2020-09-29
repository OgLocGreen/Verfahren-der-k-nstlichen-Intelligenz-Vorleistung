import numpy
# scipy.special for the sigmoid function expit()
import scipy
import matplotlib.pyplot
import imageio
import cv2
import os
import glob
import csv
import time
import scipy
from scipy import special
import math

# Klasse KünstlichesNeuronalesNetz Definieren
class NeuronalesNetz:
    
    # Initalisierung des Netzes Vornehmmen mit folgenden Argumenten 
    def __init__(self, EingangsNeuronen, VersteckteNeuronen, AusgangsNeuronenen, Lernrate):
        # Übergeben der Arumente als Parameter
        self.eNeuronen = EingangsNeuronen
        self.vNeuronen = VersteckteNeuronen
        self.aNeuronen = AusgangsNeuronenen
        self.vBias = VersteckteNeuronen
        self.aBias = AusgangsNeuronenen
        self.lr = Lernrate

        # Gewichtewichtsmatix anlegen, für wih, who
        # Gewichte in den Arrays, es ist immer von Neuron i zu Neuron j der nächsten Schicht
        # w11 w21
        # w12 w22 usw. 
        # Normalverteilung um 0 mit einer Standardabweichung von 0.5 am schluss ist die größe der Arrays
        self.gewichteVersteckte = numpy.random.normal(0.0, 0.005, (self.vNeuronen, self.eNeuronen))
        self.gewichteAusgang = numpy.random.normal(0.0, 0.005, (self.aNeuronen, self.vNeuronen))        
    	self.vBias = numpy.random.normal(0.0, 0.005,  self.vNeuronen)
    	self.aBias = numpy.random.normal(0.0, 0.005,  self.AusgangsNeuronenen)


        # Aktiverungsfunktion ist die Sigmoidfunktion 
        # leider konnte die in der Vorlesung besproche 1/(1+np.exp(-x)) Formel nicht verwendet werden
        # da es sonst zu einem Overflow kommt weshalb eine Sigmoid aus Scipy importiert wurde
        # scipy.special.expit(x) =  1/(1+np.exp(-X))
        self.aktivierungs_funktion = lambda x: scipy.special.expit(x)
        pass

    # Trainieren des KünstlichenNeuornalenNetzes
    def training(self, Eingabe_liste, Ziel_list):

        # Vorwärtsdurchgang

        # Eingabe in 2-Dimensionale Arrays formatieren
        eingaben = numpy.array(Eingabe_liste, ndmin=2).T
        ziele = numpy.array(Ziel_list, ndmin=2).T
        
        # berechnen der Signale von Eingangs- zur Verstecktenschicht
        # Numpy.dot() ist das Punktprodukt der beiden Arrays
        versteckte_eingang = numpy.dot(self.gewichteVersteckte, eingaben)
        versteckte_eingang_2 = numpy.dot(self.vBias, versteckte_eingang)
        # berechnen der Signale von Versteckten-eingang zum Versteckten-ausgang
        versteckte_ausgang = self.aktivierungs_funktion(versteckte_eingang_2)
        
        # berechnen der Signale von Versteckterschichtausgang zu Ausgabeschichteingang
        ausgang_eingang = numpy.dot(self.gewichteAusgang, versteckte_ausgang)
        ausgang_eingang_2 = numpy.dot(self.aBias, ausgang_eingang)

        # calculate the signals emerging from final output layer
        ausgang_ausgang = self.aktivierungs_funktion(ausgang_eingang_2)
        
        #Berechnen der Fehler

        # Fehler der AusgangsSchicht berechnen (ziel - wirklichesErgebniss)
        ausgang_fehler = ziele - ausgang_ausgang
        # Anteiligen Fehler der VerstecktenSchicht berechnen 
        versteckte_fehler = numpy.dot(self.gewichteAusgang.T, ausgang_fehler) 
        
        # Anpassen der Gewicht je nach Fehler

        # Anpassen der Gechte der Ausgangsschicht
        self.gewichteAusgang += self.lr * numpy.dot((ausgang_fehler * ausgang_ausgang * (1.0 - ausgang_ausgang)), numpy.transpose(versteckte_ausgang))
        self.aBias += self.lr * numpy.dot((ausgang_fehler * ausgang_ausgang * (1.0 - ausgang_ausgang)), numpy.transpose(versteckte_ausgang))
        
        # Anpassen der Gewichte der VerstecktenSchicht
        self.gewichteVersteckte += self.lr * numpy.dot((versteckte_fehler * versteckte_ausgang * (1.0 - versteckte_ausgang)), numpy.transpose(eingaben))
        self.vBias += self.lr * numpy.dot((versteckte_fehler * versteckte_ausgang * (1.0 - versteckte_ausgang)), numpy.transpose(eingaben))
        pass

    
    # Testen des KünstlichenNeuronalenNetzes
    def testen(self, Eingabe_liste):
        # Eingabe in 2-Dimensionale Arrays formatieren
        eingaben = numpy.array(Eingabe_liste, ndmin=2).T

        # berechnen der Signale von Eingangs- zur Verstecktenschicht
        # Numpy.dot() ist das Punktprodukt der beiden Arrays
        versteckte_eingang = numpy.dot(self.gewichteVersteckte, eingaben)
        # berechnen der Signale von Versteckten-eingang zum Versteckten-ausgang
        versteckte_ausgang = self.aktivierungs_funktion(versteckte_eingang)
        
        # berechnen der Signale von Versteckterschichtausgang zu Ausgabeschichteingang
        ausgang_eingang = numpy.dot(self.gewichteAusgang, versteckte_ausgang)
        # calculate the signals emerging from final output layer
        ausgang_ausgang = self.aktivierungs_funktion(ausgang_eingang)

        return ausgang_ausgang

            
def daten_laden(datenset):
    trainings_liste_pfade = []
    test_liste_pfade = []
    ordner_liste = os.listdir("./dataset/"+datenset)
    for klassen_ordner in ordner_liste:
        if klassen_ordner == "fork":
            klassen_nummer = 0
        elif klassen_ordner == "spoon":
            klassen_nummer = 1
        elif klassen_ordner == "knife":
            klassen_nummer = 2
        bilder_liste = glob.glob("./dataset/"+datenset+"/"+klassen_ordner+"/"+"*.png")
        #bilder_liste = glob.glob("./dataset/dataset_blackandwhite"+"/"+klassen_ordner+"/"+"*.png")
        i = 0
        for line in bilder_liste:
            string = str(klassen_nummer)+","+line[:-4]+",\n"
            if i % 3:
                trainings_liste_pfade.append(string)
            else:
                test_liste_pfade.append(string)
            i += 1
    return trainings_liste_pfade, test_liste_pfade


if __name__ == "__main__":
    #datenset = "dataset_real_simple"
    datenset = "dataset_blackandwhite"
    trainings_data_liste, test_data_liste = daten_laden(datenset)

    # Definieren der größe des Netzes
    eingangs_neuronen = 40000 # (40000 = 200x200 )
    versteckte_neuronen = 8000
    ausgangs_neuronen = 3 # da wir dreiklassen besitzen

    # Lernrate
    lernrate = 0.01

    # Aufösung für die Bilder berechnen 
    auflösung = int(math.sqrt(eingangs_neuronen))

    # create instance of neural network
    n = NeuronalesNetz(eingangs_neuronen,versteckte_neuronen,ausgangs_neuronen, lernrate)

    # epochen ist die Anzahl wie oft das Netz auf den Trainingsdatensatztrainiert wird
    epochs = 25
    print("Training starten")
    for e in range(epochs):
        # einzeln durch alle Einträge der Trainigns
        for Einträge in trainings_data_liste:
            # alle Einträge am "," aufteilen
            alle_Einträge = Einträge.split(",")
            # größe der Bilderanpassen wenn nötig und von 2D in einen 1D Arraytransformieren
            eingabe = cv2.imread(alle_Einträge[1]+".png",0)
            eingabe = cv2.resize(eingabe,(auflösung,auflösung))
            eingabe = eingabe.reshape(eingangs_neuronen)
            # create the target output values (all 0.01, except the desired label which is 0.99)
            ziele = numpy.zeros(ausgangs_neuronen) + 0.01
            # all_values[0] is the target label for this record
            ziele[int(alle_Einträge[0])] = 0.99
            n.training(eingabe, ziele)
            pass
        print("Epochen:",e)

    # Testen des Neuronalen Netzes
    # erstellen eines Ergebnissarrays
    ergebniss_array = []

    # durch alle Einträge im testdatensatz gehen
    for Einträge in test_data_liste:
        # alle Einträge am "," aufteilen
        alle_Einträge = Einträge.split(',')
        # das richtige Ergebniss steht im ersten Ersteneintrag
        richtes_label = int(alle_Einträge[0])
        # scale and shift the inputs
        eingabe = cv2.imread(alle_Einträge[1]+".png",0)
        eingabe = cv2.resize(eingabe,(auflösung,auflösung))
        eingabe = eingabe.reshape(eingangs_neuronen)
        
        # Testen des Netzes
        Ausgabe = n.testen(eingabe)
        # der eintrag mit dem größen Wert ist das Ergebniss
        Ausgabe_label = numpy.argmax(Ausgabe)
        # anfügen einer 0 für falsche Ausgabe und eine 1 bei richtiger Ausgabe
        if (Ausgabe_label == richtes_label):
            ergebniss_array.append(1)
        else:
            ergebniss_array.append(0)
            pass
        pass

    # Berechnen wie oft das Netz richtig lag = Genauigkeit
    ergebniss_array = numpy.asarray(ergebniss_array)
    print ("Genuigkeit = ", ergebniss_array.sum() / ergebniss_array.size)

    # kann auch auf eigenen Bildern getestet werden 
    # normalerweise deaktiviert, wenn sie es testen wollen 
    # hier auf True stellen und Bild mit dem Namen "test.png" in den Ordner einfügen
    example = True

    if example == True:
        test_img = cv2.imread("test.png",0)
        test_img = cv2.resize(test_img,(200,200))
        img_data = test_img.reshape(eingangs_neuronen)
        cv2.imshow("img_img",test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        Ausgabe = n.testen(img_data)
        label = numpy.argmax(Ausgabe)
        label_lookup = {0:"Gabel",1:"Löffel",2:"Messer"}
        print("network says ", label_lookup[label])
    pass