import intentii
import nltk
import random
import numpy as np
from tensorflow import keras

# pentru partea de UI
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.core.window import Window

# stemmer folosit pentru a lua fiecare cuvant dintr-un model si a-l reduce pentru antrenare
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# functiile pentru modelul retelei
def preluare_date():

    lista_intentii = intentii.lista_intentii

    # date de input
    cuvinte = []
    etichete = []
    cuv_tokenizate = []
    tag_cuv_tokenizate = []

    # loop-uri cu care se preiau toate modelele de input de la utilizator
    for intentie in range(len(lista_intentii)):
        for model in range(len(lista_intentii[intentie][1])):  # adica in dimensiunea listei cu modele de input
            cuvant = nltk.word_tokenize(lista_intentii[intentie][1][model])
            cuvinte.extend(cuvant)  # rezulta o lista cu toate cuvintele din modelele existente
            cuv_tokenizate.append(cuvant)  # lista de cuvinte tokenizate
            tag_cuv_tokenizate.append(lista_intentii[intentie][0])  # lista cu tag-ul fiecarui cuvant din lista de mai sus

        if lista_intentii[intentie][0] not in etichete:
            etichete.append(lista_intentii[intentie][0])

    cuvinte = [stemmer.stem(cuv.lower()) for cuv in cuvinte if cuv not in ["?", "!", "."]]
    cuvinte = sorted(list(set(cuvinte)))

    etichete = sorted(etichete)

    # one hot encoding
    # fiecare pozitie din lista reprezinta daca exista cuvantul sau nu
    # e un input bun pentru o retea neuronala

    antrenament = []
    output = []

    # one hot pt tag-uri
    out_empty = [0 for _ in range(len(lista_intentii))]
    for x, cuv_tok in enumerate(cuv_tokenizate):
        bag = []

        lista_aux = [stemmer.stem(cuv) for cuv in cuv_tok]

        for cuv in cuvinte:
            if cuv in lista_aux:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[etichete.index(tag_cuv_tokenizate[x])] = 1

        antrenament.append(bag)
        output.append(output_row)

    antrenament = np.array(antrenament)
    output = np.array(output)

    return lista_intentii, etichete, cuvinte, antrenament, output


# reteaua neuronala
def construire_retea(antrenament, output):
    retea = keras.Sequential([
        keras.layers.Input(shape=(len(antrenament[0]))),  # input layer
        keras.layers.Dense(8, activation="relu"),  # hidden layer(1) cu rectify linear unit
        keras.layers.Dense(8, activation="relu"),  # hidden layer(2)
        keras.layers.Dense(len(output[0]), activation="softmax")])  # output layer cu distributie de probabilitate

    retea.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])  # metrics se refera la output

    retea.fit(antrenament, output, epochs=1000)

    return retea


def prelucrare_input(s, cuvinte):

    # s reprezinta input-ul de la utilizator

    bag = [0 for _ in range(len(cuvinte))] # o lista cu 0 pt fiecare cuvant din lista de cuvinte

    # tokenizarea cuvintelor primite ca input
    lista_cuv_tokenizate = nltk.word_tokenize(s)
    lista_cuv_tokenizate = [stemmer.stem(cuv) for cuv in lista_cuv_tokenizate]

    for cuv_input in lista_cuv_tokenizate:
        for i, cuv in enumerate(cuvinte):
            if cuv_input == cuv:
                bag[i] = 1

    return bag


# clasa cu elementele de interfata a programului
class Interfata(FloatLayout):
    text = ObjectProperty(None)
    button = ObjectProperty(None)
    text_out = ObjectProperty(None)
    conversatie = []

    def __init__(self, **kwargs):
        super(Interfata, self).__init__(**kwargs)
        Window.bind(on_key_down = self._on_keyboard_down)

    def _on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):
        self.text.focus = True
        if keycode == 40:  # 40 - Tasta Enter
            self.apasare_buton()

    def apasare_buton(self):
        self.text_out.text = ("Tu: " + self.text.text)
        self.conversatie.append(self.text_out.text)

        mesaj = self.text.text
        raspuns = ChatBot().chat(mesaj)
        self.text.text = ""

        self.conversatie.append(raspuns)
        self.text_out.text = '\n'.join(i for i in self.conversatie)


# clasa principala a programului
class ChatBot(App):
    lista_intentii, etichete, cuvinte, antrenament, output = preluare_date()
    retea = construire_retea(antrenament, output)

    def build(self):
        return Interfata()

    def chat(self, input):
        rezultate = self.retea.predict([prelucrare_input(input, self.cuvinte)])  # de aici rezulta o lista cu probabilitati
                                                                                 # acestea reprezinta cat de probabil este ca
                                                                                 # input-ul sa apartina unei categorii
        rezultate_index = int(np.argmax(rezultate))  # din lista de mai sus se alege indexul categoriei cu cea mai mare
                                                     # probabilitate de a fi potrivita

        categorie = self.etichete[rezultate_index]  # categoria input-ului

        if rezultate[0][rezultate_index] > 0.8:
            for intentie in self.lista_intentii:
                if intentie[0] == categorie:
                    raspuns = intentie[2]

                    raspuns_final = ("Robotul: " + random.choice(raspuns))
        else:
            raspuns_final = "Robotul: Nu am inteles."

        return raspuns_final


if __name__ == '__main__':
    ChatBot().run()


