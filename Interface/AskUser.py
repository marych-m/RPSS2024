def create_pyramid(hoehe, farbe):
    for i in range(hoehe):
        abstaende = ' ' * (hoehe - i - 1)
        bloecke = farbe * (2 * i + 1)
        print(abstaende + bloecke + abstaende)

def main():
    hoehe = int(input("Geben Sie die Höhe der Pyramide ein (1, 2, 3 oder 4): "))
    farbe = input("Geben Sie die Farbe der Blöcke ein (orange=* oder schwarz=+): ")

    print("Hier ist Ihre Pyramide:")
    create_pyramid(hoehe, farbe)

if __name__ == "__main__":
    main()
