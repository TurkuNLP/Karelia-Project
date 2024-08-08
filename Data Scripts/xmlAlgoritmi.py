import xml.etree.ElementTree as ET
import json

# Alusta tyhjä lista tallentamaan tiedot JSON-muodossa
data_list = []

# Käy läpi jokainen XML-tiedosto
for volume, file_name in enumerate(['SK.xml', 'SK2.xml', 'SK3.xml', 'SK4.xml'], start=1):
    # Lue XML-tiedosto
    tree = ET.parse(file_name)
    root = tree.getroot()

    # Käy läpi jokainen PERSON-elementti
    for person in root.findall('PERSON'):
        # Alusta henkilön tiedot sanakirjana
        person_data = {}

        # Lisää indeksi, nimi ja sivu
        person_data["index"] = len(data_list) + 1
        person_data["name"] = person.get('name')
        person_data["page"] = person.get('approximated_page')
        person_data["volume"] = volume

        # Lisää kaikki tiedot
        data = ''.join(person.itertext()).strip()
        person_data["info"] = data

        # Lisää henkilön tiedot listaan
        data_list.append(person_data)

# Muuta tiedot JSON-muotoon
json_data = json.dumps(data_list, indent=4, ensure_ascii=False)

# Tallenna JSON-tiedosto
with open('yhdistetty_henkilotiedot.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

print("Yhdistetty JSON-tiedosto 'yhdistetty_henkilotiedot.json' on luotu onnistuneesti.")
