import json
import csv

# Contenu du fichier texte
document_content = '''[
    {
        "question": "Quels sont les types d'assurance automobile en Côte d'Ivoire ?",
        "answer": "En Côte d'Ivoire, il existe principalement trois types d'assurance automobile : l'assurance responsabilité civile (obligatoire), l'assurance tous risques et l'assurance tiers."
    },
    {
        "question": "Quelle est l'assurance automobile obligatoire en Côte d'Ivoire ?",
        "answer": "L'assurance responsabilité civile est obligatoire en Côte d'Ivoire. Elle couvre les dommages causés aux tiers en cas d'accident."
    },
    {
        "question": "Quel âge minimum pour souscrire une assurance automobile ?",
        "answer": "Généralement, il faut être âgé d'au moins 18 ans et posséder un permis de conduire valide pour souscrire une assurance automobile."
    },
    {
        "question": "Quels documents sont nécessaires pour souscrire une assurance ?",
        "answer": "Vous aurez besoin de : une copie de la carte grise, une copie du permis de conduire, une photocopie de la carte nationale d'identité, et un certificat d'immatriculation du véhicule."
    },
    {
        "question": "Combien coûte une assurance automobile en Côte d'Ivoire ?",
        "answer": "Le prix varie selon plusieurs facteurs : type de véhicule, âge du conducteur, couverture choisie. En moyenne, comptez entre 50 000 et 150 000 FCFA par an."
    },
    {
        "question": "Que couvre une assurance tous risques ?",
        "answer": "Une assurance tous risques couvre : les dommages causés à des tiers, les dommages à votre propre véhicule, le vol, l'incendie, et les dégâts naturels."
    },
    {
        "question": "Comment déclarer un sinistre ?",
        "answer": "Pour déclarer un sinistre : contactez votre assureur dans les 5 jours suivant l'incident, remplissez un constat amiable, et fournissez tous les documents nécessaires."
    },
    {
        "question": "Quelles sont les franchises en assurance automobile ?",
        "answer": "La franchise est le montant que vous devez payer en cas de sinistre. Elle varie généralement entre 50 000 et 200 000 FCFA selon le type de dommage."
    },
    {
        "question": "Un véhicule de plus de 10 ans peut-il être assuré ?",
        "answer": "Oui, mais les conditions et les tarifs peuvent être plus restrictifs. Certains assureurs proposent des formules spécifiques pour les véhicules anciens."
    },
    {
        "question": "Quels sont les délais de remboursement après un sinistre ?",
        "answer": "En général, le délai de remboursement est de 30 à 60 jours après la déclaration complète du sinistre et la fourniture de tous les documents nécessaires."
    },
    {
        "question": "Dois-je assurer un véhicule de fonction ?",
        "answer": "Si le véhicule appartient à l'entreprise, celle-ci doit le couvrir. Mais si vous l'utilisez à titre personnel, une assurance complémentaire est recommandée."
    },
    {
        "question": "Que faire en cas de vol de véhicule ?",
        "answer": "En cas de vol : déposez une plainte à la police immédiatement, informez votre assureur dans les 48h, et fournissez tous les documents de police et d'assurance."
    },
    {
        "question": "Les assurances couvrent-elles les conducteurs occasionnels ?",
        "answer": "Oui, mais il est important de les déclarer à votre assureur. Un conducteur occasionnel non déclaré pourrait invalider votre couverture en cas de sinistre."
    },
    {
        "question": "Comment réduire le coût de mon assurance automobile ?",
        "answer": "Plusieurs moyens : choisir une franchise plus élevée, installer des dispositifs antivol, avoir un bon historique de conduite, et comparer les offres des assureurs."
    },
    {
        "question": "Quelles sont les exclusions courantes en assurance automobile ?",
        "answer": "Les exclusions typiques incluent : conduite sous l'emprise de l'alcool, participation à des courses, dommages intentionnels, et usure normale du véhicule."
    },
    {
        "question": "Un étudiant peut-il souscrire une assurance automobile ?",
        "answer": "Oui, mais le tarif sera généralement plus élevé en raison du risque perçu comme plus important pour les jeunes conducteurs."
    },
    {
        "question": "Comment déclarer mes modifications de véhicule à l'assurance ?",
        "answer": "Tout changement important (modification du moteur, ajout d'équipements) doit être déclaré à votre assureur car cela peut impacter votre couverture et votre prime."
    },
    {
        "question": "L'assurance couvre-t-elle les accessoires ?",
        "answer": "Certaines assurances couvrent les accessoires, mais il faut généralement les déclarer explicitement et payer un supplément."
    },
    {
        "question": "Que faire si mon permis est suspendu ?",
        "answer": "Informez immédiatement votre assureur. La suspension peut affecter votre couverture et vos futures primes d'assurance."
    },
    {
        "question": "Les assurances couvrent-elles le transport de marchandises ?",
        "answer": "Non, une assurance automobile standard ne couvre pas le transport de marchandises. Une assurance professionnelle ou de transport est nécessaire."
    },
    {
        "question": "Comment fonctionne le bonus-malus ?",
        "answer": "Le système bonus-malus permet de moduler votre prime en fonction de votre historique de sinistres. Chaque année sans sinistre réduit votre prime."
    },
    {
        "question": "Dois-je assurer un véhicule non roulant ?",
        "answer": "Même si le véhicule ne roule pas, une assurance minimale contre le vol et l'incendie est recommandée."
    },
    {
        "question": "Quels véhicules sont plus difficiles à assurer ?",
        "answer": "Les véhicules de luxe, de sport, importés, et ceux avec des modifications importantes peuvent être plus difficiles et plus coûteux à assurer."
    },
    {
        "question": "Comment prouver mon antériorité d'assurance ?",
        "answer": "Conservez vos attestations d'assurance des années précédentes. Elles peuvent servir à obtenir de meilleurs tarifs."
    },
    {
        "question": "Les assurances couvrent-elles le prêt de véhicule ?",
        "answer": "La couverture dépend de votre contrat. Il est recommandé de vérifier les conditions spécifiques avec votre assureur."
    },
    {
        "question": "Qu'est-ce qu'une assurance au kilomètre ?",
        "answer": "C'est un type d'assurance où la prime est calculée en fonction du nombre de kilomètres réellement parcourus, offrant potentiellement des économies."
    },
    {
        "question": "Comment choisir mon assurance automobile ?",
        "answer": "Comparez les garanties, les prix, la réputation de l'assureur, les services annexes, et lisez attentivement les conditions générales."
    },
    {
        "question": "Les assurances couvrent-elles les catastrophes naturelles ?",
        "answer": "Certaines assurances incluent une garantie contre les catastrophes naturelles, mais vérifiez toujours les détails de votre contrat."
    },
    {
        "question": "Combien de temps mon assurance reste-t-elle valide ?",
        "answer": "Une assurance automobile est généralement valable un an, avec possibilité de renouvellement ou de résiliation selon les conditions contractuelles."
    }
]'''

# Parsing du contenu JSON
qa_data = json.loads(document_content)

# Chemins des fichiers
csv_filename = 'assurance_automobile_faq.csv'
txt_filename = 'assurance_automobile_faq.txt'
json_filename = 'assurance_automobile_faq.json'

# 1. Génération du fichier CSV
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Question', 'Réponse'])  # En-tête
    for qa in qa_data:
        writer.writerow([qa['question'], qa['answer']])

# 2. Génération du fichier TXT
with open(txt_filename, 'w', encoding='utf-8') as txtfile:
    for qa in qa_data:
        txtfile.write(f"Question: {qa['question']}\n")
        txtfile.write(f"Réponse: {qa['answer']}\n\n")

# 3. Génération du fichier JSON
with open(json_filename, 'w', encoding='utf-8') as jsonfile:
    json.dump(qa_data, jsonfile, ensure_ascii=False, indent=4)

print("Fichiers générés :")
print(f"1. {csv_filename}")
print(f"2. {txt_filename}")
print(f"3. {json_filename}")

# Affichage du contenu du CSV pour vérification
with open(csv_filename, 'r', encoding='utf-8') as csvfile:
    print("\nAperçu du contenu du CSV :")
    print(csvfile.read()[:1000])