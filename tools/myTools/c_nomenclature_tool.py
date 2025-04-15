import sys

nucleotidemap = {
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'E': ['GAG', 'GAA'],
    'D': ['GAC', 'GAT'],
    'A': ['GCG', 'GCA', 'GCC', 'GCT'],
    'V': ['GTG', 'GTA', 'GTC', 'GTT'],
    'R': ['AGG', 'AGA', 'CGG', 'CGA', 'CGC', 'CGT'],
    'S': ['AGC', 'AGT', 'TCG', 'TCA', 'TCC', 'TCT'],
    'K': ['AAG', 'AAA'],
    'N': ['AAC', 'AAT'],
    'T': ['ACG', 'ACA', 'ACC', 'ACT'],
    'M': ['ATG'],
    'I': ['ATA', 'ATC', 'ATT'],
    'Q': ['CAG', 'CAA'],
    'H': ['CAC', 'CAT'],
    'P': ['CCG', 'CCA', 'CCC', 'CCT'],
    'L': ['CTG', 'CTA', 'CTC', 'CTT', 'TTG', 'TTA'],
    'W': ['TGG'],
    'C': ['TGC', 'TGT'],
    'Y': ['TAC', 'TAT'],
    'F': ['TTC', 'TTT'],
    'Stop': ['TGA', 'TAG', 'TAA']
}

if len(sys.argv) != 5:
    sys.exit(1)

cDNASeq = []
with open(sys.argv[1]) as f: 
    for line in f:
        line = line.rstrip()
        for letter in line:
            cDNASeq.append(letter)

listOfCodons = []
with open(sys.argv[2]) as f:
    listOfCodons = f.read().splitlines()

listOfChanges = []
with open(sys.argv[3]) as f:
    listOfChanges = f.read().splitlines()

index = 0

with open(sys.argv[4], "w") as file:
    for codonLocation in listOfCodons:
        arrayOfcNomenclature = []
        analyze = int(codonLocation) * 3
        threeBasePair = cDNASeq[analyze - 3:analyze]
        listOfPossibleChanges = nucleotidemap[listOfChanges[index]]
        for possibleChange in listOfPossibleChanges:
            indexOfDifferences = [i for i in range(len(possibleChange)) if possibleChange[i] != threeBasePair[i]]
            if len(indexOfDifferences) == 1:
                arrayOfcNomenclature.append("c." + str(analyze - (-1 * (indexOfDifferences[0] - 2))) + str(
                    threeBasePair[indexOfDifferences[0]]) + ">" + str(possibleChange[indexOfDifferences[0]]))
        index += 1
        file.write("\n".join(arrayOfcNomenclature) + "\n")
