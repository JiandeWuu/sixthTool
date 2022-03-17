# data

## RNALocate

- from: [RNALocate](http://www.rna-society.org/rnalocate/)
- Seq: `human_RNA_sequence.fasta`
- SubCellular_Localization:
  - `All RNA subcellular localization data.txt`
  - `All experimental RNA subcellular localization data.txt`（沒）

- SubCellular_Localization 對應類別

```txt
Exosome                           Cytosolic
Nucleus                           Nucleus
Nucleoplasm                       Nucleus
Chromatin                         Nucleus
Nucleolus                         Nucleus
Cytosol                           Cytosolic
Cytoplasm                         Cytosolic
Membrane                          Cytosolic
Insoluble cytoplasm               Cytosolic
Ribosome                          Cytosolic
Ribosome-free cytosol             Cytosolic
Nuclear                           Nucleus
Mitochondrion                     Cytosolic
Nuclear speckle                   Nucleus
Paraspeckle                       Nucleus
Paraspeckles in the nucleus       Nucleus
Endoplasmic reticulum             Cytosolic
Nuclear periphery                 Nucleus
Speckle periphery                 Nucleus
Soma                              
Microsome                         Cytosolic
Nuclear(exclusion from nucleoli)  Nucleus
Nuclear membrane                  Nucleus
Perinuclear                       Nucleus
Periphery of the nucleus          Nucleus
```

- All RNA subcellular localization data.txt / SubCellular_Localization 對應類別

```txt
            SubCellular_Localization   ALL  Has Seq
0                            Exosome  6412   3344.0
1                            Nucleus   865    751.0
2                        Nucleoplasm   437    416.0
3                          Chromatin   416    397.0
4                          Nucleolus   295    283.0
5                            Cytosol   289    278.0
6                          Cytoplasm   231    149.0
7                           Membrane   135    134.0
8                Insoluble cytoplasm    47     45.0
9                           Ribosome    34     13.0
10             Ribosome-free cytosol    16      6.0
11                           Nuclear     7      5.0
12                   Nuclear speckle     5      5.0
13                     Mitochondrion     5      4.0
14                       Paraspeckle     4      2.0
15       Paraspeckles in the nucleus     2      1.0
16             Endoplasmic reticulum     2      1.0
17  Nuclear(exclusion from nucleoli)     1      1.0
18                 Nuclear periphery     1      1.0
19                  Nuclear membrane     1      1.0
20                       Perinuclear     1      1.0
21          Periphery of the nucleus     1      1.0
22                         Microsome     1      1.0
23                              Soma     1      NaN
24                 Speckle periphery     1      NaN
```

照上面的規則取代後 count

```txt        
              ALL Has Seq
Cytosolic    7172    3972
Nucleus      2037    1867
Soma            1       1
```

Gene_ID 的 SubCellular_Localization 數量

```txt
      ALL  Has Seq
1    5675     2618
2     881      831
3       1        1
```

- All experimental RNA subcellular localization data.txt

```txt
S            SubCellular_Localization  ALL  Has Seq
0                            Nucleus  306    212.0
1                          Cytoplasm  231    149.0
2                            Exosome   35     16.0
3                           Ribosome   34     16.0
4                            Cytosol   22     13.0
5              Ribosome-free cytosol   16      6.0
6                            Nuclear    7      5.0
7                    Nuclear speckle    5      5.0
8                          Chromatin    5      4.0
9                      Mitochondrion    5      3.0
10                         Nucleolus    4      3.0
11                       Nucleoplasm    4      2.0
12                       Paraspeckle    4      1.0
13             Endoplasmic reticulum    2      1.0
14       Paraspeckles in the nucleus    2      1.0
15          Periphery of the nucleus    1      1.0
16                              Soma    1      1.0
17  Nuclear(exclusion from nucleoli)    1      1.0
18                       Perinuclear    1      1.0
19                 Nuclear periphery    1      1.0
20                  Nuclear membrane    1      1.0
21                         Microsome    1      NaN
22                 Speckle periphery    1      NaN
```

照上面的規則取代後 count

```txt
Nucleus      239
Cytosolic    203
Soma           1
```

Gene_ID 的 SubCellular_Localization 數量

```txt
1    195
2     82
```


